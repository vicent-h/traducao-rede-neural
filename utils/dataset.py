from torch.utils.data import Dataset, Sampler
import random
import torch
from torch.utils.data import BatchSampler

class TranslateDataset(Dataset):
    def __init__(self, tokens_src, tokens_tgt, invert_src=False, max_len=60):
        self.tokens_src = tokens_src
        self.tokens_tgt = tokens_tgt
        self.invert_src = invert_src
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens_src)
    
    def __getitem__(self, idx):
        
        if self.invert_src:
            src = self.tokens_src[idx][::-1].copy()
            tgt = self.tokens_tgt[idx]
        else:            
            src = self.tokens_src[idx]
            tgt = self.tokens_tgt[idx]

        src = src[1:]  # Remove <BOS>

        # Trunca para um comprimento máximo, mas NÃO aplica padding aqui.
        src = list(src[:self.max_len])
        tgt = list(tgt[:self.max_len])

        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


class CurriculumLengthSampler(Sampler):
    def __init__(self, tokens_src, tokens_tgt, len_tokens, curriculum_levels):
        self.tokens_src = tokens_src
        self.tokens_tgt = tokens_tgt
        self.len_tokens = len_tokens
        self.max_length = curriculum_levels[0]["max_len"]  # Começa com o primeiro nível do currículo
        self.curriculum_levels = curriculum_levels
        self.step_count = 0
        self.level_index = 0
        self.level_actual = curriculum_levels[self.level_index]
        
        # Passo 1: Mapeamento inicial na inicialização
        # Varre o dataset e guarda o tamanho de cada sequência pelo seu índice
        self.indices_with_lengths = []
        for idx in range(len(self.tokens_src)):
            # Assumindo que o seu dataset retorna (input_tensor, target_tensor)
            # O dataset remove o <BOS> em __getitem__, então considera-se len-1
            input_tensor = self.tokens_src[idx]
            length = max(0, len(input_tensor) - 1)
            self.indices_with_lengths.append((idx, length))

    def step(self):
        self.step_count += 1
        level_actual = self.curriculum_levels[self.level_index]
        if self.step_count > level_actual["max_step"] and self.level_index < len(self.curriculum_levels) - 1:
            self.level_index += 1
            self.level_actual = self.curriculum_levels[self.level_index]

    def get_max_length(self):
        return self.level_actual["max_len"]
    
    def get_batch_size(self):
        return self.level_actual["batch_size"]
    
    def get_accum_steps(self):
        return self.level_actual["accum_steps"]

    def __iter__(self):
        # Passo 2: O Fluxo de Filtragem Passiva
        # Filtra apenas os índices cujas sentenças respeitam o limite atual
        valid_indices = [
            idx for idx, length in self.indices_with_lengths 
            if length <= self.get_max_length()
        ]
        
        # Garante o embaralhamento (shuffling) dentro do grupo de dados fáceis
        random.shuffle(valid_indices)
        
        return iter(valid_indices)

    def __len__(self):
        # O tamanho do sampler muda dinamicamente conforme mais dados são liberados
        valid_indices = [
            idx for idx, length in self.indices_with_lengths 
            if length <= self.get_max_length()
        ]
        return len(valid_indices)

class DynamicBatchSampler(BatchSampler):
    def __init__(self, sampler: CurriculumLengthSampler):
        self.sampler = sampler
        self.drop_last = False  # Adicionar esta linha
        self.batch_size = self.sampler.get_batch_size()  # Inicializa com o batch size do primeiro nível do currículo

    def step(self):
        """Método para avançar o currículo e potencialmente alterar o tamanho do lote"""
        self.sampler.step()
        self.batch_size = self.sampler.get_batch_size()  # Atualiza o batch size conforme o currículo avança
    
    def get_max_length(self):
        return self.sampler.get_max_length()
    
    def get_batch_size(self):
        return self.sampler.get_batch_size()
    
    def get_accum_steps(self):
        return self.sampler.get_accum_steps()

    def __iter__(self):
        batch = []
        # Puxa os índices filtrados que vêm do seu CurriculumLengthSampler
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.get_batch_size():
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch


class DynamicCollator:
    """Collator que aplica padding dinâmico por batch usando o max_len atual do sampler/batch_sampler."""
    def __init__(self, batch_sampler, pad_value: int = 0):
        # batch_sampler pode ser uma instância de DynamicBatchSampler
        self.batch_sampler = batch_sampler
        self.pad_value = pad_value

    def __call__(self, batch):
        # batch: list de (src_tensor, tgt_tensor) com comprimentos variáveis
        max_len = self.batch_sampler.get_max_length()

        srcs = []
        tgts = []
        for src, tgt in batch:
            # converter para lista Python para facilitar trunc/pad
            src_list = src.tolist() if isinstance(src, torch.Tensor) else list(src)
            tgt_list = tgt.tolist() if isinstance(tgt, torch.Tensor) else list(tgt)

            # Trunca (por segurança) e então aplica padding para max_len
            src_list = src_list[:max_len]
            tgt_list = tgt_list[:max_len]

            if len(src_list) < max_len:
                src_list += [self.pad_value] * (max_len - len(src_list))
            if len(tgt_list) < max_len:
                tgt_list += [self.pad_value] * (max_len - len(tgt_list))

            srcs.append(torch.tensor(src_list, dtype=torch.long))
            tgts.append(torch.tensor(tgt_list, dtype=torch.long))

        return torch.stack(srcs), torch.stack(tgts)