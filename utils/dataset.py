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

        src = list(src[:self.max_len])
        tgt = list(tgt[:self.max_len])


        if len(src) < self.max_len:
            src += [0] * (self.max_len - len(src))
        if len(tgt) < self.max_len:
            tgt += [0] * (self.max_len - len(tgt))

        return torch.tensor(src), torch.tensor(tgt)


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
            # e que o input_tensor tem a propriedade de tamanho/shape
            input_tensor = self.tokens_src[idx]
            length = len(input_tensor)
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