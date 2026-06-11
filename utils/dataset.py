from torch.utils.data import Dataset, Sampler
import random
import torch


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
    def __init__(self, tokens_src, tokens_tgt, len_tokens, list_dicts):
        self.tokens_src = tokens_src
        self.tokens_tgt = tokens_tgt
        self.len_tokens = len_tokens
        self.max_length = list_dicts[0]["max_len"]  # Começa com o primeiro nível do currículo
        self.curriculum_levels = list_dicts
        self.step_count = 0
        
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
        # Verifica se é hora de avançar para o próximo nível do currículo
        for level in self.curriculum_levels:
            if self.step_count >= level["max_step"]:
                self.max_length = level["max_len"]
                print(f"Currículo atualizado: Agora treinando com sentenças de até {self.max_length} tokens.")
                break


    def __iter__(self):
        # Passo 2: O Fluxo de Filtragem Passiva
        # Filtra apenas os índices cujas sentenças respeitam o limite atual
        valid_indices = [
            idx for idx, length in self.indices_with_lengths 
            if length <= self.max_length
        ]
        
        # Garante o embaralhamento (shuffling) dentro do grupo de dados fáceis
        random.shuffle(valid_indices)
        
        return iter(valid_indices)

    def __len__(self):
        # O tamanho do sampler muda dinamicamente conforme mais dados são liberados
        valid_indices = [
            idx for idx, length in self.indices_with_lengths 
            if length <= self.max_length
        ]
        return len(valid_indices)