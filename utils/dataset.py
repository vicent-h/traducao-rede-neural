from torch.utils.data import Dataset
import torch


class TranslateDataset(Dataset):
    def __init__(self, tokens_src, tokens_tgt, invert_src=False, max_len=45):
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

        src = list(src[:self.max_len])
        tgt = list(tgt[:self.max_len])


        if len(src) < self.max_len:
            src += [0] * (self.max_len - len(src))
        if len(tgt) < self.max_len:
            tgt += [0] * (self.max_len - len(tgt))

        return torch.tensor(src), torch.tensor(tgt)