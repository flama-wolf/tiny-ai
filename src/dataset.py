import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, encoded_text: list[int], seq_length: int):
        self.encoded_text = encoded_text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx):
        # Input sequence
        x_chunk = self.encoded_text[idx:idx + self.seq_length]
        # Target sequence (shifted by one character)
        y_chunk = self.encoded_text[idx + 1:idx + self.seq_length + 1]
        
        x = torch.tensor(x_chunk, dtype=torch.long)
        y = torch.tensor(y_chunk, dtype=torch.long)
        return x, y
