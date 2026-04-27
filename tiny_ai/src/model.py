import torch
import torch.nn as nn

class CharLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # We use an embedding layer to learn character representations
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)
        
        # If no hidden state is provided, LSTM initializes to zeros internally
        out, hidden = self.lstm(embedded, hidden)
        
        # We only need the output for cross entropy loss or generation
        # out shape: (batch_size, seq_length, hidden_size)
        out = self.fc(out)
        return out, hidden
