import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import DATA_PATH, SEQ_LENGTH, BATCH_SIZE, HIDDEN_SIZE, NUM_LAYERS, LEARNING_RATE, EPOCHS
from .tokenizer import CharTokenizer
from .dataset import CharDataset
from .model import CharLanguageModel
from utils.helpers import save_model

def train_model():
    print("Loading data...")
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found.")
        return

    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    
    encoded_text = tokenizer.encode(text)
    if len(encoded_text) <= SEQ_LENGTH:
        print("Text corpus is too short for the given SEQ_LENGTH.")
        return

    dataset = CharDataset(encoded_text, SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CharLanguageModel(tokenizer.vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training on {device}...")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output, _ = model(batch_x)
            
            # Reshape output and target to match CrossEntropyLoss expectations
            loss = criterion(output.view(-1, tokenizer.vocab_size), batch_y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, tokenizer)

    print("Training finished!")
