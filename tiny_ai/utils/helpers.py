import torch
import os
from src.config import MODEL_SAVE_PATH

def save_model(model, tokenizer):
    """Saves the model state dictionary and tokenizer."""
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer
    }, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

def load_model():
    """Loads the model and tokenizer from disk."""
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: {MODEL_SAVE_PATH} not found. Please train the model first.")
        return None
        
    print(f"Loading model from {MODEL_SAVE_PATH}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
    return checkpoint
