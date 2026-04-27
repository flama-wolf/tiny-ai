import torch
import torch.nn.functional as F

from .config import HIDDEN_SIZE, NUM_LAYERS
from .model import CharLanguageModel
from utils.helpers import load_model

def generate_text(prompt: str, max_length: int = 100, temperature: float = 0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_data = load_model()
    if not model_data:
        return
        
    tokenizer = model_data['tokenizer']
    state_dict = model_data['model_state_dict']
    
    model = CharLanguageModel(tokenizer.vocab_size, HIDDEN_SIZE, NUM_LAYERS)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if not prompt:
        prompt = " "
        
    encoded_prompt = tokenizer.encode(prompt)
    if not encoded_prompt:
        # Fallback to first char in vocab if prompt contains only unknown characters
        encoded_prompt = [0]
        
    input_seq = torch.tensor([encoded_prompt], dtype=torch.long).to(device)
    
    hidden = None
    generated_chars = list(encoded_prompt)

    print(f"\n--- Generating text ---")
    print(f"Prompt: '{prompt}'")
    
    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(input_seq, hidden)
            
            # Get the predictions for the last character in the sequence
            last_char_logits = output[0, -1, :]
            
            # Apply temperature scaling
            scaled_logits = last_char_logits / temperature
            
            # Top-K sampling to prevent generating highly unlikely characters
            k = 10
            v, _ = torch.topk(scaled_logits, min(k, scaled_logits.size(0)))
            scaled_logits[scaled_logits < v[-1]] = -float('Inf')
            
            probs = F.softmax(scaled_logits, dim=0)
            
            # Sample from the distribution
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            generated_chars.append(next_char_idx)
            
            # Update input sequence for the next step (we only feed the generated char)
            input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)

    generated_text = tokenizer.decode(generated_chars)
    print(f"Generated text: \n{generated_text}\n")
    return generated_text
