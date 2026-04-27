import argparse
from src.train import train_model
from src.generate import generate_text

def main():
    parser = argparse.ArgumentParser(description="Tiny AI: Character-level Neural Language Model")
    parser.add_argument('mode', choices=['train', 'generate'], help="Mode to run: 'train' or 'generate'")
    parser.add_argument('--prompt', type=str, default="The ", help="Prompt text for generation")
    parser.add_argument('--length', type=int, default=100, help="Number of characters to generate")
    parser.add_argument('--temperature', type=float, default=0.8, help="Generation temperature (higher = more random)")
    
    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    elif args.mode == 'generate':
        generate_text(prompt=args.prompt, max_length=args.length, temperature=args.temperature)

if __name__ == "__main__":
    main()
