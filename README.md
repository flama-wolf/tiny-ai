# Tiny AI: Character-Level Neural Language Model

This is a minimal, from-scratch character-level language model built entirely in PyTorch. It trains on raw text data, learns the relationships between characters, and predicts the next character to generate text autoregressively.

Currently, it is set up to train on a small subset of the *Tiny Shakespeare* dataset.

## Requirements

You only need standard Python and a couple of essential deep learning libraries.

```bash
pip install torch tqdm
```

*(Note: PyTorch will run significantly faster if you install it with CUDA support for your GPU, but it will work fine on CPU for this small model).*

## Project Structure

```text
tiny_ai/
│
├── data/
│   └── corpus.txt       # The raw text data used for training (currently Shakespeare)
│
├── src/
│   ├── tokenizer.py     # Converts characters to integers and vice versa
│   ├── dataset.py       # PyTorch Dataset for loading sequences
│   ├── model.py         # The LSTM Neural Network architecture
│   ├── train.py         # Training loop with loss tracking and checkpointing
│   ├── generate.py      # Autoregressive text generation with Top-K sampling
│   └── config.py        # Hyperparameters (epochs, learning rate, sequence length, etc.)
│
├── utils/
│   └── helpers.py       # Functions to save and load the model and tokenizer
│
└── main.py              # The Command Line Interface (CLI)
```

## How to Run

There are two main modes: `train` and `generate`. Both are accessible via the `main.py` entry point.

### 1. Training the Model

Before you can generate text, you must train the model so it can learn from `data/corpus.txt`.

```bash
python main.py train
```

- This will start training based on the hyperparameters defined in `src/config.py`.
- It tracks the loss and automatically saves the best performing model weights to `model.pth`.
- By default, `EPOCHS` is set to 1 for quick testing. You can increase it to `100` in `src/config.py` for better results.

### 2. Generating Text

Once the model is trained (and `model.pth` exists in the root folder), you can ask it to generate text based on a prompt.

```bash
python main.py generate --prompt "The " --length 100 --temperature 0.8
```

**Arguments:**
- `--prompt`: The starting characters you want the model to complete (e.g., `"ROMEO: "`).
- `--length`: The number of characters to generate.
- `--temperature`: Controls the randomness. `1.0` is standard, `0.8` is slightly more conservative/safe, and higher values like `1.5` make it much more chaotic.

## Customizing

- **Change the Dataset**: Just paste any text you want into `data/corpus.txt`. The model will automatically learn its vocabulary and syntax on the next training run.
- **Tweak the AI**: Open `src/config.py` to change the Sequence Length (`SEQ_LENGTH`), the size of the neural network (`HIDDEN_SIZE`), or the learning rate (`LEARNING_RATE`).
