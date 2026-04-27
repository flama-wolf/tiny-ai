import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'corpus.txt')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'model.pth')

# Hyperparameters
SEQ_LENGTH = 16
BATCH_SIZE = 8
HIDDEN_SIZE = 128
NUM_LAYERS = 2
LEARNING_RATE = 0.005
EPOCHS = 1
