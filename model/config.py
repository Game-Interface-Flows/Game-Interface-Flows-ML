RANDOM_SEED = 42

# DataModule
DATA_PATH = "data/"
BATCH_SIZE = 32
NUM_WORKERS = 4
TRAIN_TEST_RATIO = 0.9
TRAIN_VAL_RATIO = 0.9

# Hyperparametres
MIN_EPOCHS = 1
MAX_EPOCHS = 5
LEARNING_RATE = 0.001

# Compute related
ACCELERATOR = "mps"
DEVICES = 1
PRECISION = 16

# Save Path
MODEL_PATH = "../api/model.pt"
