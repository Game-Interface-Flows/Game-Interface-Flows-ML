RANDOM_SEED = 42

# DataModule
DATA_PATH = "data/"
BATCH_SIZE = 64
NUM_WORKERS = 4
TRAIN_TEST_RATIO = 0.7
TRAIN_VAL_RATIO = 0.2

# Hyperparametres
MIN_EPOCHS = 1
MAX_EPOCHS = 10
LEARNING_RATE = 0.001

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 16

# Save Path
MODEL_PATH = "../api/model.pt"
