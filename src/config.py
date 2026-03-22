import torch

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "data/Train"
TEST_DIR = "data/Test"
TEST_CSV = "data/Test.csv"
DATA_ROOT = "data"

IMG_SIZE = 32
NUM_CLASSES = 43
BATCH_SIZE = 128
EPOCHS = 15
FINETUNE_EPOCHS = 3
LR = 1e-3
FINETUNE_LR = 1e-4
PRUNE_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]
NUM_WORKERS = 2

RESULTS_CSV = "outputs/gtsrb_pruning_results.csv"
PRUNING_PLOT = "outputs/accuracy_vs_pruning_ratio.png"
TRAINING_PLOT = "outputs/baseline_training_curves.png"
