import os
import torch
from encode_decode import Tokenizer

VERSION = 2
DATA_VERSION = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
# VOCAB_SIZE = 95
NUM_EPOCHS = 10
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 640
MAX_SEQUENCE_LENGTH = 79
WANDB = True

MODEL_INPUT_SHAPE = (640, 32, 3)

DATA_PATH = f'E:\\EFR\\Projects\\OCR\\Experiments\\data\\Synthetic_Rec_En_V{DATA_VERSION}\\'
TRAIN_DATA_PATH = f"E:\\EFR\\Projects\\OCR\\Experiments\\data\\Synthetic_Rec_En_V{DATA_VERSION}\\train"
TRAIN_IMAGES_PATH = os.path.join(TRAIN_DATA_PATH,'images')
LABEL_PATH = os.path.join(DATA_PATH,'train/gt.txt')

VAL_DATA_PATH = f'E:\\EFR\\Projects\\OCR\\Experiments\\data\\Synthetic_Rec_En_V{DATA_VERSION}\\test'
VAL_LABEL_PATH = os.path.join(VAL_DATA_PATH,'gt.txt')
VAL_IMAGES_PATH = os.path.join(VAL_DATA_PATH,'images')

TEST_DATA_PATH = r'E:\EFR\Projects\OCR\Experiments\CRNN\torch\Synthetic_Rec_En_V5\test\images'

MODEL_DIR = r'models\\torch\\'
OUTPUT_MODEL_PATH = os.path.join(
    MODEL_DIR, 
    f"OCR_CRNN_V{VERSION}_D{DATA_VERSION}_{IMAGE_HEIGHT}_{IMAGE_WIDTH}.pt"
)

#CHECKPOINT CONFIGS
TORCH_MODEL_CKPT_PATH = f'E:\\EFR\\Projects\\OCR\\Experiments\\CRNN\\torch\\checkpoints\\V{VERSION}_{DATA_VERSION}'
os.makedirs(TORCH_MODEL_CKPT_PATH,exist_ok=True)
RELOAD_CHECKPOINT = True
RELOAD_CHECKPOINT_PATH = r'E:\EFR\Projects\OCR\Experiments\CRNN\torch\checkpoints\V2_2\model_5.pt'

TOKENIZER = Tokenizer(LABEL_PATH,MAX_SEQUENCE_LENGTH)
OPTIMIZER = "Adam"

# #KERAS CONFIG
# KERAS_VERSION = 1
# KERAS_INPUT_SHAPE = (24, 416, 3)
# KERAS_MAX_SEQ_LENGTH = 50
# KERAS_BATCH_SIZE = 32
# DF_SAVE_PATH = f'E:\\EFR\\Projects\\OCR\\Experiments\\data\\Synthetic_Rec_En_V{DATA_VERSION}\\'
# KERAS_CHECKPOINT_PATH = f'E:\\EFR\\Projects\\OCR\\Experiments\\CRNN\\keras_checkpoints\\V{KERAS_VERSION}\\best_model.keras'

