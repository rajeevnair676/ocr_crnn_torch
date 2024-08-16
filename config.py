import os
import torch
from encode_decode import Tokenizer

VERSION = 3
DATA_VERSION = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 20
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 640
MAX_SEQUENCE_LENGTH = 79
WANDB = True

MODEL_INPUT_SHAPE = (640, 32, 3)

DATA_PATH = f'data\\Synthetic_Rec_En_V{DATA_VERSION}\\'
TRAIN_DATA_PATH = f"data\\Synthetic_Rec_En_V{DATA_VERSION}\\train"
TRAIN_IMAGES_PATH = os.path.join(TRAIN_DATA_PATH,'images')
LABEL_PATH = os.path.join(DATA_PATH,'train/gt.txt')

VAL_DATA_PATH = f'data\\Synthetic_Rec_En_V{DATA_VERSION}\\test'
VAL_LABEL_PATH = os.path.join(VAL_DATA_PATH,'gt.txt')
VAL_IMAGES_PATH = os.path.join(VAL_DATA_PATH,'images')

TEST_DATA_PATH = r'data\test_data\train\images'

MODEL_DIR = r'models\\torch\\'
OUTPUT_MODEL_PATH = os.path.join(
    MODEL_DIR, 
    f"OCR_CRNN_V{VERSION}_D{DATA_VERSION}_{IMAGE_HEIGHT}_{IMAGE_WIDTH}.pt"
)

TOKENIZER = Tokenizer(LABEL_PATH,MAX_SEQUENCE_LENGTH)
OPTIMIZER = "RMSprop"   #[Adam,RMSprop]

#CHECKPOINT CONFIGS
TORCH_MODEL_CKPT_PATH = f'checkpoints\\V{VERSION}_{DATA_VERSION}'
os.makedirs(TORCH_MODEL_CKPT_PATH,exist_ok=True)
RELOAD_CHECKPOINT = True
RELOAD_CHECKPOINT_PATH = f'checkpoints\V{VERSION}_{DATA_VERSION}\model_6.pt'




