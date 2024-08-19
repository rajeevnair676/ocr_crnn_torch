import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import CustomDataset
from model2 import CRNNModel,BidirectionalLSTM
from torch.optim import Adam,RMSprop
from encode_decode import Tokenizer
from tqdm import tqdm
import os
from metric import WERMetric,CharacterErrorRate,ctc_decoder
import wandb
import config


os.makedirs(config.MODEL_DIR,exist_ok=True)

if config.WANDB:
    wandb.login()
    wandb.init(config={"learning_rate": config.LEARNING_RATE,
        "batch_size":config.BATCH_SIZE,
        "model_input_shape":config.MODEL_INPUT_SHAPE,
        "architecture": "CRNN",
        "dataset": f"Synthetic_Rec_En_V{config.DATA_VERSION}",
        "epochs": config.NUM_EPOCHS,
        # "model_file_used": MODEL_FILE_USED,
        "losses":"CTC",
        "optimizer":config.OPTIMIZER,
        "metric":"WER",
        "reduction":"Mean",
        "max_seq_length":config.MAX_SEQUENCE_LENGTH,
        "clip_grad_norm":True,
        # "grad_scaler":True,
        # "momentum":MOMENTUM,
        # "weight decay": WEIGHT_DECAY
        },
        name = f"OCR_CRNN_V{config.VERSION}_D{config.DATA_VERSION}_{config.IMAGE_HEIGHT}_{config.IMAGE_WIDTH}",
        project = "OCR_CRNN"
        )

def collate_func(batch):
    # (B, [images, encodings, lengths])
    images, encodings, lengths = zip(*batch)
    images = torch.stack(images)
    encodings = torch.cat(encodings, dim=-1)
    lengths = torch.cat(lengths, dim=-1)
    return images, encodings, lengths

tokenizer = Tokenizer(config.LABEL_PATH,config.MAX_SEQUENCE_LENGTH)

#Loading the train and validation dataset and building the respective dataloaders
train_dataset = CustomDataset(config.DATA_PATH, config.MODEL_INPUT_SHAPE[1], config.MODEL_INPUT_SHAPE[0], tokenizer)
train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True, collate_fn=collate_func)

val_dataset = CustomDataset(config.DATA_PATH,config.MODEL_INPUT_SHAPE[1],config.MODEL_INPUT_SHAPE[0],tokenizer, train=False)
val_loader = DataLoader(val_dataset,config.BATCH_SIZE, shuffle=True, collate_fn=collate_func)

#Initializing losses and WER metrics
blank_ind = config.TOKENIZER.vocab_size -1
loss_ctc = nn.CTCLoss(reduction="mean",zero_infinity=True,blank=blank_ind).to(config.DEVICE)
cer_metric = WERMetric()

#Initializing the model and optimizer
model = CRNNModel(config.TOKENIZER.vocab_size).to(config.DEVICE)

if config.OPTIMIZER=="Adam":
    optimizer = Adam(model.parameters(),lr=config.LEARNING_RATE)
elif config.OPTIMIZER=="RMSprop":
    optimizer = RMSprop(model.parameters(),lr=config.LEARNING_RATE) 

ckpt_epoch = 1
if config.RELOAD_CHECKPOINT:
    checkpoint = torch.load(config.RELOAD_CHECKPOINT_PATH,map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ckpt_epoch = checkpoint['epoch']+1
    print(f"Loaded the model checkpoint successfully and training would resume from epoch {ckpt_epoch}")
    # loss = checkpoint['loss']

prev_cer = 0.0
prev_epoch_loss = 0.0
for epoch in range(ckpt_epoch,config.NUM_EPOCHS+1):
    train_epoch_loss=0.0
    train_cer = 0.0
    print(f"Epoch: {epoch}")
    model.train()
    pbar = tqdm(enumerate(train_loader),total=len(train_loader))
    for i, (image, label, label_lens) in pbar:
        image = image.to(config.DEVICE)
        label = label.to(config.DEVICE)
        label_lens = label_lens.to(config.DEVICE)

        optimizer.zero_grad()
        log_probs = model(image)
        T, B, C = log_probs.shape   #(Seq length, Batch size, No of Classes)

        input_lengths = torch.LongTensor([T]*B)  #[T,T,T.......B]

        loss = loss_ctc(log_probs, label, input_lengths, torch.flatten(label_lens))
        out = torch.argmax(log_probs,dim=2).permute(1,0)

        out_decode = ctc_decoder(log_probs)
        # out_decode = tokenizer.batch_decode(out)

        labels = tokenizer.decode1D(label,label_lens)
        cer = cer_metric(out_decode,labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),5)

        optimizer.step()

        train_epoch_loss+=loss.item()
        train_cer += cer

        # desc_ = f"Train_Loss: {round(loss.item(),2)}:,:Train_CER: {round(cer,2)}:"
        # # pbar.set_description(desc_)
        # # pbar.update(1)

    train_epoch_loss = train_epoch_loss / len(train_loader)
    train_cer = train_cer / len(train_loader)
    print(f'Train Loss: {train_epoch_loss}')
    print(f'Train CER: {train_cer}')

    if config.WANDB:
        wandb.log({"CER":train_cer, "loss": train_epoch_loss})

    ckpt_path = os.path.join(config.TORCH_MODEL_CKPT_PATH,f'model_{epoch}.pt')
    torch.save({'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':train_epoch_loss},
                ckpt_path)
    print(f"Saved model checkpoint for epoch {epoch} to '{ckpt_path}'")

    with torch.no_grad():
        model.eval()
        val_epoch_loss=0.0
        val_cer = 0.0
        pbar = tqdm(enumerate(val_loader),total=len(val_loader))
        # val_loss_epoch,val_accuracy_epoch= 0,0
        for step,(val_img,val_label,val_label_lens) in pbar:
            val_img = val_img.to(config.DEVICE)
            val_label = val_label.to(config.DEVICE)
            val_label_lens = val_label_lens.to(config.DEVICE)

            log_probs = model(val_img)
            T, B, C = log_probs.shape
            val_input_legths = torch.LongTensor([T]*B)

            val_loss = loss_ctc(log_probs, val_label, val_input_legths, torch.flatten(val_label_lens))

            val_out = torch.argmax(log_probs,dim=2).permute(1,0)

            val_out_decode = ctc_decoder(log_probs)
            # val_out_decode = tokenizer.batch_decode(val_out)

            val_labels = tokenizer.decode1D(val_label,val_label_lens)

            cer = cer_metric(val_out_decode,val_labels)

            val_epoch_loss += val_loss.item()
            val_cer += cer

            # desc_ = f"Loss: {round(val_loss.item(),2)}:,:CER: {round(cer,2)}:"
            # pbar.set_description(desc_)

        val_epoch_loss = val_epoch_loss / len(val_loader)
        val_cer = val_cer / len(val_loader)

    print(f"Validation loss: {val_epoch_loss}:,:CER: {val_cer:.4f}")
    print()

    if val_epoch_loss < prev_epoch_loss:
        torch.save(model, config.OUTPUT_MODEL_PATH)
        print(f"Model saved to {config.OUTPUT_MODEL_PATH}")
        print()
    prev_epoch_loss = val_epoch_loss

if config.WANDB:
    wandb.finish()

print("Training complete")