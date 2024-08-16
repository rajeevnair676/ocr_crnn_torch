import os

# with open(label_file,'r') as f:
#     for line in f.readlines():
#         img_labels.append((line.split('\t')[0],line.split('\t')[1]))

DATA_PATH = r'Synthetic_Rec_En_V5'

img_path = r'Synthetic_Rec_En_V5'

img_labels = []
label_file = os.path.join(img_path,f'train/gt.txt')
        
with open(label_file,'r') as f:
    for line in f.readlines():
        img_labels.append((line.split('\t')[0],line.split('\t')[1].rstrip()))


print(img_labels[:5])