import os
import numpy as np
import tensorflow as tf
from PIL import Image

# +-* + () + 10 digit + blank + space
num_classes = 3 + 2 + 10 + 1 + 1

maxPrintLen = 100

# num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size)

charset = '0123456789+-*()'
encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN




class DataIterator:
    def __init__(self, data_dir):
        self.image = []
        self.labels = []
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                image_name = os.path.join(root, file_path)
                #im = cv2.imread(image_name, 0).astype(np.float32)/255.
                #print(image_name)
                im = np.array(Image.open(image_name).convert("L")).astype(np.float32)/255. #mycode
                # resize to same height, different width will consume time on padding
                # im = cv2.resize(im, (image_width, image_height))
                im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                self.image.append(im)

                # image is named as /.../<folder>/00000_abcd.png
                code = image_name.split('/')[-1].split('_')[1].split('.')[0]
                code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code)]
                self.labels.append(code)







###
shuffle_data = True  # shuffle the addresses before saving
cat_dog_train_path = 'Cat vs Dog/train/*.jpg'
# read addresses and labels from the 'train' folder
addrs = glob.glob(cat_dog_train_path)
labels = [0 if 'cat' in addr else 1 for addr in addrs]  # 0 = Cat, 1 = Dog
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)
    
# Divide the hata into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]
val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

###
    def __init__(self, data_dir):
        self.image = []
        self.labels = []
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                image_name = os.path.join(root, file_path)
                #im = cv2.imread(image_name, 0).astype(np.float32)/255.
                #print(image_name)
                im = np.array(Image.open(image_name).convert("L")).astype(np.float32)/255. #mycode
                # resize to same height, different width will consume time on padding
                # im = cv2.resize(im, (image_width, image_height))
                im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                self.image.append(im)

                # image is named as /.../<folder>/00000_abcd.png
                code = image_name.split('/')[-1].split('_')[1].split('.')[0]
                code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code)]
                self.labels.append(code)