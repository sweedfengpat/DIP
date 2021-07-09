import sys
import os
# for array operations
import numpy as np
# tensorflow framework
import tensorflow as tf
# keras API for deep learning
from tensorflow import keras
# for image visulaizations
import matplotlib.pyplot as plt
# for legends and other supporting functionalities
import matplotlib as mpl
# for viewing iteration status
from tqdm import tqdm
from tensorflow_examples.models.pix2pix import pix2pix

from sklearn.model_selection import train_test_split
# a list to collect paths of 1000 images

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

image_path = []
for root, dirs, files in os.walk('C:\\Users\\PAT\\Desktop\\dip\\DIP\\IMAGES'):
    # iterate over 1000 images
    for file in files:
        # create path
        path = os.path.join(root,file)
        # add path to list
        image_path.append(path)

print("Hello")
print(len(image_path))


# a list to collect paths of 1000 masks
mask_path = []
for root, dirs, files in os.walk('C:\\Users\\PAT\\Desktop\\dip\\DIP\\MASKS'):
    #iterate over 1000 masks
    for file in files:
        # obtain the path
        path = os.path.join(root,file)
        # add path to the list
        mask_path.append(path)
len(mask_path)
print(image_path[:5])
print(mask_path[:5])

image_path.sort()
mask_path.sort()

# create a list to store images
images = []
# iterate over 1000 image paths
for path in tqdm(image_path):
    # read file
    file = tf.io.read_file(path)
    # decode png file into a tensor
    image = tf.image.decode_png(file, channels=3, dtype=tf.uint8)
    # append to the list
    images.append(image)

# create a list to store masks
masks = []
# iterate over 1000 mask paths
for path in tqdm(mask_path):
    # read the file
    file = tf.io.read_file(path)
    # decode png file into a tensor
    mask = tf.image.decode_png(file, channels=1, dtype=tf.uint8)
    # append mask to the list
    masks.append(mask)

labels = ['null','accessories','bag','belt','blazer','blouse','bodysuit','boots','bra','bracelet','cape','cardigan','clogs','coat','dress','earrings','flats','glasses','gloves','hair','hat','heels','hoodie','intimate','jacket','jeans','jumper','leggings','loafers','necklace','panties','pants','pumps','purse','ring','romper','sandals','scarf','shirt','shoes','shorts','skin','skirt','sneakers','socks','stockings','suit','sunglasses','sweater','sweatshirt','swimwear','t-shirt','tie','tights','top','vest','wallet','watch','wedges']

print(list(images))
plt.figure(figsize=(16,5))
for i in range(1,4):
    plt.subplot(1,3,i)
    print(i)
    img = images[i]
    plt.imshow(img)
    plt.colorbar()
    plt.axis('off')
plt.show()
NORM = mpl.colors.Normalize(vmin=0, vmax=58)

# plot masks
plt.figure(figsize=(16,5))
for i in range(1,4):
    plt.subplot(1,3,i)
    img = masks[i]
    plt.axis('off')
    sc = plt.imshow(img, cmap='jet', norm=NORM)
    cbar = plt.colorbar(sc, ticks=range(len(labels)))
    cbar.ax.set_yticklabels(labels)

plt.show()

