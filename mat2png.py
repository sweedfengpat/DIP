import os
import cv2
from scipy import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import tensorflow as tf


images = []

for i in range(1,1001):
    url = './clothing-co-parsing/photos/%04d.jpg'%(i)
    # use OpenCV for lossless reading
    img = cv2.imread(url, 1)
    # convert BGR image into RGB image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert into a tensor
    img = tf.convert_to_tensor(img)
    # resize the image by either cropping or padding with zeros
    img = tf.image.resize_with_crop_or_pad(img,825,550)
    # add to the list
    images.append(img)

segmentations = []
# read 1000 files
for i in range(1,1001):
    url = './clothing-co-parsing/annotations/pixel-level/%04d.mat'%(i)
    # read MATLAB file as image
    file = io.loadmat(url)
    # convert into a tensor
    mask = tf.convert_to_tensor(file['groundtruth'])
    # resize expects 3D image, but we got 2D grayscale image 
    # so expand dimensions
    mask = tf.expand_dims(mask,-1)
    # resize by either cropping excess or padding with zeros
    mask = tf.image.resize_with_crop_or_pad(mask,825,550)
    # append the mask image to the list
    segmentations.append(mask)

    
label_url = './clothing-co-parsing/label_list'
# read the labels list MATLAB file
label_file = io.loadmat(label_url)['label_list']
# what is its shape?
label_file.shape
label_file = np.squeeze(label_file)
label_file.shape
labels = [label[0].astype(str) for label in label_file]
plt.figure(figsize=(7,7))
example_seg = segmentations[0]
# obtain unique values (the class numbers)
annotations = np.unique(example_seg.numpy().ravel())
# read the names
names = [labels[a] for a in annotations]
# the values range from 0 to 58, hence normalize for homogeneity
NORM = mpl.colors.Normalize(vmin=0, vmax=58)
for i in range(1,9):
    # display a mask with legends
    plt.subplot(4,4,2*i-1)
    example_seg = segmentations[i]
    annotations = np.unique(example_seg.numpy().ravel())
    names = [labels[a] for a in annotations]
    NORM = mpl.colors.Normalize(vmin=0, vmax=58)
    

for i in tqdm(range(1000)):
    img = images[i]
    # encode into PNG
    img = tf.io.encode_png(img)
    # create a path
    path = os.path.join('IMAGES','img_%04d.png'%(i+1))
    file_name = tf.constant(path)
    # write the PNG file
    tf.io.write_file(file_name, img)

PNG_IMAGES = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if 'img_' in file:
            # get image paths
            path = os.path.join(root, file)
            PNG_IMAGES.append(path)

PNG_IMAGES.sort()
print(PNG_IMAGES[:10])

im = tf.io.read_file(PNG_IMAGES[102])
# decode it into a tensor
dec = tf.io.decode_png(im, channels=3, dtype=tf.dtypes.uint8)
# visualize it
plt.figure(figsize=(4,7))
plt.imshow(dec)
plt.show()

for i in tqdm(range(1000)):
    seg = segmentations[i]
    # encode the tensor into PNG
    seg = tf.io.encode_png(seg)
    # create a path to write
    path = os.path.join('MASKS','seg_%04d.png'%(i+1))
    file_name = tf.constant(path)
    # write the PNG file
    tf.io.write_file(file_name, seg)

PNG_MASKS = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if 'seg_' in file:
            # obtain the path
            path = os.path.join(root, file)
            PNG_MASKS.append(path)

PNG_MASKS.sort()
# view some paths
print(PNG_MASKS[:10])

# read a sample mask
im = tf.io.read_file(PNG_MASKS[102])
# decode into a tensor
dec = tf.io.decode_png(im, channels=0, dtype = tf.dtypes.uint8)
# visualize the mask
plt.figure(figsize=(5,7))
plt.imshow(dec, cmap='jet', norm=NORM)
plt.colorbar()
plt.show()
for i in range(6,1000):
    # define the paths
    image_path = './IMAGES/img_%04d.png'%(i)
    mask_path = './MASKS/seg_%04d.png'%(i)
    # delete the image
    if os.path.exists(image_path):
        os.remove(image_path)
    
    # delete the mask
    if os.path.exists(mask_path):
        os.remove(mask_path)
        
labels = np.array(labels)
labels = pd.Series(labels, name='label_list')
labels.to_csv('labels.csv')

