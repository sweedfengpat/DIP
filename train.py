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
image_path = []
for root, dirs, files in os.walk('C:\\Users\\patza\\Desktop\\dip\\IMAGES'):
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
for root, dirs, files in os.walk('C:\\Users\\patza\\Desktop\\dip\\MASKS'):
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
    plt.imshow(img, cmap='jet', norm=NORM)
    plt.colorbar()
    plt.axis('off')
plt.show()

base = keras.applications.DenseNet121(input_shape=[128,128,3],include_top=False,weights='imagenet')
len(base.layers)
#keras.utils.plot_model(base, show_shapes=True)
skip_names = ['conv1/relu','pool2_relu','pool3_relu','pool4_relu','relu']
skip_outputs = [base.get_layer(name).output for name in skip_names]
for i in range(len(skip_outputs)):
    print(skip_outputs[i])

downstack = keras.Model(inputs=base.input,
                       outputs=skip_outputs)
# freeze the downstack layers
downstack.trainable = False

upstack = [pix2pix.upsample(512,3),pix2pix.upsample(256,3),pix2pix.upsample(128,3),pix2pix.upsample(64,3)]
inputs = keras.layers.Input(shape=[128,128,3])
upstack[0].layers
down = downstack(inputs)
out = down[-1]

# prepare skip-connections
skips = reversed(down[:-1])
# choose the last layer at first 4 --> 8

# upsample with skip-connections
for up, skip in zip(upstack,skips):
    out = up(out)
    out = keras.layers.Concatenate()([out,skip])
    
# define the final transpose conv layer
# image 128 by 128 with 59 classes
out = keras.layers.Conv2DTranspose(59, 3,strides=2,padding='same',)(out)
# complete unet model
unet = keras.Model(inputs=inputs, outputs=out)
#keras.utils.plot_model(unet, show_shapes=True)
images[0].shape, masks[0].shape
def resize_image(image):
    # scale the image
    image = tf.cast(image, tf.float32)
    image = image/255.0
    # resize image
    image = tf.image.resize(image, (128,128))
    return image

def resize_mask(mask):
    # resize the mask
    mask = tf.image.resize(mask, (128,128))
    mask = tf.cast(mask, tf.uint8)
    return mask    

X = [resize_image(i) for i in images]
y = [resize_mask(m) for m in masks]
len(X), len(y)
images[0].dtype, masks[0].dtype, X[0].dtype, y[0].dtype
plt.imshow(X[0])
plt.colorbar()
plt.show()

#plot a mask
plt.imshow(y[0], cmap='jet')
plt.colorbar()
plt.show()

# split data into 80/20 ratio
train_X, val_X,train_y, val_y = train_test_split(X, y, test_size=0.2,random_state=0)
# develop tf Dataset objects
train_X = tf.data.Dataset.from_tensor_slices(train_X)
val_X = tf.data.Dataset.from_tensor_slices(val_X)

train_y = tf.data.Dataset.from_tensor_slices(train_y)
val_y = tf.data.Dataset.from_tensor_slices(val_y)

# verify the shapes and data types
train_X.element_spec, train_y.element_spec, val_X.element_spec, val_y.element_spec
def brightness(img, mask):
    # adjust brightness of image
    # don't alter in mask
    img = tf.image.adjust_brightness(img, 0.1)
    return img, mask

def gamma(img, mask):
    # adjust gamma of image
    # don't alter in mask
    img = tf.image.adjust_gamma(img, 0.1)
    return img, mask

def hue(img, mask):
    # adjust hue of image
    # don't alter in mask
    img = tf.image.adjust_hue(img, -0.1)
    return img, mask

def crop(img, mask):
    # crop both image and mask identically
    img = tf.image.central_crop(img, 0.7)
    # resize after cropping
    img = tf.image.resize(img, (128,128))
    mask = tf.image.central_crop(mask, 0.7)
    # resize afer cropping
    mask = tf.image.resize(mask, (128,128))
    # cast to integers as they are class numbers
    mask = tf.cast(mask, tf.uint8)
    return img, mask

def flip_hori(img, mask):
    # flip both image and mask identically
    img = tf.image.flip_left_right(img)
    mask = tf.image.flip_left_right(mask)
    return img, mask

def flip_vert(img, mask):
    # flip both image and mask identically
    img = tf.image.flip_up_down(img)
    mask = tf.image.flip_up_down(mask)
    return img, mask

def rotate(img, mask):
    # rotate both image and mask identically
    img = tf.image.rot90(img)
    mask = tf.image.rot90(mask)
    return img, mask

# zip images and masks
train = tf.data.Dataset.zip((train_X, train_y))
val = tf.data.Dataset.zip((val_X, val_y))

# perform augmentation on train data only

a = train.map(brightness)
b = train.map(gamma)
c = train.map(hue)
d = train.map(crop)
e = train.map(flip_hori)
f = train.map(flip_vert)
g = train.map(rotate)

# concatenate every new augmented sets
train = train.concatenate(a)
train = train.concatenate(b)
train = train.concatenate(c)
train = train.concatenate(d)
train = train.concatenate(e)
train = train.concatenate(f)
train = train.concatenate(g)

BATCH = 64
AT = tf.data.AUTOTUNE
BUFFER = 1000

STEPS_PER_EPOCH = 800//BATCH
VALIDATION_STEPS = 200//BATCH

train = train.cache().shuffle(BUFFER).batch(BATCH).repeat()
train = train.prefetch(buffer_size=AT)
val = val.batch(BATCH)

example = next(iter(train))
preds = unet(example[0])
# visualize an image
plt.imshow(example[0][60])
plt.colorbar()
plt.show()

# visualize the predicted mask
pred_mask = tf.argmax(preds, axis=-1)
pred_mask = tf.expand_dims(pred_mask, -1)
plt.imshow(pred_mask[0], cmap='jet', norm=NORM)
plt.colorbar()

unet.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=keras.optimizers.RMSprop(lr=0.001),metrics=['accuracy']) 

hist = unet.fit(train,validation_data=val,steps_per_epoch=STEPS_PER_EPOCH,validation_steps=VALIDATION_STEPS,epochs=200)

# select a validation data batch
img, mask = next(iter(val))
# make prediction
pred = unet.predict(img)
plt.figure(figsize=(10,5))
# plot the predicted mask
for i in pred:
    plt.subplot(121)
    i = tf.argmax(i, axis=-1)
    plt.imshow(i,cmap='jet', norm=NORM)
    plt.axis('off')
    plt.title('Prediction')
    break
# plot the groundtruth mask
plt.subplot(122)
plt.imshow(mask[0], cmap='jet', norm=NORM)
plt.axis('off')
plt.title('Ground Truth')
plt.show()
history = hist.history
acc=history['accuracy']
val_acc = history['val_accuracy']

plt.plot(acc, '-', label='Training Accuracy')
plt.plot(val_acc, '--', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
