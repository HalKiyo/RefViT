import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

# ---Introduction---
# This example implements the Vision Transformer(ViT) model by Al3exey Dosovitskiy et al.
# for image classification, and demonstrates it on the CIFAR-100 dataset.
# The ViT model applies the Transformer architecture with self-attention to sequences of image patches,
# without using convolution layers.

# ---Prepare the Data---
num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

# ---Configure the hyperparameters---
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = 72 # We'll resize input images to this size
patch_size = 6 # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
        projection_dim * 2,
        projection_dim,
] # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024] # Size of the dense layers of the final classifier

# ---Use data argumentation---
data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(image_size, image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
                ),
        ],
        name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)

# ---Implement multilayer perceptron(MLP)---
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# ---Implement patch creastion as a layer---
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images): # images[1,72,72,3]
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1,],
                padding="VALID",
        )
        patch_dims = patches.shape[-1] # patches[1, 12, 12, 108] -> 108
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        print(patches.shape)
        return patches

# Let's display patches for a sample image
def plot(x_train):
    plt.figure(figsize=(4, 4))
    image = x_train[np.random.choice(range(x_train.shape[0]))]
    plt.imshow(image.astype("uint8"))
    plt.axis("off")

    resized_image = tf.image.resize(
            tf.convert_to_tensor([image]), size=(image_size, image_size)
    )
    patches = Patches(patch_size)(resized_image)
    print(f"Image size: {image_size} X {image_size}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}") # 6 patch x 6 patch x 3 rgb channels

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i+1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")

# ---Implement the patch encoding layer---
# The PatchEncoder layer will linearly transform a patch 
# by projection in into a vector of size projection_dim. In addition,
# it adds a learnable postition embedding to the projected vector.
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
                input_dim=num_patches, ouput_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

if __name__ == "__main__":
    print('done')
