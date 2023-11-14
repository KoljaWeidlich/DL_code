# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:03:52 2023

@author: Kolja 'Kotti' Weidlich
23/10/23 Kolja: now training on cropped, padded and resized images (original 608x800 --> 512x512)
24/10/23 Kolja: made the finding of step_per_epoch parameter more robust to user error -.-'

"""

#Overview: 
    # The code is designed to train a U-Net model, utilizing the VGG16 architecture as its encoder,
    # to predict heatmaps for two structures (referred to as P and D) from input images. 
    # The model predicts two channels of output for these two structures.

import os
import numpy as np
import pickle
import datetime
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from tkinter import filedialog, Tk
# import tensorflow as tf
# tf.config.run_functions_eagerly(True)

# VGG16 U-Net Model Architecture
    # This function is designed to build a U-Net model using the VGG16 architecture as its encoder.
    # The encoder is obtained from the pre-trained VGG16 model, while the decoder is custom-built using upsampling and convolution layers.
    # The model has two output channels, corresponding to the P and D structures.
def build_vgg_unet(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Encoder (VGG16 layers) captures the main features of the image
    c1 = base_model.get_layer('block1_conv2').output
    c2 = base_model.get_layer('block2_conv2').output
    c3 = base_model.get_layer('block3_conv3').output
    c4 = base_model.get_layer('block4_conv3').output
    c5 = base_model.get_layer('block5_conv3').output

    # Decoder uses the encoder information to reconstruct the image (or a segmentation of the image) 
    u1 = UpSampling2D((2, 2))(c5)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u1)
    u2 = UpSampling2D((2, 2))(c6)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u2)
    u3 = UpSampling2D((2, 2))(c7)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u3)
    u4 = UpSampling2D((2, 2))(c8)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u4)

    # Two output channels: One for P and one for D, '2' mean stwo channels correspond to two types of outputs: one for "P" and another for "D", (1,1) 1x1 Kernel size means
    # operation on each pixel independently, 'sigmoid' maps the output values to the range from 0 to 1 (i.e. probabilities)
    # in short: this layer is transforming the deep feature maps from the decoder into two separate probability maps corresponding to "P" and "D".
    # Each of these maps will indicate the likelihood (from 0 to 1) of each pixel belonging to "P" or "D".
    outputs = Conv2D(2, (1, 1), activation='sigmoid')(c9)

    return Model(base_model.input, outputs)

# Load previously trained model
def load_previous_model():
    model_path = filedialog.askdirectory(title="Select the previously trained model directory")
    return load_model(model_path)


# Custom Data Generator for two separate heatmap directories
#    A custom generator function that loads and augments the images and their corresponding heatmaps for P and D from the specified directories.
#    The generator yields batches of input images and their corresponding heatmaps stacked along the channel dimension.
def custom_data_generator(image_dir, heatmap_p_dir, heatmap_d_dir, batch_size):
    # Use ImageDataGenerator for augmentation and normalization
    image_datagen = ImageDataGenerator(rescale=1./255)
    heatmap_datagen = ImageDataGenerator(rescale=1./255)

    image_generator = image_datagen.flow_from_directory(
        image_dir,
        class_mode=None,
        batch_size=batch_size,
        seed=42,
        color_mode="rgb",
        target_size=(608, 608)
    )

    heatmap_p_generator = heatmap_datagen.flow_from_directory(
        heatmap_p_dir,
        class_mode=None,
        batch_size=batch_size,
        seed=42,
        color_mode="grayscale",
        target_size=(608, 608)
    )

    heatmap_d_generator = heatmap_datagen.flow_from_directory(
        heatmap_d_dir,
        class_mode=None,
        batch_size=batch_size,
        seed=42,
        color_mode="grayscale",
        target_size=(608, 608)
    )

    while True:
        x_batch = image_generator.next()
        y_p_batch = heatmap_p_generator.next()
        y_d_batch = heatmap_d_generator.next()
        y_batch = np.concatenate([y_p_batch, y_d_batch], axis=-1)
        yield x_batch, y_batch


if __name__ == "__main__":
    # Directory selection for training data
    root = Tk()
    root.withdraw()
    image_dir = filedialog.askdirectory(title="Select directory containing original images")
    heatmap_p_dir = filedialog.askdirectory(title="Select directory containing P heatmaps")
    heatmap_d_dir = filedialog.askdirectory(title="Select directory containing D heatmaps")
    
    # Training parameters
    batch_size = 14
    epochs = 10
    
    
    # Model
    should_load_previous_model = input("Do you want to load a previously trained model? (yes/no): ")
    if should_load_previous_model.lower() == 'yes':
        model = load_previous_model()
    else:
        model = build_vgg_unet((608, 608, 3))
    model.summary()
    # Compiler
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy') #, run_eagerly=True)

    # Data generators
    train_gen = custom_data_generator(image_dir, heatmap_p_dir, heatmap_d_dir, batch_size)
    
      
    # Calculate steps per epoch, count the number of files of the original images to set the steps per epoch
    num_samples = sum([len(files) for r, d, files in os.walk(image_dir) if any(file.endswith('.png') for file in files)])
    steps_per_epoch = num_samples // batch_size
    
    # FOR DEBUGGING
    # Get one batch of data from the generator
    x_batch, y_batch = next(train_gen)
    
    # Inspect the shapes
    print("Input batch shape:", x_batch.shape)
    print("Output batch shape:", y_batch.shape)
    
    # Inspect the data types
    print("Input batch data type:", x_batch.dtype)
    print("Output batch data type:", y_batch.dtype)
    
    # Inspect the data range
    print("Input batch min, max values:", x_batch.min(), x_batch.max())
    print("Output batch min, max values:", y_batch.min(), y_batch.max())
    
    
    # Model Checkpoint
    checkpoint = ModelCheckpoint('best_weights_PT.h5', save_best_only=True, save_weights_only=True, monitor='loss', mode='min', verbose=1)
    # Set the directory to write the TensorBoard logs
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create the TensorBoard callback
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
    # Train the model
    # model.fit(train_gen, steps_per_epoch=len(os.listdir(image_dir)) // batch_size, epochs=epochs, callbacks=[checkpoint])
    history = model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[checkpoint, tensorboard_callback])
    
    # Save the model
    save_directory = filedialog.askdirectory(title="Select directory to save the model")
    model.save(save_directory)
    
    # Save the history object
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)


