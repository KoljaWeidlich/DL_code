# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:33:46 2023

@author: Kotti
"""
import os
import numpy as np
import pickle
import datetime
import gc
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tkinter import filedialog, Tk #GUI
from tqdm import tqdm 
from multiprocessing import Pool


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

    # One output channel for MTJ, '1'  (1,1) 1x1 Kernel size means
    # operation on each pixel independently, 'sigmoid' maps the output values to the range from 0 to 1 (i.e. probabilities)
    # in short: this layer is transforming the deep feature maps from the decoder into two separate probability maps corresponding to MTJ.
    # Each of these maps will indicate the likelihood (from 0 to 1) of each pixel belonging to "P" or "D".
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    return Model(base_model.input, outputs)


# Load previously trained model
def load_previous_model():
    model_path = filedialog.askdirectory(title="Select the previously trained model directory")
    # model_path = filedialog.askopenfilename(title="Select the previously trained model file")
    # Ensure that custom objects are provided to the load_model function
    custom_objs = {'dice_coefficient': dice_coefficient}  # Add any other custom objects if there are any
    return load_model(model_path, custom_objects=custom_objs)

# Image handling
# Check if the image can be read
def can_open_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Using verify instead of load to be faster
        return True
    except:
        return False

def filter_corrupted_files_parallel(files):
    # This function will be called with multiprocessing.Pool
    with Pool() as pool:
        # Use pool.map to apply can_open_image to all files
        valid_files = list(tqdm(pool.imap(can_open_image, files), total=len(files)))
    return [file for file, valid in zip(files, valid_files) if valid]


# Filter out corrupted images and return the filenames (without extension) of valid images
def filter_corrupted_files(image_dir, heatmap_dir):
    all_image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(image_dir) for f in filenames if f.endswith('.png')]
    all_heatmap_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(heatmap_dir) for f in filenames if f.endswith('.png')]

    print(f"Checking {len(all_image_files)} original images for corruption...")
    valid_image_files = filter_corrupted_files_parallel(all_image_files)
    
    print(f"Checking {len(all_heatmap_files)} MTJ heatmaps for corruption...")
    valid_heatmap_files = filter_corrupted_files_parallel(all_heatmap_files)
    
     # Extract just the file names without the path to compare across directories
    valid_image_filenames = set([os.path.basename(f) for f in valid_image_files])
    
    # Modify the heatmap filenames to match the format of the original image filenames
    valid_heatmap_filenames = set([os.path.basename(f).replace("_heatmap.png", ".png") for f in valid_heatmap_files])
 
    # Only consider the files that are valid in all three lists
    valid_files = valid_image_filenames & valid_heatmap_filenames

  # Debug prints
    print("Valid original images:", len(valid_image_files))
    print("Valid MTJ heatmaps:", len(valid_heatmap_files))

    # Debug print
    print("Number of valid files after intersection:", len(valid_files))
    print("First few original image filenames:", list(valid_image_filenames)[:5])
    print("First few MTJ heatmap filenames:", list(valid_heatmap_filenames)[:5])


    # Clear memory after filtering valid images and before returning
    del valid_image_files
    del valid_heatmap_files
    gc.collect()
    
    return list(valid_files)


# Custom Data Generator for two separate heatmap directories
#    A custom generator function that loads and augments the images and their corresponding heatmaps for P and D from the specified directories.
#    The generator yields batches of input images and their corresponding heatmaps stacked along the channel dimension.
def custom_data_generator(filenames, image_dir, heatmap_dir, batch_size):
    # Use ImageDataGenerator for augmentation and normalization
    image_datagen = ImageDataGenerator(rescale=1./255)
    heatmap_datagen = ImageDataGenerator(rescale=1./255)
    
    # Gather all image paths using os.walk
    all_image_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(image_dir) for f in filenames if f.endswith('.png')]
    
    while True:
        # Randomly select paths for this batch
        batch_paths = np.random.choice(all_image_paths, batch_size)
        
        x_batch = []
        y_batch = []
        
        
        for img_path in batch_paths:
            try:
                # Load the original image
                x = Image.open(img_path)
                x_array = np.array(x)
                
                # Load the MTJ heatmap
                y = Image.open(img_path.replace(image_dir, heatmap_dir).replace(".png", "_heatmap.png"))
                y_array = np.array(y)
                
                                # If all images are successfully loaded, append them to the batches
                x_batch.append(x_array)
                y_batch.append(np.expand_dims(y_array, axis=-1))

            except Exception as e:
                print(f"Error reading file {img_path}. Skipping. Details: {e}")
                continue  # Skip the corrupted image or heatmap

        # Check if the batch is incomplete due to skipped images, and if so, fill in the rest
        while len(x_batch) < batch_size:
            img_filename = np.random.choice(filenames)
            try:
                img_path = os.path.join(image_dir, img_filename)
                x = Image.open(img_path)
                y = Image.open(os.path.join(heatmap_dir, img_filename.replace(".png", "_heatmap.png")))
                
                x_batch.append(np.array(x))
                y_batch.append(np.expand_dims(np.array(y), axis=-1))
            except Exception as e:
                print(f"Error reading file {img_path}. Skipping. Details: {e}")
                # Note: In an unlikely scenario where many images are corrupted, this could lead to an infinite loop.
                # You may want to implement additional checks or limits on the number of attempts.

        # Normalize the batches
        x_batch = image_datagen.flow(np.array(x_batch), batch_size=batch_size, shuffle=False).next()
        y_batch = heatmap_datagen.flow(np.array(y_batch), batch_size=batch_size, shuffle=False).next()
         
       
        yield x_batch, y_batch

# Dice coefficient (or Dice similarity coefficient, DSC). 
# The Dice coefficient is a measure of overlap between two samples
# A value of 1 indicates perfect overlap (perfect segmentation), and a value of 0 indicates no overlap.
def dice_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


if __name__ == "__main__":
    # Directory selection for training data
    root = Tk()
    root.withdraw()
    image_dir = filedialog.askdirectory(title="Select directory containing original images")
    heatmap_dir = filedialog.askdirectory(title="Select directory containing MTJ heatmaps")
    
    # Training parameters
    batch_size = 14
    epochs = 20
   # Model loading
   
    should_load_previous_model = input("Do you want to load a previously trained model? (yes/no): ")
    if should_load_previous_model.lower() == 'yes':
        model = load_previous_model()
    else:
        model = build_vgg_unet((512, 512, 3)) 
        
    # Ask the user if they want to check for corrupted images
    should_check_corruption = input("Do you want to check for corrupted images? (yes/no): ").lower()
    
    if should_check_corruption == 'yes':
        valid_image_filenames = filter_corrupted_files(image_dir, heatmap_dir)
        print(f"Found {len(valid_image_filenames)} valid images and corresponding heatmap after filtering out corrupted files.")
    else:
        valid_image_filenames = [f for dp, dn, filenames in os.walk(image_dir) for f in filenames if f.endswith('.png')]
        print(f"Using all {len(valid_image_filenames)} images without checking for corruption.")
    
    # If there are no valid images, we should stop execution.
    if not valid_image_filenames:
        print("No valid images found. Please check your directories and try again.")
        exit()

    # Split the valid images into training and validation sets
    train_images, val_images = train_test_split(valid_image_filenames, test_size=0.20, random_state=42)
    
    
   # Create generators for training and validation
    train_gen = custom_data_generator(train_images, image_dir, heatmap_dir, batch_size)
    val_gen = custom_data_generator(val_images, image_dir, heatmap_dir, batch_size)
    
   
    
    # Model summary
    model.summary()
    # Compiler
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=[dice_coefficient]) #, run_eagerly=True)

    
    # Calculate steps per epoch for training and validation
    steps_per_epoch_train = len(train_images) // batch_size
    steps_per_epoch_val = len(val_images) // batch_size
    
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
    checkpoint = ModelCheckpoint('best_weights_AT_512x512_1_epoch_20.h5', save_best_only=True, save_weights_only=False, monitor='loss', mode='min', verbose=1)
    # Set the directory to write the TensorBoard logs
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create the TensorBoard callback
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1,
                                       write_grads=True, update_freq='batch',
                                       profile_batch=2)
    #Learning rate reduction parameters
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.0000001)
    
    # Train the model
    history = model.fit(train_gen, 
                        steps_per_epoch=steps_per_epoch_train, 
                        validation_data=val_gen,
                        validation_steps=steps_per_epoch_val, 
                        epochs=epochs, 
                        callbacks=[checkpoint, tensorboard_callback, reduce_lr])
    
    # Save the model
    save_directory = filedialog.askdirectory(title="Select directory to save the model")
    model.save(save_directory)
    
    # Save the history object
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)