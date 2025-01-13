# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:03:52 2023

@author: Kolja 'Kotti' Weidlich

24/10/23 Kolja: implemented preprocessing of images/videos to 608x608

TO DO save preprocessed images from video file!

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tkinter import filedialog, Tk, Button, Label
import tempfile
import os

# Load the trained model
def load_trained_model(model_path):
    custom_objects = {'dice_coefficient': dice_coefficient}
    return load_model(model_path, custom_objects=custom_objects)


def dice_coefficient(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


# Preprocess the input image
def preprocess_image(img_path, target_size=(608, 608)):
    # Load the image and convert it to an array
    img = image.load_img(img_path)
    img_array = image.img_to_array(img)
    
    # Crop the image
    cropped_img_array = img_array[128:-128, 192:]
    
    # Add padding to the bottom to make the image 608x608
    padding_shape = (608 - cropped_img_array.shape[0], cropped_img_array.shape[1])
    padded_img_array = cv2.copyMakeBorder(cropped_img_array, 0, padding_shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    # Normalize and expand dimensions to make it a batch
    padded_img_array = np.expand_dims(padded_img_array, axis=0) / 255.
    return padded_img_array

# Predict heatmaps using the model
def predict(img_path):
    input_image = preprocess_image(img_path)
    return model.predict(input_image)

# def post_process_predictions(predictions):
#     # Remove padding from the bottom to revert the image to 608x800
#     return predictions[:, 128:-128, 192:, :]


# GUI actions
def load_model_action():
    global model
    model_path = filedialog.askdirectory(title="Select the trained model")
    model = load_trained_model(model_path)
    lbl.configure(text="Model Loaded!")

def load_and_predict_action():
    test_image_path = filedialog.askopenfilename(title="Select an image for prediction")
    predictions = predict(test_image_path)
    visualize_predictions_with_coordinates(test_image_path, predictions)
    
    
# Extract coordinates from the heatmap
def heatmap_to_coordinates(heatmap):
    # Find the coordinate with the maximum value in the heatmap
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return x, y

def map_coordinates_to_original(coord, preprocessed_shape, original_shape, crop_x=192, crop_y_top=128, crop_y_bottom=128):
    """
    Map coordinates from the preprocessed image back to the original image dimensions.
    
    Args:
        coord (tuple): Coordinates (x, y) in the preprocessed image.
        preprocessed_shape (tuple): Shape of the preprocessed image (height, width).
        original_shape (tuple): Shape of the original image (height, width).
        crop_x (int): Amount of cropping applied from the left on the x-axis.
        crop_y_top (int): Amount of cropping applied from the top on the y-axis.
        crop_y_bottom (int): Amount of cropping applied from the bottom on the y-axis.

    Returns:
        tuple: Mapped coordinates (x, y) in the original image.
    """
    # Extract the width and height of the cropped region
    cropped_width = original_shape[1] - crop_x
    cropped_height = original_shape[0] - crop_y_top - crop_y_bottom

    # Calculate scale factors
    scale_x = cropped_width / preprocessed_shape[1]
    scale_y = cropped_height / preprocessed_shape[0]

    # Map coordinates from preprocessed to original dimensions
    x_original = coord[0] * scale_x + crop_x
    y_original = original_shape[0] - (coord[1] * scale_y + crop_y_top)

    return int(round(x_original)), int(round(y_original))




def visualize_predictions_with_coordinates(img_path, predictions):
    # Load the original image for dimension reference
    original_image = image.load_img(img_path)
    original_shape = original_image.size[::-1]  # PIL size is (width, height), reverse it for (height, width)

    # Load the preprocessed image
    preprocessed_image_array = preprocess_image(img_path)
    preprocessed_shape = preprocessed_image_array.shape[1:3]  # Get height and width

    # Convert array to a viewable image
    preprocessed_image = tf.keras.preprocessing.image.array_to_img(preprocessed_image_array[0])
    
    # Get coordinates from the heatmaps
    coord_p = heatmap_to_coordinates(predictions[0, :, :, 0])
    coord_d = heatmap_to_coordinates(predictions[0, :, :, 1])

    # Map coordinates to original image dimensions
    coord_p_original = map_coordinates_to_original(coord_p, preprocessed_shape, original_shape)
    coord_d_original = map_coordinates_to_original(coord_d, preprocessed_shape, original_shape)

    # Print the coordinates
    print(f"Coordinate for P (preprocessed): {coord_p}")
    print(f"Coordinate for D (preprocessed): {coord_d}")
    print(f"Coordinate for P (original): {coord_p_original}")
    print(f"Coordinate for D (original): {coord_d_original}")

    # Plot the preprocessed image with overlaid coordinates
    plt.figure(figsize=(15, 5))
    
    # 1. Preprocessed image with overlaid coordinates
    plt.subplot(1, 3, 1)
    plt.imshow(preprocessed_image)
    plt.scatter(*coord_p, c='r', s=10, marker='o', label='P')  # P in red color
    plt.scatter(*coord_d, c='b', s=10, marker='x', label='D')  # D in blue color
    plt.legend()
    plt.title("Preprocessed Image with Coordinates")

    # 2. Predicted heatmap for P
    plt.subplot(1, 3, 2)
    plt.imshow(predictions[0, :, :, 0], cmap='hot', interpolation='nearest')
    plt.title("Predicted Heatmap for P")

    # 3. Predicted heatmap for D
    plt.subplot(1, 3, 3)
    plt.imshow(predictions[0, :, :, 1], cmap='hot', interpolation='nearest')
    plt.title("Predicted Heatmap for D")

    plt.tight_layout()
    plt.show()

    return coord_p_original, coord_d_original



def predict_videos_action():
    # Prompt user to select a folder to save .csv files
    csv_save_directory = filedialog.askdirectory(title="Select a folder to save the CSV files")
    
    # Prompt user to select a folder to save plots (optional)
    save_plots = input("Do you want to save plots? (yes/no): ").strip().lower() == "yes"
    plot_save_directory = None
    if save_plots:
        plot_save_directory = filedialog.askdirectory(title="Select a folder to save the plots")

    # Allow user to select multiple video files
    video_paths = filedialog.askopenfilenames(title="Select videos for prediction", filetypes=[("Video files", "*.avi;*.mp4")])

    for video_path in video_paths:
        video_name = os.path.basename(video_path).split('.')[0]  # Extracting video filename without extension
        cap = cv2.VideoCapture(video_path)
        results = []

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            frame_path = temp_file.name
            frame_counter = 0  # Initialize a frame counter
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame to the required format and predict
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = image.array_to_img(frame_rgb)
                frame_pil.save(frame_path)
                predictions = predict(frame_path)

                # Extracting coordinates from the heatmaps
                coord_p = heatmap_to_coordinates(predictions[0, :, :, 0])
                coord_d = heatmap_to_coordinates(predictions[0, :, :, 1])

                # Map coordinates to original image dimensions
                original_shape = frame.shape[:2]  # (height, width)
                preprocessed_shape = (608, 608)
                coord_p_original = map_coordinates_to_original(coord_p, preprocessed_shape, original_shape)
                coord_d_original = map_coordinates_to_original(coord_d, preprocessed_shape, original_shape)

                results.append([
                    video_name,
                    frame_counter,
                    coord_p_original[0], coord_p_original[1],
                    coord_d_original[0], coord_d_original[1]
                ])
                
                # Visualization (if enabled)
                if save_plots:
                    visualize_predictions_with_coordinates(frame_path, predictions)
                    plt.savefig(os.path.join(plot_save_directory, f"{video_name}_frame_{frame_counter}.png"))

                frame_counter += 1  # Increment the frame counter

            # Save the results for the current video in a .csv file
            csv_filename = os.path.join(csv_save_directory, f"{video_name}.csv")
            with open(csv_filename, 'w') as csv_file:
                csv_file.write("Video_Name, Frame, P_x_original, P_y_original, D_x_original, D_y_original\n")  # Header
                for result in results:
                    csv_file.write(f"{result[0]}, {result[1]}, {result[2]}, {result[3]}, {result[4]}, {result[5]}\n")


        # Clean up temporary file
        if os.path.exists(frame_path):
            os.unlink(frame_path)

            



# Main execution
if __name__ == "__main__":
    root = Tk()
    root.title("Model Tester")

    lbl = Label(root, text="Load your model and then select an image or video for prediction")
    lbl.pack(pady=20)

    load_model_btn = Button(root, text="Load Model", command=load_model_action)
    load_model_btn.pack(pady=20)

    predict_image_btn = Button(root, text="Select Image and Predict", command=load_and_predict_action)
    predict_image_btn.pack(pady=20)

    predict_video_btn = Button(root, text="Select Videos and Predict", command=predict_videos_action)
    predict_video_btn.pack(pady=20)

    root.mainloop()