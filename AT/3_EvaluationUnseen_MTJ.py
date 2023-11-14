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
    return load_model(model_path)

# Preprocess the input image
def preprocess_image(img_path, target_size=(608, 800)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert single image to a batch
    img_array /= 255.  # Normalize to [0,1]
    return img_array

# Predict heatmaps using the model
def predict(img_path):
    input_image = preprocess_image(img_path)
    return model.predict(input_image)

# Visualize the original image and the predicted heatmaps
def visualize_predictions(img_path, predictions):
    # Visualize the original image
    plt.imshow(image.load_img(img_path))
    plt.title("Original Image")
    plt.show()

    # Visualize the predicted heatmaps
    plt.imshow(predictions[0, :, :, 0], cmap='hot', alpha=0.5)
    plt.title("Predicted Heatmap for 'P'")
    plt.show()

    plt.imshow(predictions[0, :, :, 1], cmap='hot', alpha=0.5)
    plt.title("Predicted Heatmap for 'D'")
    plt.show()

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


def visualize_predictions_with_coordinates(img_path, predictions):
    # Load the original image
    original_image = tf.keras.preprocessing.image.load_img(img_path)
    
    # Get coordinates from the heatmaps
    coord_p = heatmap_to_coordinates(predictions[0, :, :, 0])
    coord_d = heatmap_to_coordinates(predictions[0, :, :, 1])

    # Print the coordinates
    print(f"Coordinate for P: {coord_p}")
    print(f"Coordinate for D: {coord_d}")

    # Plot the original image with overlaid coordinates
    plt.figure(figsize=(15, 5))
    
    # 1. Original image with overlaid coordinates
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.scatter(*coord_p, c='r', s=10, marker='o', label='P')  # P in red color
    plt.scatter(*coord_d, c='b', s=10, marker='x', label='D')  # D in blue color
    plt.legend()
    plt.title("Original Image with Coordinates")

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

def predict_videos_action():
    # Prompt user to select a folder to save .csv files
    csv_save_directory = filedialog.askdirectory(title="Select a folder to save the CSV files")

    # Allow user to select multiple video files
    video_paths = filedialog.askopenfilenames(title="Select videos for prediction", filetypes=[("Video files", "*.avi;*.mp4")])

    for video_path in video_paths:
        video_name = os.path.basename(video_path).split('.')[0]  # Extracting video filename without extension
        cap = cv2.VideoCapture(video_path)
        results = []

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            frame_path = temp_file.name

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

                results.append([video_name, frame, coord_p[0], coord_p[1], coord_d[0], coord_d[1]])

            # Save the results for the current video in a .csv file
            csv_filename = os.path.join(csv_save_directory, f"{video_name}.csv")
            with open(csv_filename, 'w') as csv_file:
                csv_file.write("Video_Name, Frame, P_x, P_y, D_x, D_y\n")  # Header
                for result in results:
                    csv_file.write(f"{result[0]}, {result[1]}, {result[2]}, {result[3]}, {result[4]}, {result[5]}\n")

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