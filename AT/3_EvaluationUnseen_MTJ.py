import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
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
    img = image.load_img(img_path)
    img_array = image.img_to_array(img)
    
    # Crop the image
    cropped_img_array = img_array[:-180, 190:-190]
    
    # Resize
    resized_img_array = cv2.resize(cropped_img_array, (608, 608))
    
    resized_img_array = np.expand_dims(resized_img_array, axis=0)  # Convert single image to a batch
    resized_img_array /= 255.  # Normalize to [0,1]
    return resized_img_array

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
    plt.title("Predicted Heatmap for 'MTJ'")
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
    
    # Load the preprocessed image
    preprocessed_image_array = preprocess_image(img_path)
    
    # Convert array to a viewable image
    preprocessed_image = tf.keras.preprocessing.image.array_to_img(preprocessed_image_array[0])
    
    # Get coordinates from the heatmap
    coord_MTJ = heatmap_to_coordinates(predictions[0, :, :, 0])


    # Print the coordinates
    print(f"Coordinate for MTJ: {coord_MTJ}")


    # Plot the preprocessed image with overlaid coordinates
    plt.figure(figsize=(15, 5))
    
    # 1. Preprocessed image with overlaid coordinates
    plt.subplot(1, 2, 1)
    plt.imshow(preprocessed_image)
    plt.scatter(*coord_MTJ, c='r', s=10, marker='o', label='MTJ')  # P in red color
    plt.legend()
    plt.title("Preprocessed Image with Coordinates")

    # 2. Predicted heatmap for MTJ
    plt.subplot(1, 2, 2)
    plt.imshow(predictions[0, :, :, 0], cmap='hot', interpolation='nearest')
    plt.title("Predicted Heatmap for MTJ")


    plt.tight_layout()
    plt.show()


def predict_videos_action():
    # Prompt user to select a folder to save .csv files
    csv_save_directory = filedialog.askdirectory(title="Select a folder to save the CSV files")
    
    # Prompt user to select a folder to save plots
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
                coord_MTJ = heatmap_to_coordinates(predictions[0, :, :, 0])

                results.append([video_name, frame_counter, coord_MTJ[0], coord_MTJ[1]])
                
                # Visualization
                visualize_predictions_with_coordinates(frame_path, predictions)
                # Save the plot
                plt.savefig(os.path.join(plot_save_directory, f"{video_name}_frame_{frame_counter}.png"))
                frame_counter += 1  # Increment the frame counter
            # Save the results for the current video in a .csv file
            csv_filename = os.path.join(csv_save_directory, f"{video_name}.csv")
            with open(csv_filename, 'w') as csv_file:
                csv_file.write("Video_Name, Frame, MTJ_x, MTJ_y\n")  # Header
                for result in results:
                    csv_file.write(f"{result[0]}, {result[1]}, {result[2]}, {result[3]}\n")



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