# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:03:52 2023

@author: Kolja 'Kotti' Weidlich
18/10/23 Arno 'Alu' Schroll joins the Deep Learning addiction
23/10/23 Kolja: Addition of different heatmaps for P and D. P remains a circle, D becomes more elliptical
23/10/23 Kolja: remove irrelevant information by cropping 192 pixels in x direction(from left to right), and -128 pixels y-direction (from top and bottom) 
24/10/23 Kolja: added black padding from the bottom --> final size 608x608
31/10/23 Kolja: added scaling sigma (heatmap spreading) based on image width. Added handling of multiple trk files for one video. Added summary output
                of processed videos. Changed trk and video file mapping strategy to make it more robust against filename discrepancies. Added error handling
"""



import os # interact with the operating system, such as reading file paths or creating directories
import re # Regular expressions module, useful for string matching and manipulation.
import cv2 #toolkit for computer vision tasks
import numpy as np #numerical operations, especially with matrices
from tkinter import filedialog, Tk #GUI library, used here for opening file dialogs
from PIL import Image
import csv


# This function reads the .trk file which contains tendon position information. 
# It searches for specific keywords that define the position of the tendons and
# extracts their (x, y) coordinates for each frame.

def extract_tendon_position(trk_file, keywords):
    with open(trk_file, 'r') as f:
        CStr = f.readlines()

    results = []


    for keyword in keywords:
        expression = f'<property name="name" type="string">{keyword}</property>'
        start_indices = [i for i, s in enumerate(CStr) if expression in s]
        end_indices = [i for i, s in enumerate(CStr) if '<property name="array" type="string">{' in s]
        frame_indices = [i for i, s in enumerate(CStr) if '<property name="[' in s]

        if not start_indices:
            continue

        start = start_indices[0]
        stop = next(i for i in end_indices if i > start)

        
        matrix_info = []

        idx = 0
        for i in range(start, stop + 1):
            if i in frame_indices:
                frame = int(re.search(r'\"\[(\d+)\]\"', CStr[i]).group(1))
                
                x_match = re.search(r'\">([\d\.\-]+)<', CStr[i + 2])
                y_match = re.search(r'\">([\d\.\-]+)<', CStr[i + 3])
                
                if x_match is None or y_match is None:
                    print(f"Warning: Expected coordinate pattern not found in {trk_file} around line {i + 2}")
                    return [], False  # Return an empty list and a failed flag
                
                x = float(x_match.group(1))
                y = float(y_match.group(1))
                
                matrix_info.append([frame, x, y])
                idx += 1

        results.append(matrix_info)

    return results, True  # Return results and a success flag

# Given the content of a .trk file, it extracts the name of the corresponding video file.
def extract_video_name_from_trk(trk_content):
    """Extract the video filename from the given .trk content."""
    for line in trk_content:
        if '<property name="path" type="string">' in line:
            return line.split(">")[1].split("<")[0]
    return None

def map_trk_to_videos(trk_files, video_files):
    """Map each .trk file to its associated video file."""
    mapping = {}

    for trk_file in trk_files:
        with open(trk_file, 'r') as f:
            trk_content = f.readlines()
        video_name = extract_video_name_from_trk(trk_content)
        if video_name:
            # Find the video that matches the video_name
            corresponding_video = next((vf for vf in video_files if video_name in vf), None)
            if corresponding_video:
                if corresponding_video not in mapping:
                    mapping[corresponding_video] = []
                mapping[corresponding_video].append(trk_file)

    return mapping


# Generates a heatmap around a given coordinate (x,y) on an image. 
# This heatmap is a 2D Gaussian distribution centered at the coordinate. 
# The parameter sigma determines the spread of the Gaussian function. 
# The function returns a normalized heatmap.
# In short: essentially paints a "blob" (or hotspot) at the specified (x, y) coordinate,
# with the intensity of the blob decreasing as you move away from this center, mimicking the behavior of a 2D Gaussian distribution.

def generate_heatmaps(image_shape, P_coord, D_coord, base_sigma=40):
    """
    Generate heatmaps for given coordinates with Gaussian distributions.
    Returns a tuple of two heatmaps: one for P and an elliptical one for D.
    """
    x = np.arange(0, image_shape[1], 1, float)  # Width
    y = np.arange(0, image_shape[0], 1, float)  # Height
    y = y[:, np.newaxis]
    
    # 2D Gaussian centered at the coordinate
    # computes the value of a 2D Gaussian function centered at the provided coordinate (coord). 
    # x​0 and y0​ are coordinates around which the heatmap is centered (i.e., coord). Sigma is the spread of the Gaussian function.
    
    # Compute sigma based on image width
    sigma = (image_shape[1]/800) * base_sigma
    
    # Heatmap for P
    heatmap_P = np.exp(-((x - P_coord[0]) ** 2 + (y - P_coord[1]) ** 2) / (2 * sigma ** 2))
    
    # Elliptical heatmap for D
    heatmap_D = np.exp(-((x - D_coord[0]) ** 2 / (2 * (2*sigma) ** 2) + (y - D_coord[1]) ** 2 / (2 * sigma ** 2)))
    
    # Normalize the heatmaps
    # heatmap is normalized by dividing every pixel value by the maximum value in the heatmap. 
    # This ensures that pixel values are in the range [0,1], with 1 being the intensity at the center of the heatmap.
    heatmap_P /= np.max(heatmap_P)
    heatmap_D /= np.max(heatmap_D)
    
    return heatmap_P, heatmap_D

# Unifies pathsyntax to forward slashes (sometimes useful on Windows natives)
def forward_slash_path(path):
    return path.replace("\\", "/")

# Initialization: A Tkinter window GUI is created and then hidden.
# File Selection: File dialogs are opened to select video files and their corresponding .trk files.
# Sorting and Mapping: The selected video and .trk files are sorted. A dictionary (trk_to_video_mapping) is created to map each video file to its corresponding .trk files.
# Directory Init.: user is asked for a directory where images will be saved. If none is provided, the directory of the video files is used. 
# Directories for original images and heatmaps are then created.

# Processing Each Video: For each video in the mapping:
#     The filename is decomposed to extract study information.
#     Each associated .trk file is processed to extract tendon positions.
#     The video is read frame by frame.
#     For each frame, tendon positions are extracted and heatmaps are generated.
#     The original frame and the generated heatmaps are saved as images.

def VideoToHeatmaps():
    root = Tk()
    root.withdraw()  # Hide the main window

    # Step 1: File Selection
    video_files = filedialog.askopenfilenames(title="Select video files", filetypes=[("Video files", "*.avi")])
    trk_files = filedialog.askopenfilenames(title="Select .trk files", filetypes=[("TRK files", "*.trk")])

    trk_to_video_mapping = map_trk_to_videos(trk_files, video_files)

    # Step 2: Initialization
    keywords = ['P', 'D']
    save_directory = filedialog.askdirectory(title="Select a directory to save images")
    if not save_directory:  # If user cancels the directory selection, use the directory of the video files
        save_directory = os.path.dirname(video_files[0])

    original_images_dir = forward_slash_path(os.path.join(save_directory, "original_images", "all data"))
    heatmap_P_dir = forward_slash_path(os.path.join(save_directory, "heatmaps_P", "all data"))
    heatmap_D_dir = forward_slash_path(os.path.join(save_directory, "heatmaps_D", "all data"))

    # Create the directories if they don't exist
    os.makedirs(original_images_dir, exist_ok=True)
    os.makedirs(heatmap_P_dir, exist_ok=True)
    os.makedirs(heatmap_D_dir, exist_ok=True)

    # Initialize summary data list with headers
    # summary_data = [["Video_name", "trk file found", "number of original images"]]
    video_process_count = {video: 0 for video in trk_to_video_mapping.keys()}

    # Step 3: Processing each video
    for video_file, trk_file_list in trk_to_video_mapping.items():
        video_name = os.path.basename(video_file).split('.')[0]  # Extracting the video filename without extension

        version_counter = 1  # Initialize the version counter (in case several .trk files exist for one video)
        for trk_file in trk_file_list:  # Process each trk file associated with the video
            
            positions, success = extract_tendon_position(trk_file, keywords)  # Extract positions and check the success flag
            
            if not success:
                print(f"Skipping processing for video {video_file} due to issues with associated trk file {trk_file}")
                break  # Skip the rest of the processing for this video and move on to the next one
        
            cap = cv2.VideoCapture(video_file)
            ret, frame = cap.read()
            frame_count = 1
                    
            while ret:
                if np.array(positions[0]).ndim == 2 and np.array(positions[1]).ndim == 2:
                    P_positions_array = np.array(positions[0])
                    D_positions_array = np.array(positions[1])
            
                    filtered_positions_P = P_positions_array[P_positions_array[:, 0] == frame_count][:, 1:3]
                    filtered_positions_D = D_positions_array[D_positions_array[:, 0] == frame_count][:, 1:3]
            
                    if filtered_positions_P.size > 0 and filtered_positions_D.size > 0:
                        P_position = filtered_positions_P[0]
                        D_position = filtered_positions_D[0]
                   
                        heatmap_P, heatmap_D = generate_heatmaps(frame.shape, P_position, D_position)  # Generate both heatmaps
                    else:
                        P_position = None
                        D_position = None
                        
                    if P_position is not None and D_position is not None:
                        original_filename = forward_slash_path(os.path.join(
                            original_images_dir,
                            f"{video_name}_v{version_counter}_frame_{frame_count}.png"
                        ))
                        heatmap_P_filename = forward_slash_path(os.path.join(
                            heatmap_P_dir,
                            f"{video_name}_v{version_counter}_heatmap_P_frame_{frame_count}.png"
                        ))
                        heatmap_D_filename = forward_slash_path(os.path.join(
                            heatmap_D_dir,
                            f"{video_name}_v{version_counter}_heatmap_D_frame_{frame_count}.png"
                        ))

                        
                        # Crop the original and heatmap images
                        cropped_frame = frame[128:-128, 192:]
                        cropped_heatmap_P = heatmap_P[128:-128, 192:] if P_position is not None else None
                        cropped_heatmap_D = heatmap_D[128:-128, 192:] if D_position is not None else None
                
                        # Add padding to the bottom to make the images 608x608
                        padding_shape = (608 - cropped_frame.shape[0], cropped_frame.shape[1])
                        padded_frame = cv2.copyMakeBorder(cropped_frame, 0, padding_shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
                        if cropped_heatmap_P is not None:
                            padded_heatmap_P = cv2.copyMakeBorder(cropped_heatmap_P, 0, padding_shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
                        if cropped_heatmap_D is not None:
                            padded_heatmap_D = cv2.copyMakeBorder(cropped_heatmap_D, 0, padding_shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
                            
                        # # Resize to 512x512
                        # resized_frame = cv2.resize(padded_frame, (512, 512))
                        # if cropped_heatmap_P is not None:
                        #     resized_heatmap_P = cv2.resize(padded_heatmap_P, (512, 512))
                        # if cropped_heatmap_D is not None:
                        #     resized_heatmap_D = cv2.resize(padded_heatmap_D, (512, 512))

                        # Save the padded images
                        # success_original = cv2.imwrite(original_filename, padded_frame)
                        # if not success_original:
                        #     print(f"Failed to save original image to: {original_filename}")
                        
                        # if P_position is not None:
                        #     print(f"Saving heatmap P to: {heatmap_P_filename}")  # Debug print
                        #     print(f"Heatmap P shape: {padded_heatmap_P.shape}, dtype: {padded_heatmap_P.dtype}, min: {padded_heatmap_P.min()}, max: {padded_heatmap_P.max()}")
                        #     success_P = cv2.imwrite(heatmap_P_filename, (padded_heatmap_P * 255).astype(np.uint8))
                        #     if not success_P:
                        #         print(f"Failed to save heatmap P to: {heatmap_P_filename}")
                        
                        # if D_position is not None:
                        #     print(f"Saving heatmap D to: {heatmap_D_filename}")  # Debug print
                        #     print(f"Heatmap D shape: {padded_heatmap_D.shape}, dtype: {padded_heatmap_D.dtype}, min: {padded_heatmap_D.min()}, max: {padded_heatmap_D.max()}")
                        #     success_D = cv2.imwrite(heatmap_D_filename, (padded_heatmap_D * 255).astype(np.uint8))
                        #     if not success_D:
                        #         print(f"Failed to save heatmap D to: {heatmap_D_filename}")
                        # Convert the OpenCV image format (BGR) to Pillow format (RGB)
                       
                        pil_img = Image.fromarray(cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB))
                        pil_img.save(original_filename)
                        
                        if P_position is not None:
                            pil_img_P = Image.fromarray((padded_heatmap_P * 255).astype(np.uint8))
                            pil_img_P.save(heatmap_P_filename)
                        
                        if D_position is not None:
                            pil_img_D = Image.fromarray((padded_heatmap_D * 255).astype(np.uint8))
                            pil_img_D.save(heatmap_D_filename)
                            
                        version_counter +=1 # Increment the version counter for the next .trk file
                        video_process_count[video_file] += 1
    

            
                ret, frame = cap.read()
                frame_count += 1
                # num_original_images += 1  # Increment the count for each processed frame

        # summary_data.append([os.path.basename(video_file), trk_found, num_original_images])
        cap.release()
        
    # After processing all videos, save the summary data to a CSV file
    summary_csv_file = forward_slash_path(os.path.join(save_directory, "summary.csv"))
    # Write the summary to the CSV
    with open(summary_csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Video_name", "trk file found (yes/no)", "Number of original images", "Times processed"])
        for video, count in video_process_count.items():
            trk_found = "yes" if video in trk_to_video_mapping and trk_to_video_mapping[video] else "no"
            num_images = count / len(trk_to_video_mapping[video]) if trk_found == "yes" else 0  # Assuming each trk file results in one image
            times_vid = count / num_images
            csvwriter.writerow([os.path.basename(video), trk_found, num_images, times_vid])

    return "Video to heatmap processing completed."
VideoToHeatmaps()