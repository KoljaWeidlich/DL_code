# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:33:46 2023

@author: Kotti
"""

import os
import cv2
import numpy as np
from tkinter import filedialog, Tk
from scipy.io import loadmat
import multiprocessing
from tqdm import tqdm

# from scipy.interpolate import interp1d

def extract_MTJ_positions_from_mat(mat_file):
    mat_data = loadmat(mat_file)
    s = mat_data['s']
    video_name_array = s['model'][0, 0]['fName'][0]
    actual_video_path = video_name_array[0][0]
    
    # Get the video name and change the extension to .avi if it is .mp4
    video_name = os.path.basename(actual_video_path)
    video_name_without_ext, ext = os.path.splitext(video_name)
    if ext.lower() == '.mp4':
        video_name = video_name_without_ext + '.avi'
    
    # Adjusting how we access the nested structure of support_points
    support_points_nested = s['MTJ'][0, 0]['supportPoints']
    support_points = support_points_nested[0][0]
    
    # Interpolating the MTJ positions to handle NaN values
    valid_indices = ~np.isnan(support_points[:, 0])
    valid_x = support_points[valid_indices, 0]
    valid_y = support_points[valid_indices, 1]
    interpolated_x = np.interp(np.arange(len(support_points)), np.where(valid_indices)[0], valid_x)
    interpolated_y = np.interp(np.arange(len(support_points)), np.where(valid_indices)[0], valid_y)

    mtj_positions = list(zip(interpolated_x, interpolated_y))

    return video_name, mtj_positions

def generate_heatmap(image_shape, coord, base_sigma=20):
    x = np.arange(0, image_shape[1], 1, float)  # Width
    y = np.arange(0, image_shape[0], 1, float)  # Height
    y = y[:, np.newaxis]
    # 2D Gaussian centered at the coordinate
    # computes the value of a 2D Gaussian function centered at the provided coordinate (coord). 
    # x​0 and y0​ are coordinates around which the heatmap is centered (i.e., coord). Sigma is the spread of the Gaussian function.
    
    # Compute sigma based on image width
    sigma = (image_shape[1]/800) * base_sigma
    
    # Heatmap for MTJ
    heatmap = np.exp(-((x - coord[0]) ** 2 + (y - coord[1]) ** 2) / (2 * sigma ** 2))
       
    # Normalize the heatmap
    # heatmap is normalized by dividing every pixel value by the maximum value in the heatmap. 
    # This ensures that pixel values are in the range [0,1], with 1 being the intensity at the center of the heatmap.
    heatmap /= np.max(heatmap)
    return heatmap

# Unifies pathsyntax to forward slashes (sometimes useful on Windows natives)
def forward_slash_path(path):
    return path.replace("\\", "/")

def process_video(args):
    mat_file, save_directory, heatmap_directory, csv_directory, original_image_directory = args
    video_name, mtj_positions = extract_MTJ_positions_from_mat(mat_file)
    mat_file_directory = os.path.dirname(mat_file)
    video_file_path = os.path.join(mat_file_directory, video_name)

    if not os.path.exists(video_file_path):
        print(f"Warning: Video file {video_name} not found for mat file {os.path.basename(mat_file)}. Skipping...")
        return

    video_name_short = os.path.basename(video_file_path).split('.')[0]
    csv_path = os.path.join(csv_directory, f"{video_name_short}.csv")

    with open(csv_path, 'w') as csv_file:
        csv_file.write("Video_Name,Framenumber,MTJ_x,MTJ_y\n")

        cap = cv2.VideoCapture(video_file_path)
        ret, frame = cap.read()
        height = frame.shape[0]
        mtj_positions = [(x, height - y) for x, y in mtj_positions]
        frame_count = 1
        
        while ret:
            mtj_x, mtj_y = mtj_positions[frame_count - 1]
            heatmap = generate_heatmap(frame.shape, (mtj_x, mtj_y))

            # Crop the original and heatmap images
            # Crop from top and both sides
            cropped_frame = frame[:-180, 190:-190]
            cropped_heatmap = heatmap[:-180, 190:-190]

            # Resize the cropped images to 608x608
            resized_frame = cv2.resize(cropped_frame, (608, 608))
            resized_heatmap = cv2.resize(cropped_heatmap, (608, 608))

            # Save the cropped and resized frame and heatmap
            original_image_filename = os.path.join(
                original_image_directory, f"{video_name_short}_frame_{frame_count}.png")
            heatmap_filename = os.path.join(
                heatmap_directory, f"{video_name_short}_frame_{frame_count}_heatmap.png")

            cv2.imwrite(original_image_filename, resized_frame)
            cv2.imwrite(heatmap_filename,
                        (resized_heatmap * 255).astype(np.uint8))

            ret, frame = cap.read()
            frame_count += 1

        cap.release()


def VideoToHeatmaps_MTJ():
    root = Tk()
    root.withdraw()

    mat_files = filedialog.askopenfilenames(title="Select .mat files", filetypes=[("MAT files", "*.mat")])
    save_directory = filedialog.askdirectory(title="Select a directory to save images and CSVs")
    if not save_directory:
        save_directory = os.path.dirname(mat_files[0])

    heatmap_directory = os.path.join(save_directory, "heatmaps_MTJ")
    os.makedirs(heatmap_directory, exist_ok=True)

    csv_directory = os.path.join(save_directory, "csv_files")
    os.makedirs(csv_directory, exist_ok=True)

    original_image_directory = os.path.join(save_directory, "original_images")
    os.makedirs(original_image_directory, exist_ok=True)

    # Create a list of arguments for each process
    args_list = [(mat_file, save_directory, heatmap_directory, csv_directory, original_image_directory) for mat_file in mat_files]

    # Use tqdm with multiprocessing
    with multiprocessing.Pool() as pool:
        list(tqdm(pool.imap(process_video, args_list), total=len(args_list)))

    return "Video to heatmap processing completed."

if __name__ == "__main__":
    VideoToHeatmaps_MTJ()