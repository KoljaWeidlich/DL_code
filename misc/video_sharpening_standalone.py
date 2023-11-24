# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:15:54 2023

@author: Kolja Weidlich

In this script:

    -The process_video function takes a tuple of arguments (video_path, save_path).
    -The sharpen_videos function prepares a list of arguments for each video file, which are then processed in parallel using a multiprocessing Pool.
    -tqdm is used to display a progress bar, tracking the completion of video processing tasks.

"""

import cv2
import numpy as np
from tkinter import filedialog, Tk
import os
from multiprocessing import Pool
from tqdm import tqdm

# Define the sharpening kernel (convolution matrix)
""" This kernel is applied to each pixel of the image 
    Center Value (5 in this case): This value is multiplied with the pixel value at the center of the kernel.
    A value greater than 1 enhances the contrast of the center pixel relative to its neighbors,
    making it stand out more and creating a sharpening effect.

    Surrounding Values (-1 in this case): These values are multiplied with the neighboring pixels. 
    Negative values here mean that the neighboring pixels will be subtracted from the center pixel, 
    further enhancing the contrast and edges. This subtraction emphasizes the differences in intensity at the edges,
    making them appear sharper.

    Corner Values (0 in this case): The corners of the kernel are set to zero, 
    meaning they don't contribute to the convolution process.
    This choice focuses the sharpening effect on the direct neighbors of each pixel, not on the diagonal neighbors,
    maintaining a balance in the sharpening process."""
sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])

def sharpen_frame(frame):
    """ Apply sharpening filter to a single frame """
    return cv2.filter2D(frame, -1, sharpening_kernel)

def process_video(args):
    """ Process the video with image sharpening and save it """
    video_path, save_path = args
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        sharpened_frame = sharpen_frame(frame)
        out.write(sharpened_frame)

    cap.release()
    out.release()
    return video_path

def sharpen_videos():
    root = Tk()
    root.withdraw()

    video_files = filedialog.askopenfilenames(title="Select video files", filetypes=[("Video files", "*.mp4;*.avi")])
    save_directory = filedialog.askdirectory(title="Select a directory to save sharpened videos")
    if not save_directory:
        print("No save directory selected. Exiting.")
        return

    # Prepare arguments for multiprocessing
    args = [(video_file, os.path.join(save_directory, os.path.basename(video_file))) for video_file in video_files]

    # Process videos in parallel using multiprocessing
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(process_video, args), total=len(args), desc="Processing Videos"):
            pass

    print("All videos have been processed.")

if __name__ == "__main__":
    sharpen_videos()
