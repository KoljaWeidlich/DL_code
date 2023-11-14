# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 12:06:35 2023
@author: Kotti
"""

from PIL import Image
import os
from tqdm import tqdm
from tkinter import filedialog, Tk
import shutil
from multiprocessing import Pool

def is_image_corrupted(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return None
    except Exception as e:
        return file_path

def move_files(problematic_files, orig_dir, heatmap_p_dir, heatmap_d_dir, destination):
    for file in problematic_files:
        # Define a function to move a file and handle any errors
        def move_file(src, dest):
            if os.path.exists(src):
                try:
                    shutil.move(src, dest)
                    print(f"Moved {src} to {dest}")
                except Exception as e:
                    print(f"Error moving {src}: {e}")

        move_file(file, destination)  # Move the problematic file
        
        # Construct paths to associated heatmaps or original image
        basename = os.path.basename(file)
        if "_heatmap_" not in basename:
            heatmap_p_file = os.path.join(heatmap_p_dir, basename.replace("_frame_", "_heatmap_P_frame_"))
            heatmap_d_file = os.path.join(heatmap_d_dir, basename.replace("_frame_", "_heatmap_D_frame_"))
            move_file(heatmap_p_file, destination)
            move_file(heatmap_d_file, destination)
        else:
            orig_file_basename = basename.replace("_heatmap_P_frame.png", "_frame_").replace("_heatmap_D_frame.png", "_frame_")
            orig_file = os.path.join(orig_dir, orig_file_basename)
            move_file(orig_file, destination)
            # In case the other heatmap is in a different folder, construct its path too
            other_heatmap_file = os.path.join(
                heatmap_p_dir if "heatmap_D" in basename else heatmap_d_dir,
                orig_file_basename.replace("_frame_", "_heatmap_P_frame.png" if "heatmap_D" in basename else "_heatmap_D_frame_")
            )
            move_file(other_heatmap_file, destination)

def check_images(directory):
    all_files = [os.path.join(root, file) 
                 for root, _, files in os.walk(directory) 
                 for file in files 
                 if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    with Pool() as pool:
        results = list(tqdm(pool.imap(is_image_corrupted, all_files), total=len(all_files)))
    
    problematic_images = [file_path for file_path in results if file_path is not None]
    return problematic_images

def main():
    root = Tk()
    root.withdraw()
    
    # Ask for each specific directory
    orig_dir = filedialog.askdirectory(title="Select directory containing original images")
    heatmap_p_dir = filedialog.askdirectory(title="Select directory containing P heatmaps")
    heatmap_d_dir = filedialog.askdirectory(title="Select directory containing D heatmaps")
    destination = filedialog.askdirectory(title="Select directory to move problematic images")
    
    num_checks = int(input("How many times would you like to check the dataset for corrupted images? "))
    
    all_problematic_files = set()
    
    for i in range(num_checks):
        print(f"Check number {i+1}/{num_checks}")
        
        # Perform checks in each directory
        problematic_files_orig = check_images(orig_dir)
        problematic_files_heatmap_p = check_images(heatmap_p_dir)
        problematic_files_heatmap_d = check_images(heatmap_d_dir)
        
        # Combine all problematic files from this check
        problematic_files = set(problematic_files_orig + problematic_files_heatmap_p + problematic_files_heatmap_d)
        all_problematic_files.update(problematic_files)
        
        if not problematic_files:
            print("No corrupted images found in this check.")
            # break  # Exit the loop if no corrupted images are found in the current check
        
        print(f"Total problematic images found in check {i+1}: {len(problematic_files)}")
    
    # After checking for problematic images
    if all_problematic_files:
        choice = input("Do you want to move the problematic files to a separate folder? (yes/no): ").lower()
        if choice == 'yes':
            move_files(all_problematic_files, orig_dir, heatmap_p_dir, heatmap_d_dir, destination)
    
    print(f"Total problematic images found across all checks: {len(all_problematic_files)}")

if __name__ == "__main__":
    main()

