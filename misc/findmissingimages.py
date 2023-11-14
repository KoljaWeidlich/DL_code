# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:20:10 2023

@author: Kotti
"""

import os
from tkinter import filedialog, Tk 

def get_file_list(directory, file_extension='.png'):
    return set([f for f in os.listdir(directory) if f.endswith(file_extension)])

def find_missing_files(original_dir, heatmap_p_dir, heatmap_d_dir):
    # Get lists of files without extension
    original_files = get_file_list(original_dir)
    heatmap_p_files = get_file_list(heatmap_p_dir)
    heatmap_d_files = get_file_list(heatmap_d_dir)
    
    # Adjust the names of the heatmap files to match the original files for comparison
    heatmap_p_files_adjusted = {f.replace("_heatmap_P_frame_", "_frame_") for f in heatmap_p_files}
    heatmap_d_files_adjusted = {f.replace("_heatmap_D_frame_", "_frame_") for f in heatmap_d_files}

    # Find missing files by comparing with the original files
    missing_in_heatmap_p = original_files - heatmap_p_files_adjusted
    missing_in_heatmap_d = original_files - heatmap_d_files_adjusted
    missing_in_original = (heatmap_p_files_adjusted | heatmap_d_files_adjusted) - original_files

    return missing_in_original, missing_in_heatmap_p, missing_in_heatmap_d


# Define your directories here
root = Tk()
root.withdraw()

original_dir = filedialog.askdirectory(title="Select directory containing original images")
heatmap_p_dir = filedialog.askdirectory(title="Select directory containing P heatmaps")
heatmap_d_dir = filedialog.askdirectory(title="Select directory containing D heatmaps")

missing_files = find_missing_files(original_dir, heatmap_p_dir, heatmap_d_dir)

print('Missing files in original:', missing_files[0])
print('Missing files in heatmap P:', missing_files[1])
print('Missing files in heatmap D:', missing_files[2])
