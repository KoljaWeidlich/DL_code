# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 22:01:10 2023

@author: Kotti
"""

import tensorflow as tf
from tkinter import filedialog, Tk
root = Tk()
root.withdraw()
directory =  filedialog.askdirectory(title="Select directory")
# Replace 'path_to_saved_model' with the path to the directory containing the saved model
model = tf.keras.models.load_model(directory)
model.summary()
config = model.get_config()
print(config)
# for layer in model.layers:
#     weights = layer.get_weights()
#     print(weights)
print("Model's input tensors:", model.inputs)
print("Model's output tensors:", model.outputs)
