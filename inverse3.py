# import needed libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import idct

# Make the paths to .csv files
# Variables
train_csv = "../FullData/train.csv"
dev_csv = "../FullData/dev.csv"
eval_csv = "../FullData/eval.csv"

# Function to load data
# takes in the File path and the "Is_eval"
def load_data(file_path, is_eval=False):
    # Set a variable for the read csv, skip the first row (it is a legend)
    # Print loaded .csv with the shape or the amount of rows and columns                       
    df = pd.read_csv(file_path, skiprows=1, header=None)
    print(f"Loaded {os.path.basename(file_path)} with shape: {df.shape}")

    # If it is not a evaluation file
    # Take the first column and set it equal to the labels
    # find all the unique Labels
    # print them out
    if not is_eval:
        labels = df.iloc[:, 0]
        unique_labels = sorted(labels.unique())
        print(f"Unique labels found: {unique_labels}")
        if not all(label in range(9) for label in unique_labels):
            raise ValueError(f"Invalid labels found. Labels must be in range 0-8")

    return df

# Load 
print("\nLoading training data...")
train_data = load_data(train_csv)
dev_data = load_data(dev_csv)

def inverse_dct_2d(dct_coeffs):
    """Apply 2D inverse DCT."""
    return idct(idct(dct_coeffs, axis=0, norm='ortho'), axis=1, norm='ortho')

target_labels = [0, 2, 3, 5 ,6, 8]# [0, 2, 3, 5, 6, 8]  # Example labels you want to visualize
samples_to_plot = 10 # Number of samples per class
plotted_count = {label: 0 for label in target_labels}

fig, axs = plt.subplots(len(target_labels), samples_to_plot, figsize=(15, 10))

for index, row in dev_data.iterrows():
    label = row[0]
    if label in target_labels and plotted_count[label] < samples_to_plot:
        features = row[1:].values
        red_dct = features[0:1024].reshape(32, 32)
        green_dct = features[1024:2048].reshape(32, 32)
        blue_dct = features[2048:3072].reshape(32, 32)

        red = inverse_dct_2d(red_dct)
        green = inverse_dct_2d(green_dct)
        blue = inverse_dct_2d(blue_dct)

        rgb_image = np.stack([red, green, blue], axis=-1)
        rgb_image -= rgb_image.min()
        rgb_image /= rgb_image.max()

        row_index = target_labels.index(label)
        ax = axs[row_index, plotted_count[label]]
        ax.imshow(rgb_image)
        ax.set_title(f"Label: {label}", fontsize=8, fontweight='bold', pad=5)
        ax.axis('off')
        plotted_count[label] += 1

plt.tight_layout()
plt.show()