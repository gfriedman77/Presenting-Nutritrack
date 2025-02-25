# This code would list out all the dishIDs from the "dishes and nutrition.csv" that have pictures available. 
# Note: We are not using cafe 1 as it has irregular number of columns and pandas cannot read it. So, what i've 
#       done is, I've only copied the first 5 columns of data and named it "dishes and nutrition.csv".

import os
import pandas as pd

os.getcwd()

os.chdir("C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project")

df = pd.read_csv('dishes and nutrition.csv', sep='\t')

# Path to the 'realsense_overhead' directory containing folders for each dish
image_folder_path = "C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project/realsense_overhead"

# Get a list of all folders in 'realsense_overhead' directory
existing_dish_ids = set(os.listdir(image_folder_path))

# Filter the DataFrame to keep only the rows where the 'dish_ID' exists in the list of folders
df_filtered = df[df['dish_ID'].astype(str).isin(existing_dish_ids)]

df_filtered.shape

# Display the filtered DataFrame
df_filtered.to_csv('dishes_with_available_pics.csv', index = False)



# The below code basically creates a new folder "rgb_images" in the same path that would only have the 3,262 rgb images
# for us to work with. 

import os
import shutil
import pandas as pd

# Load your filtered dish_IDs
df_filtered = pd.read_csv('dishes_with_available_pics.csv')
filtered_dish_ids = set(df_filtered['dish_ID'].astype(str))

# Define the source path where the folders are located
source_folder_path = "C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project/realsense_overhead"

# Define the destination path for the RGB images
destination_folder_path = "C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project/rgb_images"

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder_path, exist_ok=True)

# Loop through each dish_ID in the filtered list
for dish_id in filtered_dish_ids:
    # Construct the path to the specific dish folder
    dish_folder = os.path.join(source_folder_path, dish_id)
    
    # Check if the dish folder exists
    if os.path.isdir(dish_folder):
        # Look for the RGB image file (assuming it has "rgb" in the file name)
        for file_name in os.listdir(dish_folder):
            if "rgb" in file_name.lower():
                # Define the full path to the RGB image file
                source_file_path = os.path.join(dish_folder, file_name)
                
                # Define the new path with dish_ID as the file name in the destination folder
                destination_file_path = os.path.join(destination_folder_path, f"{dish_id}.jpg")
                
                # Copy the RGB image to the new folder with dish_ID as the file name
                shutil.copy2(source_file_path, destination_file_path)
                break  # Move to the next dish_ID after finding the RGB file

print("RGB images have been copied and renamed successfully.")



# checking the shape of the pixels of the rgb images. so that the picture we might test it on later from a real world example also has 
# the same dimensions. 
# Image shape - 640*480

from PIL import Image

# Specify the path to one of your RGB images
image_path = "C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project/rgb_images/dish_1556572657.jpg"

# Open the image and get its dimensions
with Image.open(image_path) as img:
    width, height = img.size
    print(f"Image dimensions: {width} x {height}")


