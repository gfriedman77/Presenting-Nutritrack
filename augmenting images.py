import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Load the CSV file
df = pd.read_csv('dishes_with_available_pics.csv')  # Update with the correct path if needed

# Paths and settings
original_image_dir = "C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project/rgb_images"
augmented_image_dir = "C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project/augmented_images"

# Ensure the augmented directory exists
os.makedirs(augmented_image_dir, exist_ok=True)

# Image settings
IMG_HEIGHT, IMG_WIDTH = 300, 300
num_augmentations = 10  # Number of augmentations per image

# Realistic augmentation generator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,         # Small rotation to avoid unnatural angles
    width_shift_range=0.05,    # Small shift to add subtle position variety
    height_shift_range=0.05,   # Small shift vertically
    zoom_range=0.1,            # Slight zoom in/out
    brightness_range=[0.9, 1.1] # Light brightness adjustment
)

# Loop through each dish in the dataframe
for _, row in df.iterrows():
    dish_id = row['dish_ID']
    filename = f"{dish_id}.jpg"
    img_path = os.path.join(original_image_dir, filename)
    
    # Load and resize the image
    img = Image.open(img_path).resize((IMG_HEIGHT, IMG_WIDTH))
    
    # Convert image to array format for augmentation
    img_array = np.array(img).reshape((1, IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Create a subdirectory for each dish
    dish_folder = os.path.join(augmented_image_dir, dish_id)
    os.makedirs(dish_folder, exist_ok=True)
    
    # Save the original image
    img.save(os.path.join(dish_folder, f"{dish_id}_0.jpg"))
    
    # Generate and save augmented images
    for i, aug_img in enumerate(datagen.flow(img_array, batch_size=1)):
        if i >= num_augmentations:
            break
        aug_img = Image.fromarray((aug_img[0] * 255).astype(np.uint8))
        aug_img.save(os.path.join(dish_folder, f"{dish_id}_{i+1}.jpg"))
