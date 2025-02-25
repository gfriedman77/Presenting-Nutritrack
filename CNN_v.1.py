import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
import random

os.getcwd()

os.chdir("C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project")

# Paths and Constants - each image is 480 * 640. So, if we use this, we can get the best predictions. 
# For the time being, we could reduce it to 128*128 so that it is less taxing computitionally. 
#IMG_HEIGHT = 480
#IMG_WIDTH = 640
#batch_size = 32

IMG_HEIGHT = 128
IMG_WIDTH = 128
batch_size = 70

# Load and preprocess tabular data (nutritional values)
df = pd.read_csv('dishes_with_available_pics.csv')

# Select the columns for the values to be predicted
nutritional_values = df[['weight_grams', 'cal', 'fat', 'carb', 'protein']].values

# Normalize the nutritional values
scaler = MinMaxScaler()
nutritional_values = scaler.fit_transform(nutritional_values)

# Split data into training and test sets
train_df, test_df, train_nutritional_values, test_nutritional_values = train_test_split(
    df, nutritional_values, test_size=0.1, random_state=3)

# Append `.jpg` to each dish_ID to create filenames
train_df['filename'] = train_df['dish_ID'] + ".jpg"
test_df['filename'] = test_df['dish_ID'] + ".jpg"

# Create ImageDataGenerator for image data
image_data_gen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=20,         # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,     # Randomly shift images horizontally by up to 20%
    height_shift_range=0.2,    # Randomly shift images vertically by up to 20%
    zoom_range=0.2,            # Randomly zoom in or out by up to 20%
    horizontal_flip=True,      # Randomly flip images horizontally
    brightness_range=[0.8, 1.2] # Randomly adjust brightness
)

# Prepare training image data generator
train_image_generator = image_data_gen.flow_from_dataframe(
    train_df,
    directory="C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project/rgb_images",
    x_col="filename",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode=None,  # No class labels needed for regression
    shuffle=True
)

# Prepare test image data generator
test_image_generator = image_data_gen.flow_from_dataframe(
    test_df,
    directory="C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project/rgb_images",
    x_col="filename",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)

# Custom generator to pair images and labels
def custom_data_generator(image_generator, labels):
    while True:
        images = next(image_generator)
        yield images, labels[:len(images)]  # Pair images with labels

# Prepare the training and testing data generators
train_generator = custom_data_generator(train_image_generator, train_nutritional_values)
test_generator = custom_data_generator(test_image_generator, test_nutritional_values)

# CNN
model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
    MaxPool2D(),
    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(),
    Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.35),  
    Dense(5)
])

model.compile(loss="mean_squared_error",
              optimizer='RMSprop', 
              #optimizer='RMSprop', #MAE = 0.63
              metrics=["mean_absolute_error"])

# Define early stopping criteria
early_stopping = EarlyStopping(
    monitor="val_mean_absolute_error",           # training loss
    patience=3,               
    restore_best_weights=True # Restore model weights from the epoch with the best loss
)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_image_generator),
    #steps_per_epoch=200,
    epochs=50,
    validation_data=test_generator,      
    validation_steps=len(test_image_generator), 
    callbacks=[early_stopping]          
)


### Predicting nutritional values on a random test image ###
# Number of random test samples to evaluate
num_samples = 10

# Get a batch of test images from the test generator
test_images = next(test_image_generator)

# Select 10 unique random indices from the test set
random_indices = random.sample(range(len(test_images)), num_samples)

# Prepare a list to store results
results = []

# Loop through each randomly chosen index
for img_index in random_indices:
    # Get the corresponding dish ID and test image
    dish_id = test_df.iloc[img_index]['dish_ID']
    img_array = test_images[img_index:img_index + 1]  # Select a single image

    # Predict nutritional values for the selected test image
    predicted_nutritional_values = model.predict(img_array)

    # Rescale predictions back to the original scale
    predicted_nutritional_values = scaler.inverse_transform(predicted_nutritional_values)

    # Get the actual nutritional values for the selected index
    actual_nutritional_values = scaler.inverse_transform([test_nutritional_values[img_index]])

    # Append the results to the list as a dictionary
    results.append({
        "Dish ID": dish_id,
        "Predicted Weight (grams)": predicted_nutritional_values[0][0],
        "Actual Weight (grams)": actual_nutritional_values[0][0],
        "Predicted Calories": predicted_nutritional_values[0][1],
        "Actual Calories": actual_nutritional_values[0][1],
        "Predicted Fat": predicted_nutritional_values[0][2],
        "Actual Fat": actual_nutritional_values[0][2],
        "Predicted Carbohydrates": predicted_nutritional_values[0][3],
        "Actual Carbohydrates": actual_nutritional_values[0][3],
        "Predicted Protein": predicted_nutritional_values[0][4],
        "Actual Protein": actual_nutritional_values[0][4]
    })

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Display the DataFrame
print(results_df)
results_df.to_csv('testing.csv', index = False)




import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training & validation MAE values
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()



