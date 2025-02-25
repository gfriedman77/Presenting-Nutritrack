
# The below code builds a model that attempts to predict all the nutritional values. 


import os
import glob
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

os.chdir("C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project")

# Paths and settings
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Reduced resolution for computational efficiency
batch_size = 70
train_folder = "C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project/train"
test_folder = "C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project/test"

# Load and preprocess tabular data (nutritional values)
df = pd.read_csv('dishes_with_available_pics.csv')

# Normalize the nutritional values
scaler = MinMaxScaler()
nutritional_values = scaler.fit_transform(df[['weight_grams', 'cal', 'fat', 'carb', 'protein']])

# Create a dictionary mapping dish_ID to normalized nutritional values
dish_labels = {str(row['dish_ID']): nutritional_values[idx] for idx, row in df.iterrows()}

# Custom generator to load images from folders
def custom_image_generator(folder_path, labels_dict, img_height, img_width, batch_size):
    """
    Generator for loading images from a folder and corresponding labels in batches.
    
    Args:
        folder_path (str): Path to the folder containing subfolders with images.
        labels_dict (dict): Dictionary mapping dish_ID to labels.
        img_height (int): Height of resized images.
        img_width (int): Width of resized images.
        batch_size (int): Number of samples per batch.
    
    Yields:
        images, batch_labels: Arrays of images and corresponding labels.
    """
    subfolders = sorted(os.listdir(folder_path))  # Ensure consistent order
    while True:
        images = []
        batch_labels = []
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            image_files = glob.glob(os.path.join(subfolder_path, "*.jpg"))
            if not image_files:
                continue

            # Randomly sample images from the directory
            for img_path in image_files:
                img = plt.imread(img_path)
                img = np.resize(img, (img_height, img_width, 3))
                images.append(img)

                # Append corresponding label (calories)
                batch_labels.append(labels_dict[subfolder])

                # Break if batch size is met
                if len(images) == batch_size:
                    break
            if len(images) == batch_size:
                break

        # Convert to numpy arrays
        images = np.array(images, dtype="float32") / 255.0
        batch_labels = np.array(batch_labels, dtype="float32")
        yield images, batch_labels


# Training and testing data generators
train_generator = custom_image_generator(
    train_folder, dish_labels, IMG_HEIGHT, IMG_WIDTH, batch_size
)
test_generator = custom_image_generator(
    test_folder, dish_labels, IMG_HEIGHT, IMG_WIDTH, batch_size
)

# Define CNN model             
model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(),
    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(),
    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(5)
    #Dense(5, kernel_regularizer=l2(0.001))
])

model.compile(
    loss="huber",
    optimizer=RMSprop(),
    metrics=["mean_absolute_error"]
)

# Early stopping
early_stopping = EarlyStopping(
    monitor="val_mean_absolute_error",
    #patience=2,
    restore_best_weights=True
)

# Train the model
# Correct steps_per_epoch and validation_steps calculation
total_train_images = sum(len(glob.glob(os.path.join(train_folder, subfolder, "*.jpg"))) for subfolder in os.listdir(train_folder))
total_test_images = sum(len(glob.glob(os.path.join(test_folder, subfolder, "*.jpg"))) for subfolder in os.listdir(test_folder))

steps_per_epoch = total_train_images // batch_size
validation_steps = total_test_images // batch_size


history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=3,
    validation_data=test_generator,
    validation_steps=validation_steps,
    #callbacks=[early_stopping]
)

# Evaluate the model
results = []
for i, (images, labels) in enumerate(test_generator):
    if i >= validation_steps:  # Limit to the number of validation steps
        break
    predictions = model.predict(images)
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(labels)
    for pred, actual in zip(predictions, actuals):
        results.append({
            "Predicted Weight (grams)": pred[0],
            "Actual Weight (grams)": actual[0],
            "Predicted Calories": pred[1],
            "Actual Calories": actual[1],
            "Predicted Fat": pred[2],
            "Actual Fat": actual[2],
            "Predicted Carbohydrates": pred[3],
            "Actual Carbohydrates": actual[3],
            "Predicted Protein": pred[4],
            "Actual Protein": actual[4]
        })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('testing.csv', index=False)
print(results_df)

# Plotting
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()













import os
import glob
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical

# Paths and settings
os.chdir("C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project")
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Reduced resolution for computational efficiency
batch_size = 70
train_folder = "C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project/train"
test_folder = "C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project/test"

# Load and preprocess the new CSV file (dishes_with_grade.csv)
df = pd.read_csv('dishes_with_grade.csv')

# Encode Grade as integers, then one-hot encode
le = LabelEncoder()
df['Grade'] = le.fit_transform(df['Grade'])
grades = to_categorical(df['Grade'])
dish_labels = {str(row['dish_ID']): grades[idx] for idx, row in df.iterrows()}

# Calculate class weights
class_frequencies = np.array([1041, 661, 499, 383, 607, 19, 52])  # Example frequencies, replace with your actual counts
total_samples = np.sum(class_frequencies)
inverse_frequencies = total_samples / class_frequencies
normalized_class_weights = inverse_frequencies / np.sum(inverse_frequencies)  # Normalize weights

# Custom generator to load images and labels
def custom_image_generator(folder_path, labels_dict, img_height, img_width, batch_size):
    """
    Generator for loading images from a folder and corresponding labels in batches.
    
    Args:
        folder_path (str): Path to the folder containing subfolders with images.
        labels_dict (dict): Dictionary mapping dish_ID to labels.
        img_height (int): Height of resized images.
        img_width (int): Width of resized images.
        batch_size (int): Number of samples per batch.
    
    Yields:
        images, batch_labels: Arrays of images and corresponding labels.
    """
    subfolders = sorted(os.listdir(folder_path))  # Ensure consistent order
    while True:
        images = []
        batch_labels = []
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            image_files = glob.glob(os.path.join(subfolder_path, "*.jpg"))
            if not image_files:
                continue

            # Randomly sample images from the directory
            for img_path in image_files:
                img = plt.imread(img_path)
                img = np.resize(img, (img_height, img_width, 3))
                images.append(img)

                # Append corresponding label
                batch_labels.append(labels_dict[subfolder])

                # Break if batch size is met
                if len(images) == batch_size:
                    break
            if len(images) == batch_size:
                break

        # Convert to numpy arrays
        images = np.array(images, dtype="float32") / 255.0
        batch_labels = np.array(batch_labels, dtype="float32")
        yield images, batch_labels

# Training and testing data generators
train_generator = custom_image_generator(
    train_folder, dish_labels, IMG_HEIGHT, IMG_WIDTH, batch_size
)
test_generator = custom_image_generator(
    test_folder, dish_labels, IMG_HEIGHT, IMG_WIDTH, batch_size
)

# Define CNN model
num_classes = len(df['Grade'].unique())  

model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(8, kernel_size=(4, 4), activation='relu', padding='same'),
    MaxPool2D(),
    Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(),
    Flatten(),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile the model with normalized class weights
model.compile(
    loss="categorical_crossentropy",
    optimizer=RMSprop(),  
    metrics=["accuracy"]
)

# Early stopping
early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    restore_best_weights=True
)

# Train the model
total_train_images = sum(len(glob.glob(os.path.join(train_folder, subfolder, "*.jpg"))) for subfolder in os.listdir(train_folder))
total_test_images = sum(len(glob.glob(os.path.join(test_folder, subfolder, "*.jpg"))) for subfolder in os.listdir(test_folder))

steps_per_epoch = total_train_images // batch_size
validation_steps = total_test_images // batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=test_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping]
)

# Evaluate the model
results = []
for i, (images, labels) in enumerate(test_generator):
    if i >= validation_steps:  # Limit to the number of validation steps
        break
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(labels, axis=1)
    for pred, actual in zip(predicted_classes, actual_classes):
        results.append({
            "Predicted Grade": le.inverse_transform([pred])[0],
            "Actual Grade": le.inverse_transform([actual])[0]
        })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('classification_results.csv', index=False)
#print(results_df)

# Plotting
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# The below code should be used to predict grade of unseen food items.

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Path to the unseen directory
unseen_folder = "C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project/New"

# Function to load and preprocess images
def load_and_preprocess_images(folder_path, img_height, img_width):
    images = []
    image_files = []
    for img_file in glob.glob(os.path.join(folder_path, "*.jpg")):
        img = load_img(img_file, target_size=(img_height, img_width))
        img_array = img_to_array(img) / 255.0  # Normalize the image
        images.append(img_array)
        image_files.append(img_file)
    return np.array(images), image_files

# Load and preprocess unseen images
unseen_images, image_files = load_and_preprocess_images(unseen_folder, IMG_HEIGHT, IMG_WIDTH)

# Predict using the trained model
if unseen_images.size > 0:  # Check if there are images in the folder
    predictions = model.predict(unseen_images)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_grades = le.inverse_transform(predicted_classes)

    # Display predictions
    for img_file, pred_grade in zip(image_files, predicted_grades):
        print(f"Image: {os.path.basename(img_file)} - Predicted Grade: {pred_grade}")
        
    # Optionally visualize predictions
    for img, pred_grade in zip(unseen_images, predicted_grades):
        plt.imshow(img)
        plt.title(f"Predicted Grade: {pred_grade}")
        plt.axis('off')
        plt.show()
else:
    print("No images found in the 'Unseen' directory.")











