import os
import pandas as pd

os.chdir('C:/Users/abhij/OneDrive/Bentley/Bentley University/Fall 2024/MA 707 Machine Learning/Group project')
os.getcwd()

df = pd.read_csv('dishes.csv')
df.columns

import pandas as pd

df['Fat to Protein'] = pd.to_numeric(df['Fat to Protein'], errors='coerce')
df['Carb to Protein'] = pd.to_numeric(df['Carb to Protein'], errors='coerce')

# Assuming your DataFrame is named df
# Define a function to calculate points based on the given criteria
def calculate_points(row):
    points = 0
    # Check Fat-to-Protein ratio
    if 0.5 <= row['Fat to Protein'] <= 2:
        points += 10
    elif row['Fat to Protein'] > 5:
        points -= 10
    
    # Check Carb-to-Protein ratio
    if 0.5 <= row['Carb to Protein'] <= 2:
        points += 10
    elif row['Carb to Protein'] > 3:
        points -= 5
    
    return points

# Apply the function to each row and create a new column
df['Points'] = df.apply(calculate_points, axis=1)

def assign_grade(score):
    if -15 <= score <= -11:
        return 'F'
    elif -10 <= score <= -6:
        return 'D'
    elif -5 <= score <= -1:
        return 'C'
    elif 0 <= score <= 4:
        return 'B'
    elif 5 <= score <= 9:
        return 'B+'
    elif 10 <= score <= 14:
        return 'A'
    elif 15 <= score <= 20:
        return 'A+'
    else:
        return None  # For scores outside the expected range

# Apply the function to assign grades
df['Grade'] = df['Points'].apply(assign_grade)

df['Points'].value_counts()
df['Grade'].value_counts()

df.to_csv('dishes_with_score.csv')
