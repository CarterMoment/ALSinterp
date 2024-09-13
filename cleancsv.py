import pandas as pd

# Load the CSV file
df = pd.read_csv('gesture_data.csv')

# Display the first few rows to inspect the data
print(df.head())

# Filter out rows that contain erroneous data
# Example: Remove rows where the gesture label is 'wrong_gesture'
cleaned_df = df[df['gesture'] != 'letter_c']

# Optionally, remove rows based on other conditions like incorrect landmark values
# Example: Remove rows where x1 > 1 (invalid data)
# cleaned_df = df[df['x1'] <= 1]

# Save the cleaned data to a new CSV file (or overwrite the original if you prefer)
cleaned_df.to_csv('cleaned_gesture_data.csv', index=False)

print("Cleaned CSV saved as 'cleaned_gesture_data.csv'")