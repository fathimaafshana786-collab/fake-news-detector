# Step 1: Import libraries
import pandas as pd
import os

print("Loading dataset...")

# Step 2: Load Fake and Real news
fake_df = pd.read_csv("data/Fake.csv")
real_df = pd.read_csv("data/True.csv")

# Step 3: Add labels
# 1 = Fake, 0 = Real
fake_df['label'] = 1
real_df['label'] = 0

# Step 4: Combine both
df = pd.concat([fake_df, real_df], ignore_index=True)

# Step 5: Check the data
print("\nDataset size:")
print(f"Total samples : {len(df)}")
print(f"Fake news     : {len(fake_df)}")
print(f"Real news     : {len(real_df)}")

print("\nColumns in dataset:")
print(df.columns.tolist())

print("\nFirst example:")
print(df.head(2))

print("\nLabel distribution:")
print(df['label'].value_counts())
# Step 6: Clean the data
print("\nCleaning data...")

# Combine title and text into one column
df['content'] = df['title'] + " " + df['text']

# Remove unnecessary columns
df = df[['content', 'label']]

# Remove duplicates
df = df.drop_duplicates()

# Remove empty rows
df = df.dropna()

print(f"After cleaning : {len(df)} samples")

# Step 7: Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Data shuffled! ✅")

# Step 8: Split into Train and Test
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

print(f"\nTrain samples : {len(train_df)}")
print(f"Test samples  : {len(test_df)}")

# Step 9: Save cleaned data
train_df.to_csv("data/train_clean.csv", index=False)
test_df.to_csv("data/test_clean.csv", index=False)

print("\n✅ Cleaned data saved to data folder!")
print("Phase 2 Complete! 🎉")