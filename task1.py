import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("titanic.csv")

print("First 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column
df.drop(columns=['Cabin'], inplace=True)

# Encode categorical column
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# One-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Standardize Age and Fare
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Boxplot
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare")
plt.show()

# Save cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\nData cleaning completed successfully!")
