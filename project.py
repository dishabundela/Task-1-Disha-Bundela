import pandas as pd

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# First 5 rows
print("FIRST 5 ROWS")
print(df.head())

# Dataset info
print("\nDATASET INFO")
print(df.info())

# Shape
print("\nSHAPE OF DATASET")
print(df.shape)

# Column names
print("\nCOLUMN NAMES")
print(df.columns)

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Fill missing Age values with average age
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill missing Embarked values
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin column because many values are missing
df.drop('Cabin', axis=1, inplace=True)

# Check missing values again
print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

# Survival count
print("\nSURVIVAL COUNT")
print(df['Survived'].value_counts())

# Gender count
print("\nGENDER COUNT")
print(df['Sex'].value_counts())


import matplotlib.pyplot as plt
import seaborn as sns

# Survival Count Graph
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Gender Count Graph
sns.countplot(x='Sex', data=df)
plt.title("Gender Count")
plt.show()

# Passenger Class Graph
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Count")
plt.show()
