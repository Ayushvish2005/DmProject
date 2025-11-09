import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load a real-world dataset
# We use the built-in Titanic dataset from Seaborn for this example
print("Loading data...")
df = sns.load_dataset('titanic')

# 2. Simple Data Preprocessing
# Fill missing age values with the average age
df['age'].fillna(df['age'].mean(), inplace=True)

# Fill missing embarked values with the most common port
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Convert categorical features into numbers
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 3. Define Features (X) and Target (y)
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

X = df[features]
y = df[target]

# 4. Train the Model
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

# 5. Save the Model
# This is the crucial step!
model_filename = 'titanic_model.joblib'
joblib.dump(model, model_filename)

print(f"Model saved successfully as {model_filename}!")
