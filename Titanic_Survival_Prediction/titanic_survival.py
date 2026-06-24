import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("train.csv")

# Select required columns
data = data[["Pclass", "Sex", "Age", "Survived"]]

# Fill missing ages
data["Age"] = data["Age"].fillna(data["Age"].mean())

# Convert gender to numbers
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

# Features and target
X = data[["Pclass", "Sex", "Age"]]
y = data["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
print("Titanic Model Accuracy:", accuracy_score(y_test, predictions))