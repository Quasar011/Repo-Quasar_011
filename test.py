import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load the provided CSV file into a pandas DataFrame
data = pd.read_csv("/content/dataset-of-100-people.csv")

# Step 2: Data Preprocessing
# Drop the 'Languages Used' column before one-hot encoding
X = data.drop(columns=["Languages Used"])

# One-hot encoding for categorical variable 'Field'
X = pd.get_dummies(X, columns=['Field'])

# Step 3: Split Data (Remove redundant splitting)
y = data["Languages Used"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Model Selection (Unchanged)
model = RandomForestClassifier(random_state=42)

# Step 5: Model Training (Unchanged)
model.fit(X_train, y_train)

# Step 6: Model Evaluation (Unchanged)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 7: Prediction (Unchanged)
# Creating a new data point for prediction
X_new = pd.DataFrame({
    "Field": ["Your_Field"],
    # Add other fields as needed
})

# Make predictions
predicted_language = model.predict(X_new)
print("Predicted Language:", predicted_language)

# Step 8: Model Persistence (Unchanged)
joblib.dump(model, 'best_model.pkl')

# Step 9: Load Model (Unchanged)
loaded_model = joblib.load('best_model.pkl')

# Step 10: Make Predictions using the loaded model (Unchanged)
predicted_language_new = loaded_model.predict(X_new)
print("Predicted Language for New Data Point:", predicted_language_new)
