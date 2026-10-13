import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Create dummy categorical data
data = {
    'Age_Group': ['Young', 'Middle', 'Senior', 'Young', 'Middle', 'Senior', 'Young', 'Middle'],
    'Income': ['High', 'Low', 'Medium', 'Medium', 'High', 'Low', 'Low', 'High'],
    'Student': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes'],
    'Purchased': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes']
}

df = pd.DataFrame(data)

# 2. Separate features (X) and target label (y)
X = df.drop(columns=['Purchased'])
y = df['Purchased']

# 3. Split into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. Preprocessing: One-Hot Encode categorical features
categorical_features = ['Age_Group', 'Income', 'Student']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 5. Build a scikit-learn Pipeline (Preprocessing + Model)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# 6. Train the model
model_pipeline.fit(X_train, y_train)

# 7. Predict on the test set
y_pred = model_pipeline.predict(X_test)

# 8. Evaluate results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
