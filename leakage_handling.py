import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# --- Task 1: Reproduce and Identify Leakage ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)


print("--- Task 1 — Reproduce and Identify Leakage ---\n")
print(f"Train Accuracy: {model.score(X_train,y_train)}")
print(f"Test Accuracy:  {model.score(X_test,y_test)}")

print("Problem: Scaling the whole dataset leaked test statistics into the training process.\n")

# --- Task 2: Fix the Workflow Using a Pipeline ---

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")

print("\n--- Task 2 — Fix the Workflow Using a Pipeline ---\n")
print("Scores per fold: ",scores)
print(f"Mean Accuracy: {scores.mean().round(2)}")
print(f"Std Deviation: {scores.std().round(2)}")

# --- Task 3: Experiment with Decision Tree Depth ---
depths = [1, 5, 20]

print("\n--- Task 3: Decision Tree Depth Comparison ---\n")

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train,y_train)
    test_acc = model.score(X_test,y_test)
    
    print(f"Depth {depth:2d}  | Train: {train_acc:.2f}  Test: {test_acc:.2f}")

print("\nAnalysis: Depth 5 remains the best balancer between bias (underfit) and variance (overfit).")
