import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Task 1: Create Dataset and Build Model ---
n_samples = 100

# Generating synthetic features
area_sqft = np.random.randint(800, 3500, n_samples)
num_bedrooms = np.random.randint(1, 6, n_samples)
age_years = np.random.randint(0, 30, n_samples)

# Creating a target price based on a realistic formula + random noise
# Price = (Area * 0.5) + (Bedrooms * 10) - (Age * 2) + random noise
price_lakhs = (area_sqft * 0.45) + (num_bedrooms * 8) - (age_years * 1.5) + np.random.normal(0, 50, n_samples)

# Prepare data for Scikit-Learn
X = pd.DataFrame({
    'area_sqft': area_sqft,
    'num_bedrooms': num_bedrooms,
    'age_years': age_years
})
y = price_lakhs

# Build Model
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

print("--- Task 1: Model Coefficients ---")
print(f"Intercept: {model.intercept_:.2f}")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature} Coefficient: {coef:.4f}")

print("\n--- Actual vs. Predicted (First 5) ---")
results_df = pd.DataFrame({'Actual': y, 'Predicted': predictions})
print(results_df.head(5))

# --- Task 2: Evaluation Metrics ---
mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

print("\n--- Task 2: Evaluation Metrics ---")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

"""
Interpretation:
- MAE: The average error in our price prediction is about {mae:.2f} lakhs.
- RMSE: Similar to MAE but punishes larger errors more; a high RMSE relative to MAE suggests outliers.
- R²: Shows that {r2*100:.1f}% of the variance in house prices is explained by our features.
"""

# --- Task 3: Residual Analysis ---
residuals = y - predictions

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=15, color='skyblue', edgecolor='black')
plt.title('Histogram of Residuals (Price Prediction Errors)')
plt.xlabel('Residual Value (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3)
plt.show()

"""
What is a Residual? 
A residual is the difference between the actual observed house price and the price predicted by the model. 
(Residual = Actual - Predicted).

What does the shape suggest?
If the histogram looks like a 'Bell Curve' (Normal Distribution) centered around zero, it suggests 
the model's errors are random and it has captured the underlying data patterns well. If it's heavily 
skewed, our model might be missing a key feature or the relationship isn't purely linear.
"""
