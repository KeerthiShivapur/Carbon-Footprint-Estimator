# ✅ Carbon Footprint Estimator using Retail Dataset with User Input

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
df = pd.read_csv("retail_carbon_data_1000.csv")

# Drop non-numeric 'store_id'
df.drop("store_id", axis=1, inplace=True)

# -------------------------------
# STEP 2: Prepare Features
# -------------------------------
X = df.drop("carbon_footprint_kg", axis=1)
y = df["carbon_footprint_kg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# STEP 3: Train XGBoost Model
# -------------------------------
model = XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# STEP 4: Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)
print("\n--- MODEL EVALUATION ---")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------------
# STEP 5: Save Model
# -------------------------------
joblib.dump(model, "retail_carbon_model.pkl")

# -------------------------------
# STEP 6: Predict from User Input
# -------------------------------
try:
    transport = float(input("Enter Transport Emission (km): "))
    electricity = float(input("Enter Electricity Emission (kWh): "))
    packaging = float(input("Enter Packaging Emission (kg): "))

    sample_input = np.array([[transport, electricity, packaging]])
    predicted_cf = model.predict(sample_input)[0]
    print(f"\n✅ Estimated Carbon Footprint: {predicted_cf:.2f} kg CO₂")

except ValueError:
    print("❌ Invalid input. Please enter numeric values only.")

# -------------------------------
# STEP 7: Plot Actual vs Predicted
# -------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Carbon Footprint")
plt.ylabel("Predicted Carbon Footprint")
plt.title("Actual vs Predicted Carbon Footprint")
plt.grid(True)
plt.tight_layout()
plt.show()
