import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =========================
# Load Data & Clean Headers
# =========================

df = pd.read_csv("final_merged_energy_weather_population.csv", sep=None, engine='python')
df.columns = df.columns.str.strip()

if '#' in df.columns:
    df = df.drop(columns=['#'])

df["Month"] = pd.to_datetime(df["Month"])


# =========================
# Keep Valid Boroughs
# =========================

valid_boroughs = [
    "BRONX",
    "BROOKLYN",
    "MANHATTAN",
    "QUEENS",
    "STATEN ISLAND"
]

df = df[df["Borough"].isin(valid_boroughs)].copy()
df = df.sort_values(["Borough", "Month"]).reset_index(drop=True)


# =========================
# Feature Engineering & Target Recalculation
# =========================

df["Month_Num"] = df["Month"].dt.month
df["Year"] = df["Month"].dt.year

# 1. Calculate true utility billing rate per KWH ($0.10 to $0.25 typically)
df["Cost_Per_KWH"] = df["Current Charges"] / df["Consumption (KWH)"]

# 2. Base the baseline target on a single individual's monthly utility footprint (300 KWH/month)
df["Realistic_Bill_Per_Person"] = df["Cost_Per_KWH"] * 300.0


# =========================
# Cyclical Seasonality
# =========================

df["Sin_Month"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
df["Cos_Month"] = np.cos(2 * np.pi * df["Month_Num"] / 12)


# =========================
# Encode Borough
# =========================

le = LabelEncoder()
df["Borough_Encoded"] = le.fit_transform(df["Borough"])


# =========================
# Replace Infinite Values
# =========================

df = df.replace([np.inf, -np.inf], np.nan)


# =========================
# Define Features & Target
# =========================

# Features are restricted to Month and Borough to establish the baseline rate per person
features = [
    "Sin_Month",
    "Cos_Month",
    "Borough_Encoded"
]

target = "Realistic_Bill_Per_Person"


# =========================
# Drop Missing Rows
# =========================

df = df.dropna(subset=features + [target]).copy()


# =========================
# Prepare X and y
# =========================

X = df[features]
y = np.log1p(df[target])


# =========================
# Time-Based Train/Test Split
# =========================

train_mask = df["Year"] <= 2022
test_mask = df["Year"] >= 2023

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]


# =========================
# Scale Features
# =========================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# Ridge Regression
# =========================

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

y_pred_log = ridge.predict(X_test_scaled)


# =========================
# Convert Back From Log & Clip
# =========================

y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)
y_pred = np.clip(y_pred, 0, np.expm1(y_train).max())


# =========================
# Evaluation Metrics Summary
# =========================

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)

print("\n--- Evaluation Summary (Per Person Baseline) ---")
print(f"RMSE (Error Margin): ${rmse:,.2f}")
print(f"MAE (Avg Deviation):  ${mae:,.2f}")


# =========================
# Save Predictions & Model Components
# =========================

results = df[test_mask][["Borough", "Month", target]].copy()
results["Predicted_Bill_Per_Person"] = y_pred

results.to_csv("model_predictions.csv", index=False)

joblib.dump(ridge, "ridge_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModel trained successfully for single-person baselines!")







import joblib
import numpy as np

# 1. Load the Model Math Constants
ridge = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Extract explicit mathematical parameters from binary components
beta_0 = ridge.intercept_
beta = ridge.coef_
means = scaler.mean_
std_devs = scaler.scale_

print(f"beta_0 = {beta_0}")
print(f"beta = {beta}")
print(f"means = {means}")
print(f"std_devs = {std_devs}")


def calculate_household_bill(borough_name, month_num, household_people_0_to_10):
    """
    Computes the mathematical equation of the model and scales it to a household size.
    
    Parameters:
    - borough_name (str): 'BRONX', 'BROOKLYN', 'MANHATTAN', 'QUEENS', or 'STATEN ISLAND'
    - month_num (int): 1 to 12
    - household_people_0_to_10 (float/int): Number of people in household (Strictly 0 to 10)
    """
    # Enforce your strict household size validation range
    if not (0 <= household_people_0_to_10 <= 10):
        raise ValueError("Household population input must be strictly between 0 and 10 people.")
    if not (1 <= month_num <= 12):
        raise ValueError("Month input must be an integer between 1 and 12.")
        
    # Handle the 0-person edge case cleanly
    if household_people_0_to_10 == 0:
        return 0.0
        
    # Transform categorical text into its correct numerical index mapping
    try:
        borough_encoded = le.transform([borough_name.upper().strip()])[0]
    except ValueError:
        raise ValueError(f"Borough must be one of: {list(le.classes_)}")
        
    # Apply cyclical monthly seasonality functions
    sin_month = np.sin(2 * np.pi * month_num / 12)
    cos_month = np.cos(2 * np.pi * month_num / 12)
    
    # Collect features into a uniform 1D NumPy array (Matching the 3 features used in training)
    raw_features = np.array([sin_month, cos_month, float(borough_encoded)])
    
    # --- THE MATHEMATICAL EQUATION ---
    
    # Step A: Standardize features manually using training means and variances
    standardized_features = (raw_features - means) / std_devs
    
    # Step B: Compute linear dot-product combo + Intercept bias
    predicted_log_y = beta_0 + np.dot(beta, standardized_features)
    
    # Step C: Exponentiate to reverse log1p scaling back to dollar values per individual
    base_cost_per_person = np.expm1(predicted_log_y)
    base_cost_per_person = max(0.0, base_cost_per_person)
    
    # Step D: Scale up the single-person baseline to match the total household headcount
    total_household_bill = base_cost_per_person * household_people_0_to_10
    
    return total_household_bill


# ==========================================================
# INTERACTIVE DEMO (Simulating User Input)
# ==========================================================
if __name__ == "__main__":
    # Test parameters restricted within household ranges
    user_borough = "BROOKLYN"
    user_month = 8                   # August (Peak summer cooling season)
    user_household_size = 4          # 4 People living in the household (Range: 0 to 10)
    
    estimated_bill = calculate_household_bill(user_borough, user_month, user_household_size)
    
    print("=== Household Model Live Inference ===")
    print(f"Selected Borough:    {user_borough}")
    print(f"Selected Month:      {user_month} (August)")
    print(f"Household Size:      {user_household_size} people")
    print(f"Predicted Total Bill: ${estimated_bill:.2f} / month")
