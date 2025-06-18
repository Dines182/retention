import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Step 1: Load the dataset
df = pd.read_excel("NEWW All.xlsx")

# Step 2: Feature Engineering

# Previous year enrolment
df["Previous_Year_Enrolment"] = df.sort_values("Year").groupby("School Name")["Enrolments"].shift(1)

# Change from last year
df["Enrolment_Change"] = df["Enrolments"] - df["Previous_Year_Enrolment"]

# Region average enrolment
df["Region_Enrolment_Avg"] = df.groupby("Region Name")["Enrolments"].transform("mean")

# Whether school is above regional average
df["Above_Region_Enrolment"] = (df["Enrolments"] > df["Region_Enrolment_Avg"]).astype(int)

# Step 3: Define features to use (no attendance-based ones!)
features = [
    "Total Retention",
    "Retention_Gap",
    "Teacher Retention",
    "Non-Teacher Retention",
    "Previous_Year_Enrolment",
    "Region_Enrolment_Avg",
    "Above_Region_Enrolment"
]

target = "Enrolments"

# Drop rows with missing values (especially for lagged enrolment)
df_model = df.dropna(subset=features + [target])

X = df_model[features]
y = df_model[target]

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("📊 Enrolment Prediction Results:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

# Step 7: Feature Importances
print("\n🔍 Feature Importances:")
for feature, score in zip(features, model.feature_importances_):
    print(f"{feature}: {score:.4f}")

# Step 8: Predict for 2024 using 2023 data
df_2023 = df[df["Year"] == 2023].copy()
df_2023 = df_2023.dropna(subset=features)
X_2023 = df_2023[features]

df_2023["Predicted_Enrolments_2024"] = model.predict(X_2023).round().astype(int)
df_2023["Predicted_Year"] = 2024

# Step 9: Export prediction
df_result = df_2023[["School Name", "Region Name", "Predicted_Year", "Predicted_Enrolments_2024"]]
df_result.rename(columns={
    "Predicted_Year": "Year",
    "Predicted_Enrolments_2024": "Enrolments"
}, inplace=True)

print(df_result.columns)


df_result.to_excel("Predicted_Enrolments_2024.xlsx", index=False)
print("\nPrediction saved to 'Predicted_Enrolments_2024.xlsx'")


