import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_excel("NEWW All.xlsx")

print(df.head())

#Dropping two features to avoid multicollinearity Teacher and Non-Teacher Retention as we have Total Retention
###Used Linear Regression 1st 

df_drop = df.drop(columns=["Teacher Retention", "Non-Teacher Retention"])
print(df_drop.columns)

features = ["Enrolments", "Total Retention", "Retention_Gap", 
            "Region_Avg_Attendance", "Retention_Attendance_Impact"]

X = df_drop[features]              # Independent variables
y = df_drop["Attendance"]          # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# model = LinearRegression()####Linear Regression 
model = RandomForestRegressor(random_state=42) ####Random Forest
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

# #WITHOUT DROPPING FEATURES
# features = ["Enrolments", "Total Retention", "Retention_Gap", 
#             "Region_Avg_Attendance", "Retention_Attendance_Impact","Teacher Retention","Non-Teacher Retention"]

# X = df[features]              # Independent variables
# y = df["Attendance"]          # Target variable

# #80-20 for Train-Test
# X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                                     test_size=0.2, 
#                                                     random_state=42)

# # model = LinearRegression() ###Linear Regression
# model = RandomForestRegressor(random_state=42) ####Random Forest
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# print("MAE:", mae)
# print("RMSE:", rmse)
# print("R² Score:", r2)

#Telling which feature is important   
for feature, score in zip(features, model.feature_importances_):
    print(f"{feature}: {score:.4f}")



# 🎯 Use classification label (0 or 1)
X = df_drop[["Enrolments", "Total Retention", "Retention_Gap", 
             "Region_Avg_Attendance", "Retention_Attendance_Impact"]]
y = df_drop["Above_Region_Attendance"]  # 0 = below avg, 1 = above avg

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)


# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Manual plot using seaborn heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Below", "Predicted Above"],
            yticklabels=["Actual Below", "Actual Above"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)



###Tuning to see the performance but didnt went well with same F1 score.
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import f1_score

# # Define parameter grid
# param_grid = {
#     "n_estimators": [100, 200],
#     "max_depth": [None, 10, 20],
#     "max_features": ["sqrt", "log2"],
#     "min_samples_split": [2, 5]
# }

# # Initialize classifier
# rf = RandomForestClassifier(random_state=42)

# # Set up grid search
# grid_search = GridSearchCV(estimator=rf,
#                            param_grid=param_grid,
#                            cv=5,
#                            scoring='f1',
#                            n_jobs=-1)

# # Fit on training data
# grid_search.fit(X_train, y_train)

# # Get the best model
# best_model = grid_search.best_estimator_

# # Predict and evaluate
# y_pred_best = best_model.predict(X_test)
# best_f1 = f1_score(y_test, y_pred_best)

# print("🔧 Best Parameters:", grid_search.best_params_)
# print("🏆 Best Test F1 Score:", best_f1)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Initialize Random Forest
rf_model = RandomForestRegressor(random_state=42)

# Cross-validate with 5 folds
r2_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')

# Print results
print("Cross-validated R² scores:", r2_scores)
print("Average R²:", r2_scores.mean())



############################################ PREDICTION FOR ATTENDANCE ###############

# Extract 2023 data to predict 2024 attendance
df_2023 = df_drop[df_drop["Year"] == 2023].copy()

# Select features for prediction
X_2023 = df_2023[features]

# Predict attendance for 2024
df_2023["Predicted_Attendance_2024"] = model.predict(X_2023)

# Add predicted year column
df_2023["Year"] = 2024

# Select relevant columns to save or analyze
df_2024_pred = df_2023[["School Name", "Region Name", "Year", "Predicted_Attendance_2024"]]

# Save to Excel
df_2024_pred.to_excel("Attendas_2024.xlsx", index=False)

print("✅ 2024 Attendance predictions saved.")
print(df_2024_pred.head())


################## Predict Teacher & Non‑Teacher Retention for 2024 #####################

# Re‑use the feature list that excludes the two retention columns
base_features = ["Enrolments",
                 "Total Retention",
                 "Retention_Gap",
                 "Region_Avg_Attendance",
                 "Retention_Attendance_Impact"]

# Teacher Retention model
df_teacher = df.dropna(subset=base_features + ["Teacher Retention", "Year"]).copy()

X_teacher = df_teacher[base_features]
y_teacher = df_teacher["Teacher Retention"]

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_teacher, y_teacher, test_size=0.2, random_state=42)

teacher_model = RandomForestRegressor(random_state=42)
teacher_model.fit(X_train_t, y_train_t)

print("Teacher Retention R² on hold‑out:",
      teacher_model.score(X_test_t, y_test_t))

# Predict 2024 from 2023 rows
df_2023_t = df_teacher[df_teacher["Year"] == 2023].copy()
df_2023_t["Predicted_Teacher_Retention_2024"] = teacher_model.predict(
    df_2023_t[base_features])

# Non‑Teacher Retention model
df_nt = df.dropna(subset=base_features + ["Non-Teacher Retention", "Year"]).copy()

X_nt = df_nt[base_features]
y_nt = df_nt["Non-Teacher Retention"]

X_train_nt, X_test_nt, y_train_nt, y_test_nt = train_test_split(
    X_nt, y_nt, test_size=0.2, random_state=42)

nt_model = RandomForestRegressor(random_state=42)
nt_model.fit(X_train_nt, y_train_nt)

print("Non‑Teacher Retention R² on hold‑out:",
      nt_model.score(X_test_nt, y_test_nt))

# Predict 2024 from 2023 rows
df_2023_nt = df_nt[df_nt["Year"] == 2023].copy()
df_2023_nt["Predicted_Non_Teacher_Retention_2024"] = nt_model.predict(
    df_2023_nt[base_features])

# Combine & export 
df_ret_pred = df_2023_t[["School Name", "Region Name"]].copy()
df_ret_pred["Year"] = 2024
df_ret_pred["Predicted_Teacher_Retention_2024"] = (
    df_2023_t["Predicted_Teacher_Retention_2024"].round(3))
df_ret_pred["Predicted_Non_Teacher_Retention_2024"] = (
    df_2023_nt["Predicted_Non_Teacher_Retention_2024"].round(3))

df_ret_pred.to_excel("Retention_2024.xlsx", index=False)
print("2024 Teacher & Non‑Teacher Retention predictions saved to 'Retention_2024.xlsx'")
print(df_ret_pred.head())