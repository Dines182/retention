import pandas as pd

df_ret = pd.read_excel("Retention_2024.xlsx")
df_enrol = pd.read_excel("Predicted_Enrolments_2024.xlsx")
df_att = pd.read_excel("Attendas_2024.xlsx")

df_all = pd.read_excel("Yay All.xlsx")

df_att.rename(columns={
    "Predicted_Attendance_2024": "Attendance",
}, inplace=True)

df_ret.rename(columns={
    "Predicted_Teacher_Retention_2024": "Teacher Retention",
    "Predicted_Non_Teacher_Retention_2024" : "Non-Teacher Retention"
}, inplace=True)


# Merge all 2024 predictions on common keys
df_2024 = df_enrol.merge(df_ret, on=["School Name", "Region Name", "Year"], how="inner") \
                  .merge(df_att, on=["School Name", "Region Name", "Year"], how="inner")

# Append 2024 predictions to historical dataset
df_combined = pd.concat([df_all, df_2024], ignore_index=True)

# Save the combined dataset
df_combined.to_excel("Fulles_Data_with_2024.xlsx", index=False)
