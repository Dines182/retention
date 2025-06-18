import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

# Step 1: Loading ds
df = pd.read_excel("Yay All.xlsx")


# Step 2: Feature Engineering adding few more features
df['Total Retention'] = (df['Teacher Retention'] + df['Non-Teacher Retention'])/2 #Taking mean of features teacher and non-teacher
print(df['Total Retention'].head())
df["Retention_Gap"] = abs(df["Teacher Retention"] - df["Non-Teacher Retention"]) #Abs gap of teacher and nonteacher ratio
df["Region_Avg_Attendance"] = df.groupby("Region Name")["Attendance"].transform("mean") #grouping school by region and taking mean
df["Above_Region_Attendance"] = (df["Attendance"] > df["Region_Avg_Attendance"]).astype(int) #1 means above 0 means below region
df["Retention_Attendance_Impact"] = df["Total Retention"] * df["Attendance"] #Combined feature with attendance and retention

# print(df.head())

df.to_excel("NEWW All.xlsx", index=False) #index set to false

df = pd.read_excel("NEWW All.xlsx")

print(df.isnull().sum()) #Null value


# Basic statistics of the engineered features
feature_stats = df[["Total Retention", "Retention_Gap", "Region_Avg_Attendance",
                    "Above_Region_Attendance", "Retention_Attendance_Impact"]].describe()

print(feature_stats)

# Correlation of engineered features with target-like variables
correlations = df[["Attendance", "Enrolments", "Total Retention", "Retention_Gap", 
                   "Above_Region_Attendance", "Retention_Attendance_Impact"]].corr()

sns.heatmap(correlations, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Metrics")
plt.show()