import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_merged = pd.read_excel("Yay All.xlsx")
print(df_merged.head())

# #To see summary statistics of merged datasets
# print(df_merged.describe())

# ##Plotting 
# plt.figure(figsize=(10,6))# Set figure size to 10x6 inches
# sns.lineplot(data=df_merged, x="Year", y="Enrolments", ci=None)# Don't show confidence intervals 
# plt.title("Student Enrolments Over Time")
# plt.ylabel("Number of Students")
# plt.show()

# plt.figure(figsize=(10,6))
# sns.lineplot(data=df_merged, x="Year", y="Attendance", ci=None)
# plt.title("Student Attendance Over Time")
# plt.ylabel("Number of Students")
# plt.show()

# plt.figure(figsize=(10,6))
# sns.lineplot(data=df_merged, x="Year", y="Teacher Retention", ci=None)
# plt.title("Teacher Retention Over Time")
# plt.ylabel("Number of Students")
# plt.show()

# df_avg_retention = df_merged.groupby("Year")[["Teacher Retention", "Non-Teacher Retention"]].mean().reset_index()

# plt.figure(figsize=(10,6))
# sns.lineplot(data=df_avg_retention, x="Year", y="Teacher Retention", label="Teacher Retention", marker="o")
# sns.lineplot(data=df_avg_retention, x="Year", y="Non-Teacher Retention", label="Non-Teacher Retention", marker="o")
# plt.title("Average Retention Rates Over Time")
# plt.ylabel("Retention Rate")
# plt.ylim(0, 1)
# plt.show()


# # Group by Year and calculate both mean and median
# df_retention_stats = df_merged.groupby("Year")[["Teacher Retention", "Non-Teacher Retention"]].agg(["mean", "median"])
# df_retention_stats.columns = ['Teacher_Mean', 'Teacher_Median', 'NonTeacher_Mean', 'NonTeacher_Median']
# df_retention_stats = df_retention_stats.reset_index()

# # Plot mean vs median for Teacher and Non-Teacher retention
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=df_retention_stats, x="Year", y="Teacher_Mean", label="Teacher Mean", marker="o")
# sns.lineplot(data=df_retention_stats, x="Year", y="Teacher_Median", label="Teacher Median", marker="s")
# sns.lineplot(data=df_retention_stats, x="Year", y="NonTeacher_Mean", label="Non-Teacher Mean", marker="o")
# sns.lineplot(data=df_retention_stats, x="Year", y="NonTeacher_Median", label="Non-Teacher Median", marker="s")
# plt.title("Teacher vs Non-Teacher Retention: Mean vs Median")
# plt.ylabel("Retention Rate")
# plt.ylim(0, 1)
# plt.legend()
# plt.show()






# # List of metrics to check for outliers
# tabs = ["Enrolments", "Attendance", "Teacher Retention", "Non-Teacher Retention"]

# # Create a subplot for each metric
# plt.figure(figsize=(14, 8))

# for i, col in enumerate(tabs):
#     plt.subplot(2, 2, i+1)
#     sns.boxplot(y=df_merged[col])
#     plt.title(f"{col} - Outlier Detection")

# plt.tight_layout()
# plt.show()


# #Correlation Heatmap
# df_corr = df_merged[["Enrolments", "Attendance", "Teacher Retention", "Non-Teacher Retention"]].corr()

# sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Between Metrics")
# plt.show()

# top_schools = df_merged.groupby("School Name")["Enrolments"].mean().sort_values(ascending=False).head(5).index

# plt.figure(figsize=(12,6))
# sns.lineplot(data=df_merged[df_merged["School Name"].isin(top_schools)], 
#              x="Year", y="Enrolments", hue="School Name", marker="o")
# plt.title("Top 5 Schools by Average Enrolment")
# plt.show()

top_schools_attendance = df_merged.groupby("School Name")["Attendance"].mean().sort_values(ascending=False).head(5).index

plt.figure(figsize=(12,6))
sns.lineplot(data=df_merged[df_merged["School Name"].isin(top_schools_attendance)], 
             x="Year", y="Attendance", hue="School Name", marker="o")
plt.title("Top 5 Schools by Average Attendance")
plt.show()


top_schools_enrolments = df_merged.groupby("School Name")["Enrolments"].mean().sort_values(ascending=False).head(5).index

plt.figure(figsize=(12,6))
sns.lineplot(data=df_merged[df_merged["School Name"].isin(top_schools_enrolments)], 
             x="Year", y="Enrolments", hue="School Name", marker="o")
plt.title("Top 5 Schools by Average Enrolment")
plt.show()


# Combine both top school lists
top_schools_combined = set(top_schools_attendance).union(set(top_schools_enrolments))

# Plot Teacher Retention over years for top schools
plt.figure(figsize=(12,6))
sns.lineplot(data=df_merged[df_merged["School Name"].isin(top_schools_combined)],
             x="Year", y="Teacher Retention", hue="School Name", marker="o")
plt.title("Teacher Retention in Top Schools")
plt.ylabel("Teacher Retention")
plt.xlabel("Year")
plt.legend(title="School Name", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot Non-Teacher Retention over years for top schools
plt.figure(figsize=(12,6))
sns.lineplot(data=df_merged[df_merged["School Name"].isin(top_schools_combined)],
             x="Year", y="Non-Teacher Retention", hue="School Name", marker="o")
plt.title("Non-Teacher Retention in Top Schools")
plt.ylabel("Non-Teacher Retention")
plt.xlabel("Year")
plt.legend(title="School Name", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# # Plot region-level attendance trends over time
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=df_merged, x="Year", y="Attendance", hue="Region Name", marker="o")

# plt.title("Average Student Attendance by Region Over Time")
# plt.ylabel("Attendance Rate")
# plt.ylim(0, 1)
# plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# # Plot region-level enrolments trends over time
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=df_merged, x="Year", y="Enrolments", hue="Region Name", marker="o")

# plt.title("Average Student Enrolments by Region Over Time")
# plt.ylabel("Attendance Rate")
# plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()


# Combine both bottom school lists

bottom_schools_attendance = df_merged.groupby("School Name")["Attendance"].mean().sort_values(ascending=False).tail(5).index

plt.figure(figsize=(12,6))
sns.lineplot(data=df_merged[df_merged["School Name"].isin(bottom_schools_attendance)], 
             x="Year", y="Attendance", hue="School Name", marker="o")
plt.title("Bottom 5 Schools by Average Attendance")
plt.show()


bottom_schools_enrolments = df_merged.groupby("School Name")["Enrolments"].mean().sort_values(ascending=False).tail(5).index

plt.figure(figsize=(12,6))
sns.lineplot(data=df_merged[df_merged["School Name"].isin(bottom_schools_enrolments)], 
             x="Year", y="Enrolments", hue="School Name", marker="o")
plt.title("Bottom 5 Schools by Average Enrolment")
plt.show()


# Combine both bottom school lists
bottom_schools_combined = set(bottom_schools_attendance).union(set(bottom_schools_enrolments))

# Plot Teacher Retention over years for bottom schools
plt.figure(figsize=(12,6))
sns.lineplot(data=df_merged[df_merged["School Name"].isin(bottom_schools_combined)],
             x="Year", y="Teacher Retention", hue="School Name", marker="o")
plt.title("Teacher Retention in Top Schools")
plt.ylabel("Teacher Retention")
plt.xlabel("Year")
plt.legend(title="School Name", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot Non-Teacher Retention over years for bottom schools
plt.figure(figsize=(12,6))
sns.lineplot(data=df_merged[df_merged["School Name"].isin(bottom_schools_combined)],
             x="Year", y="Non-Teacher Retention", hue="School Name", marker="o")
plt.title("Non-Teacher Retention in Top Schools")
plt.ylabel("Non-Teacher Retention")
plt.xlabel("Year")
plt.legend(title="School Name", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# # Scatterplot: Enrolments vs Attendance
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df_merged, x="Enrolments", y="Attendance", hue="Region Name")
# plt.title("Scatterplot: Enrolments vs Attendance by Region")
# plt.xlabel("Enrolments")
# plt.ylabel("Attendance Rate")
# plt.ylim(0, 1.1)
# plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# # 3. Boxplot: Attendance by Region
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=df_merged, x="Region Name", y="Attendance")
# plt.title("Attendance Distribution by Region (2019–2023)")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# Set up a 2x2 grid of plots
fig, axs = plt.subplots(2, 2, figsize=(5, 6))

# Plot 1: Enrolments
sns.histplot(df_merged["Enrolments"], kde=True, ax=axs[0, 0], color='skyblue')
axs[0, 0].set_title("Distribution of Enrolments")

# Plot 2: Attendance
sns.histplot(df_merged["Attendance"], kde=True, ax=axs[0, 1], color='lightgreen')
axs[0, 1].set_title("Distribution of Attendance")

# Plot 3: Teacher Retention
sns.histplot(df_merged["Teacher Retention"], kde=True, ax=axs[1, 0], color='salmon')
axs[1, 0].set_title("Distribution of Teacher Retention")

# Plot 4: Non-Teacher Retention
sns.histplot(df_merged["Non-Teacher Retention"], kde=True, ax=axs[1, 1], color='plum')
axs[1, 1].set_title("Distribution of Non-Teacher Retention")

# Layout adjustment
plt.tight_layout()
plt.show()