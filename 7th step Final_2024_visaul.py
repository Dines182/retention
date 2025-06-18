
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")  # nice default look
  
df = pd.read_excel("Fulles_Data_with_2024.xlsx")


metrics = ["Enrolments", "Attendance",
           "Teacher Retention", "Non-Teacher Retention"]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for ax, metric in zip(axs.flatten(), metrics):
    sns.lineplot(data=df, x="Year", y=metric,
                 estimator="mean", ci=None, marker="o", ax=ax)
    ax.axvline(2023.5, ls="--", c="grey")   # marks start of predictions
    ax.set_title(f"Average {metric} (2019‑2024)")
plt.tight_layout()
plt.show()

#  Top / bottom 5 schools (attendance & enrolment)
def top_bottom(metric, top=True, n=5):
    order = df.groupby("School Name")[metric].mean().sort_values(ascending=not top)
    return order.head(n).index if top else order.tail(n).index

top5_att = top_bottom("Attendance", top=True)
bot5_att = top_bottom("Attendance", top=False)
top5_enr = top_bottom("Enrolments", top=True)
bot5_enr = top_bottom("Enrolments", top=False)

def plot_multi(schools, metric, title):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df[df["School Name"].isin(schools)],
                 x="Year", y=metric, hue="School Name", marker="o")
    plt.axvline(2023.5, ls="--", c="grey")
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_multi(top5_att, "Attendance", " Top5Schools by Avg Attendance")
plot_multi(bot5_att, "Attendance", " Bottom5Schools by Avg Attendance")
plot_multi(top5_enr, "Enrolments", " Top5Schools by Avg Enrolment")
plot_multi(bot5_enr, "Enrolments", " Bottom5Schools by Avg Enrolment")


#  Teacher vs Non‑Teacher retention lines (same schools)

combined_top = set(top5_att).union(top5_enr)
for metric in ["Teacher Retention", "Non-Teacher Retention"]:
    plot_multi(combined_top, metric, f"{metric} in Top Schools")

combined_bot = set(bot5_att).union(bot5_enr)
for metric in ["Teacher Retention", "Non-Teacher Retention"]:
    plot_multi(combined_bot, metric, f"{metric} in Bottom Schools")


#  Retention gap over time (Teacher minus Non‑Teacher)

df["Retention_Gap"] = df["Teacher Retention"] - df["Non-Teacher Retention"]
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="Year", y="Retention_Gap", estimator="mean",
             ci=None, marker="o", color="indianred")
plt.axvline(2023.5, ls="--", c="grey")
plt.title("Average Retention Gap (Teacher − Non‑Teacher)")
plt.tight_layout()
plt.show()

# Distributions (histograms with KDE)
fig, axs = plt.subplots(2, 2, figsize=(11, 7))
for ax, metric in zip(axs.flatten(), metrics):
    sns.histplot(df[metric], kde=True, ax=ax)
    ax.set_title(f"Distribution: {metric}")
plt.tight_layout()
plt.show()



#  Region‑level attendance vs region average 
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Year", y="Attendance",
             hue="Region Name", estimator="mean",
             ci=None, marker="o")
plt.axvline(2023.5, ls="--", c="grey")
plt.title("Average Attendance by Region (2019‑2024)")
plt.tight_layout()
plt.show()
