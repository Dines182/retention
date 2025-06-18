import pandas as pd

#As excel file have multiple sheets
xlsx = pd.ExcelFile("Task datasets.xlsx")
print(xlsx.sheet_names)

#Accessing Dictionary sheet and fetching first 5's data
df_dict = pd.read_excel("Task datasets.xlsx", sheet_name= 'Data Dictionary')
print("Data Dictionary",df_dict.head())
print(df_dict.shape)

#Accessing Student Enrolments sheet and fetching first 5's data
df_enrol = pd.read_excel("Task datasets.xlsx", sheet_name= 'Student Enrolments')
print("\nStudent Enrolments",df_enrol.head())
print(df_enrol.shape)

#Accessing Student Enrolments sheet and fetching first 5's data
df_attend = pd.read_excel("Task datasets.xlsx", sheet_name= 'Student Attend Rates')
print("\nStudent Attend Rates",df_attend.head())
print(df_attend.shape)

#Accessing Student Enrolments sheet and fetching first 5's data
df_teacher = pd.read_excel("Task datasets.xlsx", sheet_name= 'Teacher Retention')
print("\nTEACHER RETENTION" ,df_teacher.head())
print(df_teacher.shape)

#Accessing Student Enrolments sheet and fetching first 5's data
df_nontea = pd.read_excel("Task datasets.xlsx", sheet_name= 'Non-Teacher Retention')
print("\nNon-Teacher Retention",df_nontea.head())
print(df_nontea.shape)


##As shape tells each have 114 data with 8 columns now we are going to add all data on one excel file

# Define the year columns as we have from 2019 to 2023
year_cols = [2019, 2020, 2021, 2022, 2023]

# Melt all dataframes to long format as we have wide format
df_enrolled = df_enrol.melt(
    id_vars=["School Name", "Region Name"],
    value_vars=year_cols,
    var_name="Year",
    value_name="Enrolments"
)

df_attended = df_attend.melt(
    id_vars=["School Name", "Region Name"], ## Columns to keep fixed 
    value_vars=year_cols, #columns to melt into rows
    var_name="Year", #name the new column that holds former column names
    value_name="Attendance" #name the column that holds the values
)

df_teach = df_teacher.melt(
    id_vars=["School Name", "Region Name"],
    value_vars=year_cols,
    var_name="Year",
    value_name="Teacher Retention"
)

df_nonteach = df_nontea.melt(
    id_vars=["School Name", "Region Name"],
    value_vars=year_cols,
    var_name="Year",
    value_name="Non-Teacher Retention"
)

print(df_nonteach.head())

df_first_merges = df_enrolled.merge(df_attended , on =["School Name", "Region Name","Year"] , how="inner")
print(df_first_merges.head())
df_second_merges = df_first_merges.merge(df_teach , on =["School Name", "Region Name","Year"] , how="inner")
print(df_second_merges.head())
df_third_merges = df_second_merges.merge(df_nonteach , on =["School Name", "Region Name","Year"] , how="inner")
print(df_third_merges.head())
df_merged = df_third_merges
print(df_merged.head())

print(df_merged.shape)  # rows, columns as 114 schools × 5 years = 570 rows

# Checking any missing values exist and also confirming whether merged was accurate or not
print(df_merged.isnull().sum()) #Null value
print(df_merged.groupby("Year")["School Name"].nunique()) # check yearwise unique school
print(sorted(df_merged["Year"].unique())) #checked year
print(df_merged[df_merged["School Name"] == "Hogwarts School of Witchcraft and Wizardry"]) # Checked data using school name
print(df_merged.columns.tolist()) # to list all columns we have after merge

## Save final merged dataset output to Excel without any index as we confirm the dataset is accurate and merged succesfully
df_merged.to_excel("Yay All.xlsx", index=False)