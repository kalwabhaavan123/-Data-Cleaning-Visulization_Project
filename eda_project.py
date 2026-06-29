# ============================================================
# PROJECT 3: EXPLORATORY DATA ANALYSIS (EDA)
# Domain  : Data Analysis / Data Science
# Goal    : Analyze dataset to discover patterns and trends
# Libraries: Pandas, NumPy, Matplotlib, Seaborn
# ============================================================


# -------------------- IMPORTS --------------------

import os
import warnings

import numpy as np
import pandas as pd

# Fix Python 3.14 Tkinter matplotlib error
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


warnings.filterwarnings("ignore")


# -------------------- SETUP --------------------

sns.set_theme(style="whitegrid")

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"


os.makedirs("output", exist_ok=True)


print("=" * 60)
print("   PROJECT 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)



# ============================================================
# STEP 1: CREATE DATASET
# ============================================================


np.random.seed(42)

n = 1000


print("\n[STEP 1] Creating Student Academic Performance Dataset...")


departments = [
    "Computer Science",
    "Electronics",
    "Mechanical",
    "Civil",
    "Business"
]


cities = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Chennai",
    "Hyderabad",
    "Pune"
]


genders = [
    "Male",
    "Female"
]



df = pd.DataFrame({

    "Student_ID":
        range(1001, 1001+n),


    "Gender":
        np.random.choice(genders, n),


    "Age":
        np.random.randint(17, 25, n),


    "Department":
        np.random.choice(departments, n),


    "City":
        np.random.choice(cities, n),


    "Study_Hours":
        np.random.normal(5, 2, n)
        .round(1)
        .clip(0, 12),


    "Attendance_Pct":
        np.random.normal(80, 12, n)
        .round(1)
        .clip(40, 100),


    "Sleep_Hours":
        np.random.normal(7, 1.5, n)
        .round(1)
        .clip(3, 11),


    "Previous_Score":
        np.random.normal(65, 15, n)
        .round(1)
        .clip(0, 100),


    "Internet_Access":
        np.random.choice(
            ["Yes", "No"],
            n,
            p=[0.78, 0.22]
        ),


    "Part_Time_Job":
        np.random.choice(
            ["Yes", "No"],
            n,
            p=[0.30, 0.70]
        ),


    "Library_Usage":
        np.random.choice(
            [
                "Daily",
                "Weekly",
                "Monthly",
                "Rarely"
            ],
            n,
            p=[0.20,0.35,0.30,0.15]
        ),


    "Stress_Level":
        np.random.choice(
            [
                "Low",
                "Medium",
                "High"
            ],
            n,
            p=[0.30,0.45,0.25]
        ),


    "Extracurricular":
        np.random.choice(
            ["Yes","No"],
            n,
            p=[0.55,0.45]
        )

})



# ============================================================
# CREATE FINAL SCORE
# ============================================================


df["Final_Score"] = (

    0.35 * df["Previous_Score"]

    +
    3.5 * df["Study_Hours"]

    +
    0.30 * df["Attendance_Pct"]

    +
    2 *
    (df["Internet_Access"]=="Yes")
    .astype(int)

    -
    3 *
    (df["Part_Time_Job"]=="Yes")
    .astype(int)

    -
    2 *
    (df["Stress_Level"]=="High")
    .astype(int)

    +
    np.random.normal(0,4,n)

).round(1).clip(0,100)



# ============================================================
# GRADE CREATION
# ============================================================


def assign_grade(score):

    if score >= 90:
        return "O (Outstanding)"

    elif score >= 80:
        return "A+ (Excellent)"

    elif score >= 70:
        return "A (Very Good)"

    elif score >= 60:
        return "B+ (Good)"

    elif score >= 50:
        return "B (Above Average)"

    else:
        return "F (Fail)"



df["Grade"] = df["Final_Score"].apply(assign_grade)



df["Pass_Fail"] = df["Final_Score"].apply(
    lambda x:
    "Pass" if x >= 50 else "Fail"
)



# Save dataset

df.to_csv(
    "output/student_data.csv",
    index=False
)



print(
    f"\nDataset Created Successfully"
)

print(
    f"Rows    : {df.shape[0]}"
)

print(
    f"Columns : {df.shape[1]}"
)


print(
    f"\nPass Rate : {(df['Pass_Fail']=='Pass').mean()*100:.1f}%"
)


print(
    f"Fail Rate : {(df['Pass_Fail']=='Fail').mean()*100:.1f}%"
)



print("\nFirst 5 Rows:")

print(df.head())



# ============================================================
# STEP 2: DATA INSPECTION
# ============================================================


print("\n" + "="*60)

print(
    "[STEP 2] DATA INSPECTION"
)

print("="*60)



print(
    "\nDataset Shape:"
)

print(df.shape)



print(
    "\nMissing Values:"
)

print(
    df.isnull().sum()
)



print(
    "\nDuplicate Rows:"
)

print(
    df.duplicated().sum()
)



print(
    "\nData Types:"
)

print(
    df.dtypes
)



print(
    "\nNumerical Summary:"
)

print(
    df.describe().round(2)
)



# ============================================================
# SAVE PROFILE REPORT
# ============================================================


with open(
    "output/data_profile.txt",
    "w"
) as file:

    file.write(
        "DATA PROFILE REPORT\n"
    )

    file.write(
        "="*50
    )

    file.write(
        "\n\nShape:\n"
    )

    file.write(
        str(df.shape)
    )


    file.write(
        "\n\nMissing Values:\n"
    )

    file.write(
        str(df.isnull().sum())
    )


    file.write(
        "\n\nStatistics:\n"
    )

    file.write(
        str(df.describe())
    )



print(
    "\nProfile report saved successfully"
)
# ============================================================
# STEP 3: UNIVARIATE ANALYSIS
# ============================================================

print("\n" + "="*60)
print("[STEP 3] UNIVARIATE ANALYSIS")
print("="*60)



# ------------------------------------------------------------
# 3.1 Numerical Feature Distributions
# ------------------------------------------------------------

numerical_columns = [
    "Age",
    "Study_Hours",
    "Attendance_Pct",
    "Sleep_Hours",
    "Previous_Score",
    "Final_Score"
]


fig, axes = plt.subplots(
    2,
    3,
    figsize=(18,10)
)


axes = axes.flatten()


for i, column in enumerate(numerical_columns):

    sns.histplot(
        data=df,
        x=column,
        kde=True,
        ax=axes[i]
    )


    axes[i].set_title(
        f"{column} Distribution"
    )


plt.tight_layout()


plt.savefig(
    "output/01_univariate_numerical.png",
    dpi=120
)


plt.close()



print(
    "Saved: Numerical distribution plots"
)




# ------------------------------------------------------------
# 3.2 Categorical Feature Analysis
# ------------------------------------------------------------


categorical_columns = [
    "Gender",
    "Department",
    "City",
    "Grade",
    "Pass_Fail"
]


fig, axes = plt.subplots(
    2,
    3,
    figsize=(18,10)
)


axes = axes.flatten()



for i, column in enumerate(categorical_columns):

    counts = (
        df[column]
        .value_counts()
    )


    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        legend=False,
        ax=axes[i]
    )


    axes[i].set_title(
        f"{column} Distribution"
    )


    axes[i].tick_params(
        axis="x",
        rotation=45
    )



plt.tight_layout()


plt.savefig(
    "output/02_univariate_categorical.png",
    dpi=120
)


plt.close()



print(
    "Saved: Categorical distribution plots"
)




# ============================================================
# STEP 4: BIVARIATE ANALYSIS
# ============================================================


print("\n" + "="*60)
print("[STEP 4] BIVARIATE ANALYSIS")
print("="*60)



# ------------------------------------------------------------
# 4.1 Study Hours vs Final Score
# ------------------------------------------------------------


plt.figure(
    figsize=(10,6)
)


sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Final_Score"
)


plt.title(
    "Study Hours vs Final Score"
)


plt.xlabel(
    "Study Hours"
)


plt.ylabel(
    "Final Score"
)


plt.tight_layout()


plt.savefig(
    "output/03_study_hours_vs_score.png",
    dpi=120
)


plt.close()



print(
    "Saved: Study hours relationship plot"
)




# ------------------------------------------------------------
# 4.2 Attendance vs Final Score
# ------------------------------------------------------------


plt.figure(
    figsize=(10,6)
)


sns.regplot(
    data=df,
    x="Attendance_Pct",
    y="Final_Score",
    scatter_kws={
        "alpha":0.5
    }
)


plt.title(
    "Attendance Percentage vs Final Score"
)


plt.xlabel(
    "Attendance Percentage"
)


plt.ylabel(
    "Final Score"
)


plt.tight_layout()


plt.savefig(
    "output/04_attendance_vs_score.png",
    dpi=120
)


plt.close()



print(
    "Saved: Attendance relationship plot"
)




# ------------------------------------------------------------
# 4.3 Department Performance Analysis
# ------------------------------------------------------------


department_score = (

    df.groupby("Department")
    ["Final_Score"]
    .mean()
    .sort_values()

)



plt.figure(
    figsize=(10,6)
)


sns.barplot(
    x=department_score.values,
    y=department_score.index,
    hue=department_score.index,
    legend=False
)


plt.title(
    "Average Score by Department"
)


plt.xlabel(
    "Average Final Score"
)


plt.ylabel(
    "Department"
)


plt.tight_layout()


plt.savefig(
    "output/05_department_performance.png",
    dpi=120
)


plt.close()



print(
    "Saved: Department analysis plot"
)




# ------------------------------------------------------------
# 4.4 Gender Performance Comparison
# ------------------------------------------------------------


gender_score = (

    df.groupby("Gender")
    ["Final_Score"]
    .mean()

)



plt.figure(
    figsize=(8,5)
)


sns.barplot(
    x=gender_score.index,
    y=gender_score.values,
    hue=gender_score.index,
    legend=False
)


plt.title(
    "Average Score by Gender"
)


plt.xlabel(
    "Gender"
)


plt.ylabel(
    "Average Final Score"
)


plt.tight_layout()


plt.savefig(
    "output/06_gender_performance.png",
    dpi=120
)


plt.close()



print(
    "Saved: Gender comparison plot"
)




# ------------------------------------------------------------
# 4.5 Library Usage vs Score
# ------------------------------------------------------------


library_score = (

    df.groupby("Library_Usage")
    ["Final_Score"]
    .mean()
    .sort_values()

)



plt.figure(
    figsize=(10,6)
)


sns.barplot(
    x=library_score.values,
    y=library_score.index,
    hue=library_score.index,
    legend=False
)


plt.title(
    "Library Usage Impact on Score"
)


plt.xlabel(
    "Average Final Score"
)


plt.ylabel(
    "Library Usage"
)


plt.tight_layout()


plt.savefig(
    "output/07_library_usage_analysis.png",
    dpi=120
)


plt.close()



print(
    "Saved: Library usage analysis plot"
)



print("\nUnivariate and Bivariate Analysis Completed Successfully")
# ============================================================
# STEP 5: MULTIVARIATE ANALYSIS
# ============================================================

print("\n" + "="*60)
print("[STEP 5] MULTIVARIATE ANALYSIS")
print("="*60)



# ------------------------------------------------------------
# 5.1 Correlation Heatmap
# ------------------------------------------------------------


print("\nGenerating Correlation Heatmap...")


numeric_columns = [

    "Age",
    "Study_Hours",
    "Attendance_Pct",
    "Sleep_Hours",
    "Previous_Score",
    "Final_Score"

]


correlation_matrix = (
    df[numeric_columns]
    .corr()
)



plt.figure(
    figsize=(10,7)
)


sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f"
)


plt.title(
    "Correlation Heatmap of Numerical Features"
)


plt.tight_layout()


plt.savefig(
    "output/08_correlation_heatmap.png",
    dpi=120
)


plt.close()



print(
    "Saved: Correlation heatmap"
)




# ------------------------------------------------------------
# 5.2 Pairplot Analysis
# ------------------------------------------------------------


print(
    "\nGenerating Pairplot..."
)


pair_columns = [

    "Study_Hours",
    "Attendance_Pct",
    "Previous_Score",
    "Final_Score"

]


pair_plot = sns.pairplot(
    df[pair_columns],
    diag_kind="kde"
)


pair_plot.fig.suptitle(
    "Pairwise Relationship Analysis",
    y=1.02
)


pair_plot.savefig(
    "output/09_pairplot.png",
    dpi=120
)


plt.close()



print(
    "Saved: Pairplot"
)




# ------------------------------------------------------------
# 5.3 Department and Gender Combined Analysis
# ------------------------------------------------------------


print(
    "\nGenerating Department-Gender Analysis..."
)



dept_gender = (

    df.groupby(
        [
            "Department",
            "Gender"
        ]
    )
    ["Final_Score"]
    .mean()
    .reset_index()

)



plt.figure(
    figsize=(12,6)
)


sns.barplot(
    data=dept_gender,
    x="Department",
    y="Final_Score",
    hue="Gender"
)


plt.title(
    "Department Wise Performance by Gender"
)


plt.xlabel(
    "Department"
)


plt.ylabel(
    "Average Final Score"
)


plt.xticks(
    rotation=45
)


plt.tight_layout()


plt.savefig(
    "output/10_department_gender_analysis.png",
    dpi=120
)


plt.close()



print(
    "Saved: Department gender analysis"
)




# ------------------------------------------------------------
# 5.4 Stress Level Impact Analysis
# ------------------------------------------------------------


print(
    "\nAnalyzing Stress Impact..."
)



stress_score = (

    df.groupby(
        "Stress_Level"
    )
    ["Final_Score"]
    .mean()
    .sort_values()

)



plt.figure(
    figsize=(8,5)
)


sns.barplot(
    x=stress_score.index,
    y=stress_score.values,
    hue=stress_score.index,
    legend=False
)


plt.title(
    "Impact of Stress Level on Academic Performance"
)


plt.xlabel(
    "Stress Level"
)


plt.ylabel(
    "Average Final Score"
)


plt.tight_layout()


plt.savefig(
    "output/11_stress_analysis.png",
    dpi=120
)


plt.close()



print(
    "Saved: Stress analysis"
)




# ============================================================
# STEP 6: OUTLIER DETECTION
# ============================================================


print("\n" + "="*60)
print("[STEP 6] OUTLIER DETECTION")
print("="*60)



outlier_columns = [

    "Study_Hours",
    "Attendance_Pct",
    "Previous_Score",
    "Final_Score"

]



fig, axes = plt.subplots(
    2,
    2,
    figsize=(14,10)
)



axes = axes.flatten()



for i, column in enumerate(outlier_columns):


    sns.boxplot(
        y=df[column],
        ax=axes[i]
    )


    axes[i].set_title(
        f"Outlier Detection: {column}"
    )



plt.tight_layout()


plt.savefig(
    "output/12_outlier_detection.png",
    dpi=120
)


plt.close()



print(
    "Saved: Outlier detection plots"
)




# ------------------------------------------------------------
# 6.1 Calculate Outlier Count using IQR
# ------------------------------------------------------------


outlier_report = {}



for column in outlier_columns:


    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1


    lower_limit = Q1 - 1.5 * IQR

    upper_limit = Q3 + 1.5 * IQR



    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]



    outlier_report[column] = len(outliers)



print(
    "\nOutlier Count:"
)


for key,value in outlier_report.items():

    print(
        f"{key}: {value}"
    )



# Save outlier report

with open(
    "output/outlier_report.txt",
    "w"
) as file:


    file.write(
        "OUTLIER ANALYSIS REPORT\n"
    )


    file.write(
        "="*40
    )


    for key,value in outlier_report.items():

        file.write(
            f"\n{key}: {value}"
        )



print(
    "\nOutlier report saved successfully"
)




# ============================================================
# STEP 7: KEY STATISTICAL INSIGHTS
# ============================================================


print("\nGenerating Statistical Insights...")



highest_department = (

    df.groupby("Department")
    ["Final_Score"]
    .mean()
    .idxmax()

)



best_city = (

    df.groupby("City")
    ["Final_Score"]
    .mean()
    .idxmax()

)



best_library = (

    df.groupby("Library_Usage")
    ["Final_Score"]
    .mean()
    .idxmax()

)



highest_study_group = (

    df.groupby("Study_Hours")
    ["Final_Score"]
    .mean()
    .idxmax()

)



print(
    "\nKey Insights:"
)


print(
    "Best Performing Department:",
    highest_department
)


print(
    "Best Performing City:",
    best_city
)


print(
    "Best Library Usage Group:",
    best_library
)



print(
    "\nMultivariate Analysis Completed Successfully"
)
# ============================================================
# STEP 8: FINAL EDA DASHBOARD
# ============================================================


print("\n" + "="*60)
print("[STEP 8] CREATING FINAL DASHBOARD")
print("="*60)



fig = plt.figure(
    figsize=(18,14)
)


fig.suptitle(
    "EXPLORATORY DATA ANALYSIS DASHBOARD",
    fontsize=20,
    fontweight="bold"
)



grid = gridspec.GridSpec(
    3,
    2,
    figure=fig
)



# ------------------------------------------------------------
# Dashboard 1: Score Distribution
# ------------------------------------------------------------


ax1 = fig.add_subplot(grid[0,0])


sns.histplot(
    data=df,
    x="Final_Score",
    kde=True,
    ax=ax1
)


ax1.set_title(
    "Final Score Distribution"
)



# ------------------------------------------------------------
# Dashboard 2: Department Performance
# ------------------------------------------------------------


ax2 = fig.add_subplot(grid[0,1])


dept_avg = (

    df.groupby("Department")
    ["Final_Score"]
    .mean()
    .sort_values()

)



sns.barplot(
    x=dept_avg.values,
    y=dept_avg.index,
    hue=dept_avg.index,
    legend=False,
    ax=ax2
)



ax2.set_title(
    "Average Score by Department"
)



# ------------------------------------------------------------
# Dashboard 3: Pass Fail Ratio
# ------------------------------------------------------------


ax3 = fig.add_subplot(grid[1,0])


pass_count = (

    df["Pass_Fail"]
    .value_counts()

)



ax3.pie(
    pass_count.values,
    labels=pass_count.index,
    autopct="%1.1f%%"
)


ax3.set_title(
    "Pass / Fail Distribution"
)




# ------------------------------------------------------------
# Dashboard 4: Attendance vs Score
# ------------------------------------------------------------


ax4 = fig.add_subplot(grid[1,1])


sns.scatterplot(
    data=df,
    x="Attendance_Pct",
    y="Final_Score",
    hue="Stress_Level",
    ax=ax4
)



ax4.set_title(
    "Attendance vs Final Score"
)



# ------------------------------------------------------------
# Dashboard 5: Study Hours Impact
# ------------------------------------------------------------


ax5 = fig.add_subplot(grid[2,0])


study_avg = (

    df.groupby("Study_Hours")
    ["Final_Score"]
    .mean()

)



sns.lineplot(
    x=study_avg.index,
    y=study_avg.values,
    marker="o",
    ax=ax5
)



ax5.set_title(
    "Study Hours Impact on Score"
)


ax5.set_xlabel(
    "Study Hours"
)


ax5.set_ylabel(
    "Average Score"
)



# ------------------------------------------------------------
# Dashboard 6: Grade Distribution
# ------------------------------------------------------------


ax6 = fig.add_subplot(grid[2,1])


grade_count = (

    df["Grade"]
    .value_counts()

)



sns.barplot(
    x=grade_count.values,
    y=grade_count.index,
    hue=grade_count.index,
    legend=False,
    ax=ax6
)



ax6.set_title(
    "Grade Distribution"
)



plt.tight_layout(
    rect=[0,0,1,0.96]
)



plt.savefig(
    "output/13_EDA_FINAL_DASHBOARD.png",
    dpi=120
)



plt.close()



print(
    "Dashboard saved successfully"
)




# ============================================================
# STEP 9: GENERATE FINAL EDA REPORT
# ============================================================


print("\nGenerating Final Report...")



total_students = df.shape[0]


average_score = (

    df["Final_Score"]
    .mean()

)



highest_department = (

    df.groupby("Department")
    ["Final_Score"]
    .mean()
    .idxmax()

)



highest_city = (

    df.groupby("City")
    ["Final_Score"]
    .mean()
    .idxmax()

)



highest_study_hours = (

    df["Study_Hours"]
    .mean()

)



pass_percentage = (

    df["Pass_Fail"]
    .value_counts(normalize=True)
    ["Pass"]
    *100

)



report = f"""

==================================================
EXPLORATORY DATA ANALYSIS REPORT
==================================================


1. DATASET OVERVIEW
-------------------

Total Students Analyzed:
{total_students}


Number of Features:
{df.shape[1]}



2. ACADEMIC PERFORMANCE ANALYSIS
--------------------------------


Average Final Score:
{average_score:.2f}


Pass Percentage:
{pass_percentage:.2f}%



3. KEY PERFORMANCE INSIGHTS
---------------------------


Best Performing Department:
{highest_department}


Best Performing City:
{highest_city}


Average Study Hours:
{highest_study_hours:.2f} hours



4. DATA ANALYSIS TECHNIQUES USED
--------------------------------


- Statistical Summary Analysis

- Univariate Analysis

- Bivariate Analysis

- Multivariate Analysis

- Correlation Analysis

- Outlier Detection

- Data Visualization



5. CONCLUSION
-------------


The dataset was successfully analyzed using
Exploratory Data Analysis techniques.

Important patterns, relationships, and trends
were identified using statistical methods and
visualizations.

The generated insights help understand the
factors influencing student academic performance.



==================================================
END OF REPORT
==================================================

"""



with open(
    "output/final_EDA_report.txt",
    "w"
) as file:

    file.write(report)



print(report)



# ============================================================
# PROJECT COMPLETION
# ============================================================


print("="*60)

print(
    "EDA PROJECT COMPLETED SUCCESSFULLY"
)

print(
    "All datasets, charts and reports saved in output folder"
)

print("="*60)