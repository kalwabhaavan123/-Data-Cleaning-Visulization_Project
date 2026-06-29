# ============================================================
# PROJECT 1: DATA CLEANING & VISUALIZATION
# Domain: Data Science / Data Analytics
# Goal: Clean a raw dataset and visualize insights
# Libraries: Pandas, NumPy, Matplotlib, Seaborn
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------- Setup ----------------
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

os.makedirs("output", exist_ok=True)

print("=" * 60)
print("   DATA CLEANING & VISUALIZATION PROJECT")
print("=" * 60)


# ============================================================
# 1. GENERATE RAW DATASET
# ============================================================

np.random.seed(42)

n = 500

products = [
    "Laptop",
    "Smartphone",
    "Tablet",
    "Headphones",
    "Monitor",
    "Keyboard",
    "Mouse",
    "Speaker",
    "Webcam",
    "Smartwatch"
]

cities = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Chennai",
    "Hyderabad",
    "Pune",
    "Kolkata",
    "Jaipur"
]

payments = [
    "Credit Card",
    "Debit Card",
    "UPI",
    "Net Banking",
    "COD"
]


df = pd.DataFrame({

    "Order_ID": range(1001, 1001+n),

    "Product":
        np.random.choice(products, n),

    "City":
        np.random.choice(cities, n),

    "Quantity":
        np.random.randint(1, 6, n),

    "Unit_Price":
        np.random.normal(15000, 8000, n).round(2),

    "Rating":
        np.random.uniform(1, 5, n).round(1),

    "Payment_Method":
        np.random.choice(payments, n)

})


df["Total_Price"] = (
    df["Unit_Price"] *
    df["Quantity"]
).round(2)


# ---------------- Add data problems ----------------

# Missing values

df.loc[
    df.sample(frac=0.08, random_state=1).index,
    "Unit_Price"
] = np.nan


df.loc[
    df.sample(frac=0.06, random_state=2).index,
    "City"
] = np.nan


df.loc[
    df.sample(frac=0.07, random_state=3).index,
    "Rating"
] = np.nan


# Outliers

df.loc[
    df.sample(n=8, random_state=4).index,
    "Unit_Price"
] = 500000


# Duplicate rows

df = pd.concat(
    [
        df,
        df.sample(n=20, random_state=5)
    ],
    ignore_index=True
)


# Incorrect text

df.loc[
    df.sample(n=15, random_state=6).index,
    "City"
] = "  mumbai "


df.to_csv(
    "output/raw_data.csv",
    index=False
)


print(
    f"\n[1] Raw dataset created: {df.shape[0]} rows, {df.shape[1]} columns"
)

print(df.head())


# ============================================================
# 2. DATA INSPECTION
# ============================================================

print("\n" + "="*60)
print("   DATA INSPECTION")
print("="*60)


print("\nShape:")
print(df.shape)


print("\nMissing Values:")
print(df.isnull().sum())


print(
    "\nDuplicate Rows:",
    df.duplicated().sum()
)


print(
    "\nStatistical Summary:"
)

print(
    df.describe().round(2)
)



# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\n" + "="*60)
print("   DATA CLEANING")
print("="*60)



# Remove duplicates

before = len(df)

df = (
    df
    .drop_duplicates()
    .reset_index(drop=True)
)


print(
    f"\nRemoved {before-len(df)} duplicate rows"
)



# Fix city names

df["City"] = (
    df["City"]
    .str.strip()
    .str.title()
)


print(
    "City names standardized"
)



# Missing value treatment

df["Unit_Price"] = (
    df.groupby("Product")["Unit_Price"]
    .transform(
        lambda x:
        x.fillna(x.median())
    )
)


df["City"] = (
    df["City"]
    .fillna(
        df["City"].mode()[0]
    )
)


df["Rating"] = (
    df["Rating"]
    .fillna(
        round(df["Rating"].mean(),1)
    )
)


df["Total_Price"] = (
    df["Unit_Price"] *
    df["Quantity"]
).round(2)


print(
    "Missing values handled"
)



# ============================================================
# OUTLIER REMOVAL
# ============================================================


def remove_outliers_iqr(data, column):

    Q1 = data[column].quantile(0.25)

    Q3 = data[column].quantile(0.75)

    IQR = Q3-Q1

    lower = Q1 - 1.5*IQR

    upper = Q3 + 1.5*IQR


    before = len(data)


    data = data[
        (data[column]>=lower) &
        (data[column]<=upper)
    ]


    print(
        f"Removed {before-len(data)} outliers from {column}"
    )


    return data



df = remove_outliers_iqr(
    df,
    "Unit_Price"
)


df = df.reset_index(drop=True)


df.to_csv(
    "output/cleaned_data.csv",
    index=False
)


print(
    "\nFinal Clean Dataset:",
    df.shape
)

print(
    "Saved cleaned_data.csv"
)
# ============================================================
# 4. DATA VISUALIZATION
# ============================================================

print("\n" + "="*60)
print("   GENERATING VISUALIZATIONS")
print("="*60)



# ------------------------------------------------------------
# 4.1 Unit Price Distribution
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

sns.histplot(
    df["Unit_Price"],
    kde=True
)

plt.title(
    "Unit Price Distribution",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel(
    "Unit Price"
)

plt.tight_layout()


plt.savefig(
    "output/01_price_distribution.png",
    dpi=120
)


plt.close()



# ------------------------------------------------------------
# 4.2 Revenue by Product
# ------------------------------------------------------------

plt.figure(figsize=(10,6))


product_revenue = (
    df.groupby("Product")
    ["Total_Price"]
    .sum()
    .sort_values()
)


sns.barplot(
    x=product_revenue.values,
    y=product_revenue.index,
    hue=product_revenue.index,
    legend=False
)


plt.title(
    "Total Revenue by Product",
    fontsize=14,
    fontweight="bold"
)


plt.xlabel(
    "Revenue"
)


plt.ylabel(
    "Product"
)


plt.tight_layout()


plt.savefig(
    "output/02_revenue_by_product.png",
    dpi=120
)


plt.close()



# ------------------------------------------------------------
# 4.3 Orders By City
# ------------------------------------------------------------

plt.figure(figsize=(10,6))


city_orders = (
    df["City"]
    .value_counts()
)


sns.barplot(
    x=city_orders.index,
    y=city_orders.values,
    hue=city_orders.index,
    legend=False
)


plt.title(
    "Order Count by City",
    fontsize=14,
    fontweight="bold"
)


plt.xlabel(
    "City"
)


plt.ylabel(
    "Number of Orders"
)


plt.xticks(
    rotation=45
)


plt.tight_layout()


plt.savefig(
    "output/03_orders_by_city.png",
    dpi=120
)


plt.close()



# ------------------------------------------------------------
# 4.4 Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(8,6))


numeric_columns = [
    "Quantity",
    "Unit_Price",
    "Rating",
    "Total_Price"
]


sns.heatmap(
    df[numeric_columns].corr(),
    annot=True,
    fmt=".2f"
)


plt.title(
    "Correlation Heatmap",
    fontsize=14,
    fontweight="bold"
)


plt.tight_layout()


plt.savefig(
    "output/04_correlation_heatmap.png",
    dpi=120
)


plt.close()



print(
    "4 individual charts saved successfully"
)



# ============================================================
# 5. DASHBOARD
# ============================================================


print("\nCreating Dashboard...")


fig, axes = plt.subplots(
    2,
    2,
    figsize=(16,12)
)


fig.suptitle(
    "DATA INSIGHTS DASHBOARD",
    fontsize=18,
    fontweight="bold"
)



# Chart 1

sns.histplot(
    df["Unit_Price"],
    kde=True,
    ax=axes[0,0]
)


axes[0,0].set_title(
    "Unit Price Distribution"
)



# Chart 2

sns.barplot(
    x=city_orders.index,
    y=city_orders.values,
    hue=city_orders.index,
    legend=False,
    ax=axes[0,1]
)


axes[0,1].set_title(
    "Orders By City"
)


axes[0,1].tick_params(
    axis="x",
    rotation=45
)



# Chart 3

payment_count = (
    df["Payment_Method"]
    .value_counts()
)


axes[1,0].pie(
    payment_count.values,
    labels=payment_count.index,
    autopct="%1.1f%%"
)


axes[1,0].set_title(
    "Payment Method Distribution"
)



# Chart 4

sns.scatterplot(
    data=df,
    x="Unit_Price",
    y="Total_Price",
    hue="Quantity",
    ax=axes[1,1]
)


axes[1,1].set_title(
    "Unit Price vs Total Price"
)



plt.tight_layout()


plt.savefig(
    "output/05_DASHBOARD.png",
    dpi=120
)


plt.close()



print(
    "Dashboard saved successfully"
)



# ============================================================
# 6. KEY FINDINGS REPORT
# ============================================================


top_product = (
    df.groupby("Product")
    ["Total_Price"]
    .sum()
    .idxmax()
)


top_city = (
    df["City"]
    .value_counts()
    .idxmax()
)


average_rating = (
    df["Rating"]
    .mean()
)



report = f"""

DATA STORYTELLING REPORT
========================


1. Dataset cleaning completed successfully.

2. Final dataset contains:
   {df.shape[0]} clean records.


3. Highest revenue generating product:
   {top_product}


4. City with maximum orders:
   {top_city}


5. Average customer rating:
   {average_rating:.2f}/5


6. Data visualization helped identify
   sales patterns and business insights.


CONCLUSION:
The raw dataset was cleaned, processed,
and analyzed successfully using Python
data science libraries.

The generated visualizations provide
clear insights for decision making.

"""



with open(
    "output/key_findings.txt",
    "w"
) as file:

    file.write(report)



print(report)



print("="*60)

print(
    "PROJECT 1 COMPLETED SUCCESSFULLY"
)

print(
    "All files saved inside output folder"
)

print("="*60)



# Display final dashboard

plt.show()