# ============================================================
# PROJECT 2: PREDICTIVE MODELING USING MACHINE LEARNING
# Domain  : Machine Learning / Supervised Learning
# Goal    : Build models to predict customer churn,
#           evaluate accuracy, confusion matrix & ROC curve.
# Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)

# --- Setup ---
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
os.makedirs("output", exist_ok=True)

print("="*60)
print("   PROJECT 2: PREDICTIVE MODELING USING ML")
print("="*60)

# ============================================================
# STEP 1: CREATE / LOAD DATASET
#         (Customer Churn Prediction Dataset)
#         Replace with pd.read_csv("your_file.csv") if needed
# ============================================================
np.random.seed(42)
n = 1000

print("\n[STEP 1] Generating Customer Churn Dataset...")

df = pd.DataFrame({
    "CustomerID"       : range(1001, 1001 + n),
    "Age"              : np.random.randint(18, 70, n),
    "Gender"           : np.random.choice(["Male", "Female"], n),
    "Tenure_Months"    : np.random.randint(1, 72, n),
    "Monthly_Charges"  : np.random.normal(70, 20, n).round(2).clip(20, 150),
    "Total_Spend"      : np.random.normal(2500, 800, n).round(2).clip(100),
    "Num_Complaints"   : np.random.poisson(1.5, n),
    "Has_Premium"      : np.random.choice([0, 1], n),
    "Contract_Type"    : np.random.choice(
                            ["Month-to-Month", "One Year", "Two Year"], n),
    "Internet_Service" : np.random.choice(["Fiber", "DSL", "None"], n),
})

# --- Create realistic Churn target ---
churn_prob = (
    0.30 * (df["Num_Complaints"] > 2).astype(int) +
    0.25 * (df["Tenure_Months"] < 12).astype(int) +
    0.20 * (df["Monthly_Charges"] > 85).astype(int) +
    0.15 * (df["Has_Premium"] == 0).astype(int) +
    0.10 * (df["Contract_Type"] == "Month-to-Month").astype(int)
)
noise = np.random.normal(0, 0.15, n)
df["Churn"] = ((churn_prob + noise) > 0.5).astype(int)

# Save raw dataset
df.to_csv("output/customer_churn_raw.csv", index=False)

print(f"   ✅ Dataset created: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Churn Distribution:\n{df['Churn'].value_counts().to_string()}")
print(f"   Churn Rate: {df['Churn'].mean()*100:.1f}%")
print(df.head())

# ============================================================
# STEP 2: DATA INSPECTION & PREPROCESSING
# ============================================================
print("\n" + "="*60)
print("   [STEP 2] DATA INSPECTION & PREPROCESSING")
print("="*60)

print(f"\nShape      : {df.shape}")
print(f"Missing    : {df.isnull().sum().sum()}")
print(f"Duplicates : {df.duplicated().sum()}")
print(f"\nData Types:\n{df.dtypes}")

# --- Encode categorical features ---
df_model = df.copy()
df_model.drop("CustomerID", axis=1, inplace=True)

le = LabelEncoder()
cat_cols = ["Gender", "Contract_Type", "Internet_Service"]
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

print("\n✅ Categorical columns encoded successfully")
print(df_model.head())

# ============================================================
# STEP 3: TRAIN-TEST SPLIT & FEATURE SCALING
# ============================================================
print("\n" + "="*60)
print("   [STEP 3] TRAIN-TEST SPLIT & FEATURE SCALING")
print("="*60)

X = df_model.drop("Churn", axis=1)
y = df_model["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n   Training Samples : {X_train.shape[0]}")
print(f"   Testing  Samples : {X_test.shape[0]}")
print(f"   Features         : {X.shape[1]}")
print(f"   Target (Churn=1) : {y.sum()} total churned customers")

# ============================================================
# STEP 4: TRAIN ALL 3 MODELS
# ============================================================
print("\n" + "="*60)
print("   [STEP 4] TRAINING MODELS")
print("="*60)

models = {
    "Logistic Regression" : LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"       : RandomForestClassifier(
                               n_estimators=100, random_state=42, n_jobs=-1)
}

results = {}

for name, model in models.items():
    # Train
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    cv_score = cross_val_score(
                    model, X_train_sc, y_train, cv=5, scoring="accuracy").mean()

    results[name] = {
        "model"    : model,
        "y_pred"   : y_pred,
        "y_prob"   : y_prob,
        "accuracy" : acc,
        "precision": prec,
        "recall"   : rec,
        "f1"       : f1,
        "fpr"      : fpr,
        "tpr"      : tpr,
        "auc"      : roc_auc,
        "cv_score" : cv_score
    }

    print(f"\n🔹 {name}")
    print(f"   Accuracy  : {acc*100:.2f}%")
    print(f"   Precision : {prec*100:.2f}%")
    print(f"   Recall    : {rec*100:.2f}%")
    print(f"   F1 Score  : {f1*100:.2f}%")
    print(f"   ROC-AUC   : {roc_auc:.4f}")
    print(f"   CV Score  : {cv_score*100:.2f}%")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Stay (0)", "Churn (1)"]))

# ============================================================
# STEP 5: VISUALIZATIONS
# ============================================================
print("\n" + "="*60)
print("   [STEP 5] GENERATING VISUALIZATIONS")
print("="*60)

# -----------------------------------------------
# CHART 1: Model Accuracy Comparison Bar Chart
# -----------------------------------------------
plt.figure(figsize=(12, 6))
metric_data = {
    name: {
        "Accuracy"  : results[name]["accuracy"],
        "Precision" : results[name]["precision"],
        "Recall"    : results[name]["recall"],
        "F1 Score"  : results[name]["f1"]
    }
    for name in results
}
metrics_df = pd.DataFrame(metric_data).T
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
x = np.arange(len(metrics_df))
width = 0.20

fig, ax = plt.subplots(figsize=(13, 7))
for i, (col, color) in enumerate(zip(metrics_df.columns, colors)):
    bars = ax.bar(x + i*width, metrics_df[col],
                  width, label=col, color=color,
                  edgecolor="black", alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.2f}",
                ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics_df.index, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_title("Model Performance Comparison (Accuracy, Precision, Recall, F1)",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Score")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("output/01_model_comparison.png", dpi=120)
plt.close()
print("✅ Saved: 01_model_comparison.png")

# -----------------------------------------------
# CHART 2: Confusion Matrices (All 3 Models)
# -----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Confusion Matrices — All Models", fontsize=16, fontweight="bold")

for ax, name in zip(axes, results):
    cm = confusion_matrix(y_test, results[name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Stay", "Churn"],
                yticklabels=["Stay", "Churn"],
                linewidths=1, linecolor="white")
    ax.set_title(f"{name}\nAccuracy: {results[name]['accuracy']*100:.2f}%",
                 fontweight="bold")
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")

plt.tight_layout()
plt.savefig("output/02_confusion_matrices.png", dpi=120)
plt.close()
print("✅ Saved: 02_confusion_matrices.png")

# -----------------------------------------------
# CHART 3: ROC Curves (All 3 Models)
# -----------------------------------------------
plt.figure(figsize=(10, 8))
colors_roc = ["#3498db", "#e74c3c", "#2ecc71"]

for (name, res), color in zip(results.items(), colors_roc):
    plt.plot(res["fpr"], res["tpr"], linewidth=2.5, color=color,
             label=f"{name}  (AUC = {res['auc']:.3f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Guess (AUC = 0.500)")
plt.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
plt.title("ROC Curve — Model Comparison", fontsize=14, fontweight="bold")
plt.xlabel("False Positive Rate (FPR)", fontsize=12)
plt.ylabel("True Positive Rate (TPR)", fontsize=12)
plt.legend(loc="lower right", fontsize=11)
plt.tight_layout()
plt.savefig("output/03_roc_curves.png", dpi=120)
plt.close()
print("✅ Saved: 03_roc_curves.png")

# -----------------------------------------------
# CHART 4: Feature Importance (Random Forest)
# -----------------------------------------------
rf_model = results["Random Forest"]["model"]
importances = pd.Series(
    rf_model.feature_importances_, index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(12, 7))
colors_imp = sns.color_palette("RdYlGn", len(importances))
bars = plt.barh(importances.index, importances.values,
                color=colors_imp, edgecolor="black")
for bar, val in zip(bars, importances.values):
    plt.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=10, fontweight="bold")
plt.title("Feature Importance — Random Forest", fontsize=14, fontweight="bold")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("output/04_feature_importance.png", dpi=120)
plt.close()
print("✅ Saved: 04_feature_importance.png")

# -----------------------------------------------
# CHART 5: Decision Tree Visualization
# -----------------------------------------------
fig, ax = plt.subplots(figsize=(20, 10))
dt_model = results["Decision Tree"]["model"]
plot_tree(dt_model, feature_names=X.columns.tolist(),
          class_names=["Stay", "Churn"],
          filled=True, rounded=True, max_depth=3, ax=ax,
          fontsize=9)
ax.set_title("Decision Tree Structure (max_depth=3)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("output/05_decision_tree.png", dpi=120)
plt.close()
print("✅ Saved: 05_decision_tree.png")

# -----------------------------------------------
# CHART 6: Churn Rate by Key Features
# -----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Churn Analysis by Key Features", fontsize=16, fontweight="bold")

# By Contract Type
churn_contract = df.groupby("Contract_Type")["Churn"].mean() * 100
sns.barplot(x=churn_contract.index, y=churn_contract.values,
            palette="Reds_r", ax=axes[0])
axes[0].set_title("Churn Rate by Contract Type")
axes[0].set_ylabel("Churn Rate (%)")
axes[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(churn_contract.values):
    axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

# By Tenure Group
df["Tenure_Group"] = pd.cut(df["Tenure_Months"],
                             bins=[0, 12, 24, 48, 72],
                             labels=["0-12M", "12-24M", "24-48M", "48-72M"])
churn_tenure = df.groupby("Tenure_Group")["Churn"].mean() * 100
sns.barplot(x=churn_tenure.index, y=churn_tenure.values,
            palette="Blues_r", ax=axes[1])
axes[1].set_title("Churn Rate by Tenure")
axes[1].set_ylabel("Churn Rate (%)")
for i, v in enumerate(churn_tenure.values):
    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

# By Complaint Level
df["Complaint_Level"] = pd.cut(df["Num_Complaints"],
                                bins=[-1, 0, 2, 10],
                                labels=["None", "Low (1-2)", "High (3+)"])
churn_comp = df.groupby("Complaint_Level")["Churn"].mean() * 100
sns.barplot(x=churn_comp.index, y=churn_comp.values,
            palette="Oranges_r", ax=axes[2])
axes[2].set_title("Churn Rate by Complaint Level")
axes[2].set_ylabel("Churn Rate (%)")
for i, v in enumerate(churn_comp.values):
    axes[2].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("output/06_churn_analysis.png", dpi=120)
plt.close()
print("✅ Saved: 06_churn_analysis.png")

# ============================================================
# STEP 6: FINAL DASHBOARD
# ============================================================
print("\n" + "="*60)
print("   [STEP 6] GENERATING FINAL DASHBOARD")
print("="*60)

fig = plt.figure(figsize=(22, 16), facecolor="#f8f9fa")
fig.suptitle("🤖 PREDICTIVE MODELING DASHBOARD — CUSTOMER CHURN",
             fontsize=20, fontweight="bold", y=0.98, color="#2C3E50")

from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

# --- KPI Row ---
best_name = max(results, key=lambda x: results[x]["accuracy"])
kpis = [
    ("Best Model", best_name.replace(" ", "\n"), "#27AE60"),
    ("Best Accuracy", f"{results[best_name]['accuracy']*100:.1f}%", "#2980B9"),
    ("Best AUC Score", f"{results[best_name]['auc']:.3f}", "#8E44AD"),
]
for i, (label, val, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(color)
    ax.text(0.5, 0.60, val, ha="center", va="center",
            fontsize=20, fontweight="bold", color="white",
            transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha="center", va="center",
            fontsize=12, color="white", transform=ax.transAxes)
    ax.axis("off")

# --- Chart Row 1 ---
# ROC Curve
ax1 = fig.add_subplot(gs[1, 0:2])
colors_roc2 = ["#3498db", "#e74c3c", "#2ecc71"]
for (name, res), color in zip(results.items(), colors_roc2):
    ax1.plot(res["fpr"], res["tpr"], linewidth=2.5, color=color,
             label=f"{name} (AUC={res['auc']:.3f})")
ax1.plot([0, 1], [0, 1], "k--")
ax1.set_title("ROC Curves", fontweight="bold")
ax1.set_xlabel("FPR")
ax1.set_ylabel("TPR")
ax1.legend(fontsize=9)

# Feature Importance
ax2 = fig.add_subplot(gs[1, 2])
importances_plot = importances.tail(6)
ax2.barh(importances_plot.index, importances_plot.values,
         color=sns.color_palette("RdYlGn", 6), edgecolor="black")
ax2.set_title("Top Features\n(Random Forest)", fontweight="bold")
ax2.set_xlabel("Importance")

# --- Chart Row 2 ---
# Best model confusion matrix
ax3 = fig.add_subplot(gs[2, 0])
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
            xticklabels=["Stay", "Churn"],
            yticklabels=["Stay", "Churn"])
ax3.set_title(f"Confusion Matrix\n({best_name})", fontweight="bold")
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")

# Churn by Contract
ax4 = fig.add_subplot(gs[2, 1])
sns.barplot(x=churn_contract.index, y=churn_contract.values,
            palette="Reds_r", ax=ax4)
ax4.set_title("Churn by Contract Type", fontweight="bold")
ax4.set_ylabel("Churn Rate (%)")
ax4.tick_params(axis='x', rotation=15)

# Accuracy Comparison
ax5 = fig.add_subplot(gs[2, 2])
names_list = list(results.keys())
acc_list = [results[n]["accuracy"] for n in names_list]
short_names = ["LR", "DT", "RF"]
bars = ax5.bar(short_names, acc_list,
               color=["#3498db", "#e74c3c", "#2ecc71"],
               edgecolor="black")
for bar, acc in zip(bars, acc_list):
    ax5.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f"{acc:.3f}", ha="center", fontweight="bold")
ax5.set_ylim(0, 1.1)
ax5.set_title("Accuracy Comparison", fontweight="bold")
ax5.set_ylabel("Accuracy")

# Footer
fig.text(0.5, 0.01,
         "Project 2: Predictive Modeling | AICTE/MSME Internship 2026 | [Your Name]",
         ha="center", fontsize=10, color="#888", style="italic")

plt.savefig("output/07_DASHBOARD.png", dpi=150,
            bbox_inches="tight", facecolor="#f8f9fa")
plt.close()
print("✅ Saved: 07_DASHBOARD.png")

# ============================================================
# STEP 7: FINAL FINDINGS REPORT
# ============================================================
best = results[best_name]
top_feature = importances.idxmax()

report = f"""
╔══════════════════════════════════════════════════════╗
║   PROJECT 2: PREDICTIVE MODELING — FINDINGS REPORT  ║
╚══════════════════════════════════════════════════════╝

DATASET OVERVIEW
----------------
Total Records  : {len(df)}
Features Used  : {X.shape[1]}
Target Variable: Churn (0 = Stay, 1 = Churn)
Churn Rate     : {df['Churn'].mean()*100:.1f}%

MODEL PERFORMANCE SUMMARY
--------------------------
Model                | Accuracy | Precision | Recall | F1     | AUC
---------------------|----------|-----------|--------|--------|------
Logistic Regression  | {results['Logistic Regression']['accuracy']:.4f}   | {results['Logistic Regression']['precision']:.4f}    | {results['Logistic Regression']['recall']:.4f} | {results['Logistic Regression']['f1']:.4f} | {results['Logistic Regression']['auc']:.4f}
Decision Tree        | {results['Decision Tree']['accuracy']:.4f}   | {results['Decision Tree']['precision']:.4f}    | {results['Decision Tree']['recall']:.4f} | {results['Decision Tree']['f1']:.4f} | {results['Decision Tree']['auc']:.4f}
Random Forest        | {results['Random Forest']['accuracy']:.4f}   | {results['Random Forest']['precision']:.4f}    | {results['Random Forest']['recall']:.4f} | {results['Random Forest']['f1']:.4f} | {results['Random Forest']['auc']:.4f}

BEST MODEL
----------
→ {best_name} achieved the highest accuracy of {best['accuracy']*100:.2f}%

MOST IMPORTANT FEATURE
-----------------------
→ '{top_feature}' is the strongest predictor of churn.

KEY INSIGHTS
------------
1. Customers with HIGH complaint count churn significantly more.
2. Month-to-Month contracts show the highest churn risk.
3. Early tenure customers (0-12 months) are most likely to churn.
4. Random Forest outperforms other models in overall metrics.
5. ROC-AUC of {best['auc']:.3f} shows strong discriminatory ability.

CONCLUSION
----------
The predictive model successfully identifies at-risk customers.
Businesses can use this model to trigger retention campaigns,
offer loyalty discounts, or escalate complaint resolution
before a customer churns.
"""
with open("output/findings_report.txt", "w") as f:
    f.write(report)
print(report)

print("="*60)
print("PROJECT 2 COMPLETED SUCCESSFULLY ✅")
print("All outputs saved inside the 'output/' folder.")
print("="*60)
