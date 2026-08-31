"""
Heart Disease Prediction - ML Model Comparison
================================================
Trains and compares 4 classification algorithms:
  1. Decision Tree
  2. Random Forest
  3. AdaBoost
  4. Gradient Boosting

Dataset: Heart Attack Data Set (303 samples, 13 features)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ── Output directory ─────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & EXPLORE DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  STEP 1 : Loading & Exploring the Heart Attack Dataset")
print("=" * 70)

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "Heart Attack Data Set.csv")
df = pd.read_csv(csv_path)

print(f"\nDataset Shape : {df.shape}")
print(f"Features      : {list(df.columns[:-1])}")
print(f"Target Column : '{df.columns[-1]}'\n")

print("── First 5 Rows ───────────────────────────────────────────────────")
print(df.head().to_string(index=False))

print("\n── Statistical Summary ─────────────────────────────────────────────")
print(df.describe().round(2).to_string())

print("\n── Data Types ──────────────────────────────────────────────────────")
print(df.dtypes.to_string())

print("\n── Missing Values ──────────────────────────────────────────────────")
missing = df.isnull().sum()
print(missing[missing > 0].to_string() if missing.any() else "No missing values found ✓")

print("\n── Target Distribution ─────────────────────────────────────────────")
target_counts = df["target"].value_counts()
print(f"  1 (Heart Disease)    : {target_counts.get(1, 0)}")
print(f"  0 (No Heart Disease) : {target_counts.get(0, 0)}")

# ── Feature Distribution Histograms ──────────────────────────────────────────
print("\n  Showing feature distribution histograms...")
fig, axes = plt.subplots(3, 5, figsize=(18, 10))
axes = axes.flatten()
colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
          "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
          "#14b8a6", "#e11d48", "#0ea5e9", "#a855f7"]

feature_labels = {
    "age": "Age", "sex": "Gender (0=F, 1=M)", "cp": "Chest Pain Type",
    "trestbps": "Resting BP", "chol": "Cholesterol", "fbs": "Fasting Blood Sugar",
    "restecg": "Rest ECG", "thalach": "Max Heart Rate", "exang": "Exercise Angina",
    "oldpeak": "ST Depression", "slope": "Slope", "ca": "Major Vessels",
    "thal": "Thalassemia", "target": "Target"
}

for i, col in enumerate(df.columns):
    axes[i].hist(df[col], bins=20, color=colors[i], edgecolor="white", alpha=0.85)
    axes[i].set_title(feature_labels.get(col, col), fontsize=11, fontweight="bold")
    axes[i].set_xlabel(col, fontsize=9)
    axes[i].set_ylabel("Count", fontsize=9)

# hide the extra subplot (15th)
axes[-1].set_visible(False)

fig.suptitle("Feature Distributions — Heart Attack Dataset", fontsize=16, fontweight="bold")
fig.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 2 : Preprocessing")
print("=" * 70)

# Handle any missing rows (drop if any)
df.dropna(inplace=True)
print(f"  Rows after dropping NaNs : {len(df)}")

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# All columns are already numeric in this dataset – no encoding needed
print("  Categorical encoding     : Not required (all features numeric) ✓")

# Feature scaling with StandardScaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("  Feature scaling          : StandardScaler applied ✓")

# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 3 : Train-Test Split (80 / 20)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training samples : {len(X_train)}")
print(f"  Testing samples  : {len(X_test)}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 4 : Training All 4 Models")
print("=" * 70)

models = {
    "Decision Tree":     DecisionTreeClassifier(random_state=42),
    "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost":          AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
    }
    print(f"  ✓ {name:20s}  trained  |  Accuracy = {acc:.4f}")



# ══════════════════════════════════════════════════════════════════════════════
# 6. COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 6 : Performance Comparison Table")
print("=" * 70)

table = pd.DataFrame({
    name: {
        "Accuracy":  f"{r['accuracy']:.4f}",
        "Precision": f"{r['precision']:.4f}",
        "Recall":    f"{r['recall']:.4f}",
        "F1-Score":  f"{r['f1_score']:.4f}",
    }
    for name, r in results.items()
}).T

print("\n" + table.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# HEART ATTACK PREDICTION (Patient-Level)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  HEART ATTACK PREDICTIONS (Using Best Model: Random Forest)")
print("=" * 70)

best_rf = results["Random Forest"]["model"]
y_pred_rf = results["Random Forest"]["y_pred"]
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]  # probability of heart attack

# Show predictions for each test patient
pred_df = pd.DataFrame({
    "Patient #": range(1, len(y_test) + 1),
    "Actual": ["Heart Attack" if v == 1 else "No Heart Attack" for v in y_test.values],
    "Predicted": ["Heart Attack" if v == 1 else "No Heart Attack" for v in y_pred_rf],
    "Heart Attack Probability (%)": [f"{p*100:.1f}%" for p in y_proba_rf],
})
print("\n" + pred_df.to_string(index=False))

# Summary
total = len(y_pred_rf)
predicted_attacks = sum(y_pred_rf == 1)
predicted_no = sum(y_pred_rf == 0)
actual_attacks = sum(y_test.values == 1)
actual_no = sum(y_test.values == 0)

print(f"\n{'─' * 50}")
print(f"  HEART ATTACK RATE SUMMARY (Test Set: {total} patients)")
print(f"{'─' * 50}")
print(f"  Predicted Heart Attacks : {predicted_attacks}/{total} ({predicted_attacks/total*100:.1f}%)")
print(f"  Predicted No Attack     : {predicted_no}/{total} ({predicted_no/total*100:.1f}%)")
print(f"  Actual Heart Attacks    : {actual_attacks}/{total} ({actual_attacks/total*100:.1f}%)")
print(f"  Actual No Attack        : {actual_no}/{total} ({actual_no/total*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 7. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 7 : Generating Visualisations")
print("=" * 70)

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]   # blue, green, amber, red

# ── 7x. Target Distribution – Heart Attack vs No Heart Attack ────────────────
target_counts = df["target"].value_counts().sort_index()
labels_dist = ["No Heart Attack", "Heart Attack"]
counts = [target_counts.get(0, 0), target_counts.get(1, 0)]
colors_dist = ["#3b82f6", "#ef4444"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
bars = ax1.bar(labels_dist, counts, color=colors_dist, edgecolor="white",
               linewidth=1.5, width=0.5)
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
             str(count), ha="center", va="bottom", fontsize=14, fontweight="bold")
ax1.set_ylabel("Number of Patients", fontsize=12)
ax1.set_title("Patient Count", fontsize=13, fontweight="bold")
ax1.set_ylim(0, max(counts) + 25)

# Pie chart
wedges, texts, autotexts = ax2.pie(
    counts, labels=labels_dist, colors=colors_dist, autopct="%1.1f%%",
    startangle=90, textprops={"fontsize": 12},
    wedgeprops={"edgecolor": "white", "linewidth": 2})
for t in autotexts:
    t.set_fontweight("bold")
ax2.set_title("Percentage Split", fontsize=13, fontweight="bold")

fig.suptitle("Heart Attack Distribution in Dataset", fontsize=15, fontweight="bold")
fig.tight_layout()
path_dist = os.path.join(OUT_DIR, "target_distribution.png")
fig.savefig(path_dist, dpi=150)
print(f"  ✓ Saved: {path_dist}")
plt.show()

# ── 7a. Accuracy & F1-Score Grouped Bar Chart ────────────────────────────────
model_names = list(results.keys())
accuracies  = [results[n]["accuracy"]  for n in model_names]
f1_scores   = [results[n]["f1_score"]  for n in model_names]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy",
               color=PALETTE, edgecolor="white", linewidth=1.2, alpha=0.85)
bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1-Score",
               color=PALETTE, edgecolor="white", linewidth=1.2, alpha=0.55,
               hatch="//")

# value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
            fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
            fontweight="bold")

ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison — Accuracy & F1-Score", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 1.12)
ax.legend(fontsize=11)
fig.tight_layout()
path_acc = os.path.join(OUT_DIR, "accuracy_f1_comparison.png")
fig.savefig(path_acc, dpi=150)
print(f"  ✓ Saved: {path_acc}")
plt.show()

# ── 7b. All-Metrics Grouped Bar Chart ───────────────────────────────────────
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(model_names))
total_width = 0.75
bar_w = total_width / len(metrics_names)

for i, metric in enumerate(metrics_names):
    key = metric.lower().replace("-", "_")
    vals = [results[n][key] for n in model_names]
    offset = (i - len(metrics_names) / 2 + 0.5) * bar_w
    bars = ax.bar(x + offset, vals, bar_w, label=metric,
                  edgecolor="white", linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold")

ax.set_ylabel("Score", fontsize=12)
ax.set_title("All Metrics Comparison Across Models", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10, ncol=4, loc="upper center")
fig.tight_layout()
path_all = os.path.join(OUT_DIR, "all_metrics_comparison.png")
fig.savefig(path_all, dpi=150)
print(f"  ✓ Saved: {path_all}")
plt.show()

# ── 7c. Confusion Matrices (2×2 subplot grid) ───────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, res) in enumerate(results.items()):
    cm = res["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"],
                linewidths=1.5, linecolor="white",
                annot_kws={"size": 16, "weight": "bold"})
    axes[idx].set_title(name, fontsize=13, fontweight="bold")
    axes[idx].set_xlabel("Predicted", fontsize=11)
    axes[idx].set_ylabel("Actual", fontsize=11)

fig.suptitle("Confusion Matrices for All Models", fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
path_cm = os.path.join(OUT_DIR, "confusion_matrices.png")
fig.savefig(path_cm, dpi=150, bbox_inches="tight")
print(f"  ✓ Saved: {path_cm}")
plt.show()

# ── 7d. Feature Importance (Random Forest) ──────────────────────────────────
rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
feat_imp.plot.barh(ax=ax, color=sns.color_palette("viridis", len(feat_imp)),
                   edgecolor="white", linewidth=0.8)
ax.set_title("Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance", fontsize=12)
fig.tight_layout()
path_fi = os.path.join(OUT_DIR, "feature_importance_rf.png")
fig.savefig(path_fi, dpi=150)
print(f"  ✓ Saved: {path_fi}")
plt.show()

# ── 7e. Radar / Spider chart ────────────────────────────────────────────────
labels = metrics_names
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close polygon

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for idx, name in enumerate(model_names):
    values = [results[name][m.lower().replace("-", "_")] for m in labels]
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2, label=name, color=PALETTE[idx])
    ax.fill(angles, values, alpha=0.1, color=PALETTE[idx])

ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("Model Performance Radar Chart", fontsize=14, fontweight="bold", y=1.08)
ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0), fontsize=10)
fig.tight_layout()
path_radar = os.path.join(OUT_DIR, "radar_chart.png")
fig.savefig(path_radar, dpi=150, bbox_inches="tight")
print(f"  ✓ Saved: {path_radar}")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 8. FINAL CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 8 : Final Conclusion")
print("=" * 70)

best_model = max(results, key=lambda n: results[n]["f1_score"])
best_f1    = results[best_model]["f1_score"]
best_acc   = results[best_model]["accuracy"]

print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │  BEST MODEL  :  {best_model:<46s} │
  │  F1-Score    :  {best_f1:<46.4f} │
  │  Accuracy    :  {best_acc:<46.4f} │
  └──────────────────────────────────────────────────────────────────┘
""")
