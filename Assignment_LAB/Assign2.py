# ========================================
# ADVANCED CREDIT CARD FRAUD ANALYSIS
# (Different Visualization Style)
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_curve, roc_auc_score,
                             precision_recall_curve)

from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Modern theme
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (8,5)

# ----------------------------
# 1. Load Dataset
# ----------------------------
data = pd.read_csv("credit_card_fraud_dataset.csv")

if "TransactionDate" in data.columns:
    data.drop("TransactionDate", axis=1, inplace=True)

le = LabelEncoder()

if "TransactionType" in data.columns:
    data["TransactionType"] = le.fit_transform(data["TransactionType"])

if "Location" in data.columns:
    data["Location"] = le.fit_transform(data["Location"])

# ----------------------------
# 2. Class Distribution (Horizontal Bar)
# ----------------------------
plt.figure()
data["IsFraud"].value_counts().plot(kind="barh", color=["teal","crimson"])
plt.title("Transaction Class Distribution")
plt.xlabel("Count")
plt.ylabel("Class (0 = Legit, 1 = Fraud)")

# ----------------------------
# 3. Split Data
# ----------------------------
X = data.drop("IsFraud", axis=1)
y = data["IsFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ----------------------------
# 4. Before SMOTE Model
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("BEFORE SMOTE")
print(classification_report(y_test, y_pred))

# Confusion Matrix (Styled Differently)
plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, cmap="magma", fmt="d")
plt.title("Confusion Matrix (Imbalanced Data)")

# ----------------------------
# 5. Apply SMOTE
# ----------------------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Balanced Distribution (Donut Chart)
plt.figure()
counts = pd.Series(y_resampled).value_counts()
plt.pie(counts,
        labels=["Legitimate","Fraud"],
        autopct="%1.1f%%",
        wedgeprops=dict(width=0.4))
plt.title("Balanced Dataset After SMOTE")

# ----------------------------
# 6. After SMOTE Model
# ----------------------------
model_smote = LogisticRegression(max_iter=1000)
model_smote.fit(X_resampled, y_resampled)
y_pred_smote = model_smote.predict(X_test)
y_prob_smote = model_smote.predict_proba(X_test)[:,1]

print("AFTER SMOTE")
print(classification_report(y_test, y_pred_smote))

# Confusion Matrix (Different Theme)
plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred_smote),
            annot=True, cmap="viridis", fmt="d")
plt.title("Confusion Matrix (Balanced Data)")

# ----------------------------
# 7. ROC Curve
# ----------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob_smote)
roc_auc = roc_auc_score(y_test, y_prob_smote)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1])
plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

# ----------------------------
# 8. Precision-Recall Curve
# ----------------------------
precision, recall, _ = precision_recall_curve(y_test, y_prob_smote)

plt.figure()
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")

# ----------------------------
# 9. Fraud Probability Distribution
# ----------------------------
plt.figure()
sns.histplot(y_prob_smote, bins=30, kde=True)
plt.title("Fraud Probability Distribution")

# ----------------------------
# 10. Feature Importance (Styled)
# ----------------------------
importance = model_smote.coef_[0]
feat_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

plt.figure()
sns.barplot(x="Importance", y="Feature", data=feat_df)
plt.title("Feature Importance Ranking")

plt.show()
