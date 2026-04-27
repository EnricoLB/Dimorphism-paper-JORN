# Classification Pipeline
# ==================================
# This script runs SVM vs DummyClassifier comparisons
# across three feature sets: Embeddings, MFCCs, and Standard features.
# It outputs violin plots for Accuracy and Recall.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score

# ==========================
# CONFIG
# ==========================
EMBEDDINGS_PATH = "data/birdnet_embeddings.csv"
MFCC_PATH = "data/mfcc_features.csv"
METADATA_PATH = "data/metadata.csv"

N_SPLITS = 100
TEST_SIZE = 0.4
RANDOM_STATE = 42
POS_LABEL = 'M'  # change if needed

# ==========================
# LOAD DATA
# ==========================
print("Loading data...")

embeddings = pd.read_csv(EMBEDDINGS_PATH)
mfcc = pd.read_csv(MFCC_PATH, header=None)
metadata = pd.read_csv(METADATA_PATH)

# Add metadata
for df in [embeddings, mfcc]:
    df["Species"] = metadata["Species"]
    df["Sex"] = metadata["Sex"]
    df["Individual"] = metadata["Individual"]

# Standard features
standard = metadata.select_dtypes(include=[np.number]).copy()
standard["Species"] = metadata["Species"]
standard["Sex"] = metadata["Sex"]
standard["Individual"] = metadata["Individual"]

feature_sets = {
    "Embeddings": embeddings,
    "MFCC": mfcc,
    "Standard": standard
}

species_list = metadata["Species"].unique()

results = []

# ==========================
# RUN EXPERIMENT
# ==========================
for feature_name, df in feature_sets.items():
    print(f"\nRunning feature set: {feature_name}")

    for sp in species_list:
        sp_data = df[df["Species"] == sp]

        if len(sp_data) < 10:
            continue

        X = sp_data.drop(columns=["Species", "Sex", "Individual"]).values
        y = sp_data["Sex"].values
        groups = sp_data["Individual"].values

        splitter = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        svm = make_pipeline(StandardScaler(), SVC(kernel="linear"))
        dummy = DummyClassifier(strategy="most_frequent")

        for train_idx, test_idx in splitter.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_test)) < 2:
                continue

            # --- SVM ---
            svm.fit(X_train, y_train)
            svm_pred = svm.predict(X_test)

            results.append({
                "Species": sp,
                "Feature": feature_name,
                "Classifier": "SVM",
                "Accuracy": accuracy_score(y_test, svm_pred),
                "Recall": recall_score(y_test, svm_pred, pos_label=POS_LABEL, zero_division=0)
            })

            # --- Dummy ---
            dummy.fit(X_train, y_train)
            dummy_pred = dummy.predict(X_test)

            results.append({
                "Species": sp,
                "Feature": feature_name,
                "Classifier": "Dummy",
                "Accuracy": accuracy_score(y_test, dummy_pred),
                "Recall": recall_score(y_test, dummy_pred, pos_label=POS_LABEL, zero_division=0)
            })

# ==========================
# PLOTTING
# ==========================
print("\nGenerating plots...")

results_df = pd.DataFrame(results)

sns.set(style="whitegrid")

for feature in results_df["Feature"].unique():
    subset = results_df[results_df["Feature"] == feature]

    # --- Accuracy plot ---
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=subset, x="Species", y="Accuracy", hue="Classifier", inner="quartile")
    plt.ylim(0, 1)
    plt.title(f"Accuracy Distribution - {feature}")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(f"outputs/{feature}_accuracy_violin.png", dpi=300)
    plt.close()

    # --- Recall plot ---
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=subset, x="Species", y="Recall", hue="Classifier", inner="quartile")
    plt.ylim(0, 1)
    plt.title(f"Recall Distribution - {feature}")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(f"outputs/{feature}_recall_violin.png", dpi=300)
    plt.close()

print("Done!")

# ==========================
# SAVE RESULTS
# ==========================
results_df.to_csv("outputs/all_results.csv", index=False)
