import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score

# Load features
X = np.load("results/mfcc_features.npy")

# Load metadata
meta = pd.read_csv("data/metadata/annotations.csv")

y = meta["Sex"].values
groups = meta["Individual"].values

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Grouped split
gss = GroupShuffleSplit(n_splits=100, train_size=0.6, random_state=42)

precisions = []
recalls = []

for train_idx, test_idx in gss.split(X, y, groups):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = SVC(kernel="rbf")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    precisions.append(precision_score(y_test, y_pred, pos_label="M"))
    recalls.append(recall_score(y_test, y_pred, pos_label="M"))

print("Mean precision:", np.mean(precisions))
print("Mean recall:", np.mean(recalls))
