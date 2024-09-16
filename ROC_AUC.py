import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from itertools import cycle

file_path = '/mnt/c/Users/jbtff/OneDrive/Documents/WISDM.csv'
df = pd.read_csv(file_path, header=None, names=['user', 'activity', 'timestamp', 'x', 'y', 'z'], low_memory=False)

df['x'] = pd.to_numeric(df['x'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df['z'] = pd.to_numeric(df['z'], errors='coerce')

df.dropna(subset=['x', 'y', 'z'], inplace=True)

df['activity'] = df['activity'].astype('category')

y = label_binarize(df['activity'], classes=df['activity'].unique())
X = df[['x', 'y', 'z']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier = OneVsRestClassifier(LogisticRegression(solver='lbfgs', max_iter=1000))
y_score = classifier.fit(X_train_scaled, y_train).predict_proba(X_test_scaled)

n_classes = y.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {df["activity"].cat.categories[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('One-vs-All ROC Curve with AUC for Multi-class')
plt.legend(loc="lower right")
plt.show()

for i in range(n_classes):
    print(f"Class {df['activity'].cat.categories[i]} AUC: {roc_auc[i]:.2f}")
