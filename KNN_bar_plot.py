import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

file_path = '/mnt/c/Users/jbtff/OneDrive/Documents/WISDM.csv'

df = pd.read_csv(file_path, header=None, names=['user', 'activity', 'timestamp', 'x', 'y', 'z'], low_memory=False)

df['x'] = pd.to_numeric(df['x'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df['z'] = pd.to_numeric(df['z'], errors='coerce')

df.dropna(subset=['x', 'y', 'z'], inplace=True)

df['activity'] = df['activity'].astype('category').cat.codes

X = df[['x', 'y', 'z']]
y = df['activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, y_pred_knn)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

models = ['kNN', 'Decision Tree']
accuracies = [knn_accuracy, dt_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['blue', 'green'])
plt.title('Accuracy Comparison between kNN and Decision Tree')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

plt.show()

print(f"kNN Classifier Accuracy: {knn_accuracy * 100:.2f}%")
print(f"Decision Tree Classifier Accuracy: {dt_accuracy * 100:.2f}%")
