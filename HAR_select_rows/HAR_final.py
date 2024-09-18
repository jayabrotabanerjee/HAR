import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import resource

file_path = '/mnt/c/Users/jbtff/OneDrive/Documents/time_series_data_human_activities.csv'

nrows_per_activity = int(input("Enter the number of rows per activity to load: "))

activity_data = {
    'Walking': [],
    'Sitting': [],
    'Jogging': [],
    'Upstairs': [],
    'Downstairs': [],
    'Standing': []
}

chunksize = 10000
for chunk in pd.read_csv(file_path, chunksize=chunksize):
    for activity in activity_data.keys():
        if len(activity_data[activity]) < nrows_per_activity:
            filtered_chunk = chunk[chunk['activity'] == activity]
            activity_data[activity].append(filtered_chunk)

    if all(len(pd.concat(activity_data[act])) >= nrows_per_activity for act in activity_data):
        break

final_data = []
for activity, chunks in activity_data.items():
    concatenated_data = pd.concat(chunks)[:nrows_per_activity]
    final_data.append(concatenated_data)

df = pd.concat(final_data)

idle_data = pd.DataFrame({'x-axis': [0], 'y-axis': [0], 'z-axis': [0], 'activity': ['Idle']})
df = pd.concat([df, idle_data], ignore_index=True)

X = df[['x-axis', 'y-axis', 'z-axis']].values
y = df['activity'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()

training_time = end_time - start_time
memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Time: {training_time:.2f} seconds")
print(f"Memory Usage: {memory_usage} KB")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')

activity_colors = {
    'Sitting': 'blue',
    'Walking': 'green',
    'Jogging': 'red',
    'Downstairs': 'purple',
    'Upstairs': 'orange',
    'Standing': 'cyan',
    'Idle': 'black'
}

for activity, color in activity_colors.items():
    mask = np.array(y) == activity
    ax1.scatter(
        X[mask, 0],
        X[mask, 1],
        X[mask, 2],
        c=color,
        label=activity
    )

ax1.set_title('3D Scatter Plot of Activity Data')
ax1.set_xlabel('X-axis Acceleration')
ax1.set_ylabel('Y-axis Acceleration')
ax1.set_zlabel('Z-axis Acceleration')
ax1.legend()

ax2 = fig.add_subplot(122)
test_indices = np.arange(len(y_train), len(y_train) + len(y_test))
ax2.plot(test_indices, y_test, label='Actual Test Activity', color='green', linestyle='solid')
ax2.plot(test_indices, y_test_pred, label='Predicted Test Activity', color='orange', linestyle='dashed')
ax2.set_title('Testing Predictions')
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Activity')
ax2.legend()

plt.tight_layout()
plt.show()

def predict_activity(x, y, z):
    if x == 0 and y == 0 and z == 0:
        return 'Idle'
    input_data = np.array([[x, y, z]])
    prediction = model.predict(input_data)
    return prediction[0]

x_input = float(input("Enter X-axis acceleration: "))
y_input = float(input("Enter Y-axis acceleration: "))
z_input = float(input("Enter Z-axis acceleration: "))
predicted_activity = predict_activity(x_input, y_input, z_input)
print(f"The predicted activity is: {predicted_activity}")
