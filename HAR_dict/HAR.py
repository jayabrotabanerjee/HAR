import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import resource

data = {
    'activity': [
        'Sitting', 'Sitting', 'Sitting', 'Sitting', 'Sitting', 'Sitting',
        'Walking', 'Walking', 'Walking', 'Walking', 'Walking', 'Walking',
        'Jogging', 'Jogging', 'Jogging', 'Jogging', 'Jogging', 'Jogging',
        'Downstairs', 'Downstairs', 'Downstairs', 'Downstairs', 'Downstairs',
        'Upstairs', 'Upstairs', 'Upstairs', 'Upstairs', 'Upstairs', 'Upstairs',
        'Standing', 'Standing', 'Standing', 'Standing', 'Standing', 'Standing'
    ],
    'x-axis': [
        5.86, 5.79, 5.79, 5.83, 5.86, 5.9,
        -2.34, 4.44, -3.87, -4.99, -2.96, -2.37,
        11.18, -0.46, -4.9, -2.91, 5.18, 6.74,
        1.69, 1.69, 0.5, 0.5, 0.99,
        0.46, 0, 1.5, 1.08, -0.53, -0.42, 6.02,
        -1.04, -0.95, -0.93, -0.95, -0.93, -0.95
    ],
    'y-axis': [
        7.74, 7.74, 7.82, 7.78, 7.74, 7.78,
        5.6, 19.57, 4.99, 14.9, 5.86, 7.86,
        15.66, 0.84, -3.57, -1.42, 10.42, 12.98,
        8.54, 8.54, 7.08, 7.08, 6.89,
        6.59, 6.02, 4.37, 1.88, 0.8, 4.9, 18.96,
        9.43, 9.53, 9.47, 9.38, 9.47, 9.43
    ],
    'z-axis': [
        2.56, 2.6, 2.56, 2.56, 2.53, 2.56,
        6.09, 16.32, 0.42, 4.14, -3.45, -3.15,
        5.86, 1.23, -0.72, -0.04, 6.93, -4.29,
        6.66035, 6.66035, -1.1849703, -1.1849703, 4.671779,
        -0.53119355, 3.568531, 2.6832085, 2.9556155, 2.9147544, 12.026767, 4.7126403,
        -0.08, -0.04, -0.23, -0.15, -0.15, -0.19
    ]
}

X = np.column_stack((data['x-axis'], data['y-axis'], data['z-axis']))
y = np.array(data['activity'])

if len(X) != len(y):
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

idle_data = np.array([[0, 0, 0]])
X = np.vstack((X, idle_data))
y = np.append(y, 'Idle')

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
