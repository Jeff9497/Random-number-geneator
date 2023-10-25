 import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
# Provided numbers
provided_numbers = np.array([ 1.84,1.11,9.91,5.15,1.01,1.14,1.59,2.98,32.79,1.07,2.46,1.00,5.29,2.76,1.22,1.10,11.69,3.26,1.56,4.55,1.67,222.75,2.53,15.53,1.56,4.27,5.52,2.17,1.49,2.47,1.24,34.75,14.57,7.48,1.02,1.78,1.08,3.60,1.88,1.67,1.47,32.37,1.36,1.88,1.34,1.76,1.20,3.63,1.25,1.01,5.37,1.49,8.32,1.32,1.83,1.17,4.71,48.00,1.36
])

# Generate features and targets for training
# Using the index as a feature and the number as the target
X = np.arange(len(provided_numbers)).reshape(-1, 1)
y = provided_numbers

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVR model
model = SVR()
model.fit(X_train, y_train)

# Predict the next number (assuming the next index is len(provided_numbers))
next_index = len(provided_numbers)
predicted_next_number = model.predict([[next_index]])[0]

print("Provided Numbers:")
print("Predicted Next Number:", predicted_next_number)

