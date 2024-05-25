import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Generating a random dataset
np.random.seed(42) #This line sets the random seed for the NumPy random number generator.
# By setting a seed, you ensure reproducibility of the random numbers generated.
# That is, if you run the code again later, you will get the same random numbers

X = np.random.rand(280, 6)  # Features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # creating a binary target variable y based on whether the sum of values in the first and second columns of X is greater than 1.
# If the sum is greater than 1, y is set to 1; otherwise, it's set to 0.


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #random_state value has same functionality as random seed value

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('\n')
print("Confusion Matrix:")
print(conf_matrix)
print('\n')

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}"+ '%' + '\n')
# using an f-string to print the accuracy with a specific formatting.
#{accuracy:.2f}: This is the f-string formatting expression for inserting the value of accuracy into the string.
# The .2f specifies that the floating-point number (f) should be formatted with two digits after the decimal point (.2).
