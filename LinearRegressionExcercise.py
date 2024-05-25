import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate a synthetic dataset with at least 200 instances
np.random.seed(20)
instances = 320
X = np.random.rand(instances, 2)  # Feature
y = 2 * X[:, 0] + 3 * X[:, 1] + 1 + 0.1 * np.random.randn(instances)  # Linear relationship with some noise

#---------------------------------------------------------------------------------------------------------
# num_features = 4
# X = np.random.rand(300, num_features)  # 'num_features' columns
# # Linear relationship with some noise
# true_coefficients = np.arange(1, num_features + 1)  # Coefficients 1, 2, 3, ..., num_features
# y = np.dot(X, true_coefficients) + 0.1 * np.random.randn(300)
#
# I added a variable num_features to specify the number of features.
# I used true_coefficients to generate coefficients based on the number of features, ranging from 1 to the number of features. Adjust this part based on your specific coefficients.
# The linear relationship is formed using np.dot(X, true_coefficients), where np.dot performs the dot product of the feature matrix X and the coefficients.
#---------------------------------------------------------------------------------------------------------

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
print('\n')
# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * 100
print(f"Root Mean Squared Error (RMSE) = {rmse:.2f}"+'%\n\n')
