import numpy as np
import pandas as pd
import tensorflow as tf
import copy


# Read the data into pandas DataFrame
df = pd.read_csv('data_daily.csv')

df = df.rename(columns={'# Date': 'date', 'Receipt_Count': 'receipt_count'})
df['date'] = pd.to_datetime(df['date'])

# Feature Engineering
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['day_of_month'] = df['date'].dt.day

# Perform cyclical encoding for the days and month
# This helps to maintain the circular nature of days and months
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 7)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 7)

# Drop the original days and months columns
df = df.drop(['day_of_week', 'month', 'day_of_month'], axis=1)

# Split the data into features (X) and target variable (y)
X = df.drop(['date', 'receipt_count'], axis=1)
y = df['receipt_count']

# Normalize the target variable by dividing by 1e7
y = y / 1e7

# Split the data into training, validation and test sets
# Use date to index the data to allow splitting with months
X.index = df.date
y.index = df.date

# Train with 9 months of data
train_set = X.loc[:'2021-09-30']
train_label = y.loc[:'2021-09-30']

# Validate with 2 months of data
validation_set = X.loc['2021-10-01':'2021-11-30']
validation_label = y.loc['2021-10-01':'2021-11-30']

# Test with one month of data
test_set = X.loc['2021-12-01':]
test_label = y.loc['2021-12-01':]

# Convert the data to numpy arrays
train_set = np.array(train_set)
train_label = np.array(train_label)

validation_set = np.array(validation_set)
validation_label = np.array(validation_label)

test_set = np.array(test_set)
test_label = np.array(test_label)

# Build the receipt_prediction
# Initialize weights
initializer = tf.keras.initializers.GlorotUniform(seed=42)
w_init = initializer(shape=(test_set.shape[1],))
b_init = 0.
def predict(X, w, b):
    """
    Predict using linear regression
    :param X(ndarray): training examples with multiple features, shape (n,)
    :param w(ndarray): receipt_prediction parameter, shape (n,)
    :param b(scalar): receipt_prediction parameter
    :returns:
        prediction(scalar): prediction
    """
    prediction = np.dot(X, w) + b
    return prediction

def compute_loss(X, y, w, b):
    """
    Computes loss
    :param X (ndarray): Training data, m examples with n features, shape: (m, n)
    :param y (ndarray): target values, shape (m,)
    :param w (ndarray): receipt_prediction weight, shape (n,)
    :param b (scalar): receipt_prediction bias
    :returns:
        loss (scalar) : loss
    """
    m = X.shape[0]
    loss = 0.
    for i in range(m):
        forward_pass_i = np.dot(X[i], w) + b
        loss = loss + np.square((forward_pass_i - y[i]))
    loss = loss / (2 * m)
    return loss

def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression
    :param X (ndarray): Training data, m examples with n features, shape: (m, n)
    :param y (ndarray): target values, shape (m,)
    :param w (ndarray): receipt_prediction weight, shape (n,)
    :param b (scalar): receipt_prediction bias
    :returns:
        loss_grad_w: The gradient of the cost w.r.t the parameters w
        loss_grad_b: The gradient of the cost w.r.t the parameter b
    """
    m, n = X.shape
    loss_grad_w = np.zeros((n,))
    loss_grad_b = 0.

    for i in range(m):
        error = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            loss_grad_w[j] = loss_grad_w[j] + error * X[i, j]
        loss_grad_b = loss_grad_b + error
    loss_grad_w = loss_grad_w / m
    loss_grad_b = loss_grad_b / m

    return loss_grad_w, loss_grad_b

def gradient_descent(X, y, w_in, b_in, learning_rate=0.01, num_iters=1000, verbose=False):
    """
    Performs batch gradient descent
    :param X (ndarray): Training data, m examples with n features, shape: (m, n)
    :param y (ndarray): target values, shape (m,)
    :param w_in (ndarray): receipt_prediction weight, shape (n,)
    :param b_in (scalar): receipt_prediction bias
    :param learning_rate: learning rate
    :param num_iters: number of iteration to run gradient descent
    :param verbose: print training progress
    :returns:
        w: updated values of parameters (weights)
        b: updated values of parameter (bias)
    """
    loss_history = []
    history = {}
    w = copy.deepcopy(w_in) # avoid modifying global w within the function
    b = b_in

    for i in range(num_iters):
        loss_grad_w, loss_grad_b = compute_gradient(X, y, w, b)
        w -= learning_rate * loss_grad_w
        b -= learning_rate * loss_grad_b

        loss_history.append(round(compute_loss(X, y, w, b), 5))

        if verbose:
            print(f'Iteration {i}: Cost {loss_history[-1]:.5f}')

    history['weights'] = w
    history['bias'] = round(b, 5)
    history['loss'] = loss_history

    return history

# Model training
learning_rate = 0.01
num_epochs = 1000

history = gradient_descent(X=train_set, y=train_label, w_in=w_init, b_in=b_init,
                           learning_rate=learning_rate, num_iters=num_epochs)

print("\nLoss: ", history['loss'])
print("\nWeights: ", history['weights'])
print("\nBias: ", history['bias'])

# Prediction
weights = history['weights']
bias = history['bias']

prediction = predict(validation_set, weights, bias)
# print(prediction)
# print(validation_set)

mae = tf.keras.metrics.mean_absolute_error(validation_label, prediction)
mse = tf.keras.metrics.mean_squared_error(validation_label, prediction)

print(f"\nMean absolute error on validation set: {mae:.5f}")
print(f"\nMean squared error on validation set: {mse:.5f}")

# Evaluate the receipt_prediction with the test set
pred_test = predict(test_set, weights, bias)
mae = tf.keras.metrics.mean_absolute_error(test_label, pred_test)
mse = tf.keras.metrics.mean_squared_error(test_label, pred_test)

print(f"\nMean absolute error on test set: {mae:.5f}")
print(f"\nMean squared error on test set: {mse:.5f}\n")

# Predict Receipt count for the following year
# Create a DataFrame for the following year
following_year = pd.DataFrame(pd.date_range(start='2022-01-01', end='2022-12-31', freq='D'), columns=['date'])

# Feature Engineering
following_year['day_of_week'] = following_year['date'].dt.dayofweek
following_year['month'] = following_year['date'].dt.month
following_year['day_of_month'] = following_year['date'].dt.day

# Perform cyclical encoding for the days and month
# This helps to maintain the circular nature of days and months
following_year['day_of_week_sin'] = np.sin(2 * np.pi * following_year['day_of_week'] / 7)
following_year['day_of_week_cos'] = np.cos(2 * np.pi * following_year['day_of_week'] / 7)

following_year['day_of_month_sin'] = np.sin(2 * np.pi * following_year['day_of_month'] / 31)
following_year['day_of_month_cos'] = np.cos(2 * np.pi * following_year['day_of_month'] / 31)

following_year['month_sin'] = np.sin(2 * np.pi * following_year['month'] / 7)
following_year['month_cos'] = np.cos(2 * np.pi * following_year['month'] / 7)

# Drop original day, month columns
date = following_year['date']
following_year = following_year.drop(['date', 'day_of_week', 'day_of_month', 'month'], axis=1)

# Make predictions for the following year
predictions_following_year = predict(following_year, weights, bias)

# Scale the predictions by the normalization factor of 1e7
predictions_following_year = predictions_following_year * 1e7

# Combine date and predicted values into a DataFrame
result_following_year = pd.DataFrame({'date': date, 'predicted_receipt_count': predictions_following_year})

# Print  the predictions
print('Predictions for year 2022:\n', result_following_year[:20])

# Combine date and predicted values into a DataFrame
result_following_year = pd.DataFrame({'date': date, 'predicted_receipt_count': predictions_following_year})

# Print  the predictions
print('Predictions for year 2022:\n', result_following_year[:20])