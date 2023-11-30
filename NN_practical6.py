import numpy as np
import pandas as pd

dataset_path = "C:\\Users\\advai\\Downloads\\heartdisease\\heart.csv"
dataset = pd.read_csv(dataset_path)
print(dataset.dtypes)
# Display the first few rows of the dataset to inspect its structure
print(dataset.head())
# Choose features for X (input) [1025 x 13]
feature_columns = ['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']  
X = dataset[feature_columns].values
# Choose the target variable for y (output)
target_column = 'target'  # Replace with actual target column name
y = dataset[target_column].values

# extracts the data from the specified columns and assigns it to the variable X and y
X = dataset[feature_columns].values
y = dataset[target_column].values

# Normalize the input data
def standardize_data(X_train, X_test):
# The axis=0 argument specifies that the mean should be computed along the columns. 
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train_normalized = (X_train - mean) / std
    X_test_normalized = (X_test - mean) / std

    return X_train_normalized, X_test_normalized

# Split the dataset into training and testing sets
# 20% data is used for testing purpose & random splitting happens everytime on running code
def train_test_split(X, y, test_size=0.2, random_state=None):
    
    num_samples = len(X) #finds total saples i.e. no of rows in dataset
    indices = np.random.permutation(num_samples)# used for shuffling the samples(0 to n-1) to increase the randomness in data
# calculates the index at which you want to split your dataset into training and testing sets
#since 0.2 test size only defines the proportion of division so (0.8* samples) means samples in training set
    split_index = int((1 - test_size) * num_samples)                                                    
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

    return X_train, X_test, y_train, y_test

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network architecture i.e. initializing the weights & bias
input_size = X.shape[1] # no of input neurons, here columns are input neurons.
hidden_size = 2         # no of neurons in hidden layer
output_size = 1         # no of neurons in output layer

# Initialize weights and biases with random values
weights_input_hidden = np.random.rand(input_size, hidden_size) # weight for input layer to hidden layer
bias_hidden = np.zeros((1, hidden_size)) # bias for hidden layer
weights_hidden_output = np.random.rand(hidden_size, output_size) # weight for hidden layer to output layer
bias_output = np.zeros((1, output_size))  # bias for output layer

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Custom train-test split function
def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    
    num_samples = len(X)
    indices = np.random.permutation(num_samples)
    split_index = int((1 - test_size) * num_samples)

#actually splits the target and features using indexing
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

    return X_train, X_test, y_train, y_test

# Split the dataset
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

# Training the neural network
def train_neural_network(inputs, targets, epochs, learning_rate):
    global weights_input_hidden, bias_hidden, weights_hidden_output, bias_output  # Declare the variables as global

#iterate through each epoch
    for epoch in range(epochs):
        # inner loop iterate through each data & target pair & zip combine both as 1 array element 
        for input_data, target in zip(inputs, targets):
            input_data = np.array([input_data])  # Neural networks often expect input data in 2D format
            target = np.array([target])  # Convert target to 2D array

            # Forward pass
            
            #calculating output for input layer sum(xi * wi)+b
            hidden_layer_input = np.dot(input_data, weights_input_hidden) + bias_hidden 
                # apply activation function
            hidden_layer_output = relu(hidden_layer_input)  
            hidden_layer_output = sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
            predicted_output = relu(output_layer_input)
            predicted_output = sigmoid(output_layer_input)
            
            # Backward pass
            error = target - predicted_output
            output_delta = error * sigmoid_derivative(predicted_output)  #finding the delta error term for output layer
            # Calculate the errors in the hidden layer by taking the dot product of the output delta & transpose of the weights connecting hidden & output layers.
            hidden_output_errors = output_delta.dot(weights_hidden_output.T)
            # Calculate the delta for the hidden layer
            hidden_delta = hidden_output_errors * relu_derivative(hidden_layer_output)
            hidden_delta = hidden_output_errors * sigmoid_derivative(hidden_layer_output)

            # Update weights and biases
            # Update the weights between the hidden & output layers
            weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
            # Update the biases of the output layer
            bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
            #Update the weights between the input and hidden layers.
            weights_input_hidden += input_data.T.dot(hidden_delta) * learning_rate
            # Update the biases of the hidden layer
            bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

def mean_square_error(predictions, target):
        return np.mean((predictions - target) ** 2)

# Normalize the input data
X_train_normalized, X_test_normalized = standardize_data(X_train, X_test)

# Training the neural network with ReLU activation

train_neural_network(X_train_normalized, y_train, epochs=50, learning_rate=0.1)

# Test the trained network with ReLU activation
predictions_relu = []
for input_data in X_test_normalized:
    input_data = np.array([input_data])
    hidden_layer_input = np.dot(input_data, weights_input_hidden) + bias_hidden
    hidden_layer_output = relu(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output,weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    predictions_relu.append(np.round(predicted_output))

# Accuracy for ReLU activation
accuracy_relu = np.mean(np.equal(predictions_relu, y_test))
print("Accuracy (ReLU):", accuracy_relu)

# Training the neural network with Sigmoid activation

train_neural_network(X_train_normalized, y_train, epochs=50, learning_rate=0.1)

# Test the trained network with Sigmoid activation
predictions_sigmoid = []
for input_data in X_test_normalized:
    input_data = np.array([input_data])
    hidden_layer_input = np.dot(input_data, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    predictions_sigmoid.append(np.round(predicted_output))

# Accuracy for Sigmoid activation
accuracy_sigmoid = np.mean(np.equal(predictions_sigmoid, y_test))
print("Accuracy (Sigmoid):", accuracy_sigmoid)

# Calculate MSE for  activation
mse_sigmoid = mean_square_error(predictions_sigmoid, y_test)
print("MSE (Sigmoid):", mse_sigmoid)

train_neural_network(X_train_normalized, y_train, epochs=50, learning_rate=0.1)

# Test the trained network with ReLU activation
predictions_relu = [] #making an empty list to store the result
#iterating for each value of test dataset to check the training process.
for input_data in X_test_normalized:
    input_data = np.array([input_data])
    # carrying forward pass using the weights from training
    hidden_layer_input = np.dot(input_data, weights_input_hidden) + bias_hidden
    hidden_layer_output = relu(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output,weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    predictions_relu.append(np.round(predicted_output))

# Accuracy for ReLU activation
accuracy_relu = np.mean(np.equal(predictions_relu, y_test))
print("Accuracy (ReLU):", accuracy_relu)

# Calculate MSE for ReLU activation
mse_relu = mean_square_error(predictions_relu, y_test)
print("MSE (ReLU):", mse_relu)

# Training the neural network with Sigmoid activation
train_neural_network(X_train_normalized, y_train, epochs=50, learning_rate=0.1)

# Test the trained network with Sigmoid activation
predictions_sigmoid = []
for input_data in X_test_normalized:
    input_data = np.array([input_data])
    hidden_layer_input = np.dot(input_data, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    predictions_sigmoid.append(np.round(predicted_output)) # rounds to 0 or 1

# Accuracy for Sigmoid activation
accuracy_sigmoid = np.mean(np.equal(predictions_sigmoid, y_test)) # carry out the comparison
print("Accuracy (Sigmoid):", accuracy_sigmoid)

# Calculate MSE for  activation
mse_sigmoid = mean_square_error(predictions_sigmoid, y_test)
print("MSE (Sigmoid):", mse_sigmoid)

train_neural_network(X_train_normalized, y_train, epochs=50, learning_rate=0.1)

# Test the trained network with ReLU activation
predictions_relu = []
for input_data in X_test_normalized:
    input_data = np.array([input_data])
    hidden_layer_input = np.dot(input_data, weights_input_hidden) + bias_hidden
    hidden_layer_output = relu(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output,weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    predictions_relu.append(np.round(predicted_output))

# Accuracy for ReLU activation
accuracy_relu = np.mean(np.equal(predictions_relu, y_test))
print("Accuracy (ReLU):", accuracy_relu)

# Calculate MSE for ReLU activation
mse_relu = mean_square_error(predictions_relu, y_test)
print("MSE (ReLU):", mse_relu)
