import numpy as np
import matplotlib.pyplot as plt
import p2
import pandas as pd
# Load and preprocess the Iris dataset
file_path = 'irisdata.csv'
df = pd.read_csv(file_path, header=0)

# Filter data to include only 'versicolor' and 'virginica'
filtered_data = df[df['species'].isin(['versicolor', 'virginica'])]

# Extract the first two features (assuming they are numeric columns)
X = filtered_data[[filtered_data.columns[2], filtered_data.columns[3]]].values
y = (filtered_data['species'] == 'virginica').astype(int)
# Function to generate random weights and bias
def generate_random_weights_and_bias(low_range, high_range):
    # Generate random weights within the specified range
    weight_vector = np.random.uniform(low=low_range, high=high_range, size=(2,))
    
    # Generate a random bias within the specified range
    bias = np.random.uniform(low=low_range, high=high_range)
    
    return weight_vector, bias
low_range = -1.0
high_range = 1.0
# Initial weights and bias
weight_vector, bias = generate_random_weights_and_bias(low_range, high_range)


# Learning rate for gradient descent
learning_rate = 0.01

# Number of iterations
iteration = 0

# Lists to store the loss for each iteration
loss_history = []
convergence_threshold = 1e-5

# Perform gradient descent
while True:
    # Compute gradient and update weights
    gradient, gradient_bias = p2.compute_gradient(X, y, weight_vector, bias)
    weight_vector = p2.update_weights(weight_vector, gradient, learning_rate)
    bias = p2.update_weights(bias, gradient_bias, learning_rate)

    # Calculate the loss (mean squared error)
    decision_boundary = np.dot(X, weight_vector) + bias
    sigmoid_output = 1 / (1 + np.exp(-decision_boundary))
    loss = np.mean((sigmoid_output - y) ** 2)
    loss_history.append(loss)

    # Plot the decision boundary and loss at each 100 iterations
    if iteration % 500 == 0:
        # Print the loss at each iteration (optional)
        print(f"Iteration {iteration}: Loss = {loss}")
        plt.figure(figsize=(12, 4))
        y_predicted = (sigmoid_output >= 0.5).astype(int)
        # Plot the decision boundary
        plt.subplot(1, 2, 1)
        p2.plot_decision_boundary(X, y_predicted, weight_vector, bias)
        plt.title(f'Decision Boundary (Iteration {iteration})')

        # Plot the loss
        plt.subplot(1, 2, 2)
        plt.plot(loss_history)
        plt.title('Loss over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')

        plt.tight_layout()
        plt.show()
        # Check for convergence
    if iteration > 0 and np.abs(loss_history[iteration] - loss_history[iteration - 1]) < convergence_threshold:
        # Print the loss at each iteration (optional)
        print(f"Final Iteration is {iteration}: Loss = {loss}")
        plt.figure(figsize=(12, 4))

        # Plot the decision boundary
        plt.subplot(2, 2, 1)
        p2.plot_decision_boundary(X, y_predicted, weight_vector, bias)
        plt.title(f'Final Decision Boundary (converged in  {iteration})')

        # Plot the loss
        plt.subplot(2, 2, 2)
        plt.plot(loss_history)
        plt.title('Loss over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        # Plot the loss
        plt.subplot(2, 2, 3)
        plt.title('Decision boundary over the real classifction')
        p2.plot_decision_boundary(X, y, weight_vector, bias)

        plt.tight_layout()
        
        plt.show()
        break
    iteration += 1
# Print final weights and bias
print("Final Weights:", weight_vector)
print("Final Bias:", bias)