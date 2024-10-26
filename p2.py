import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
file_path = 'irisdata.csv'
df = pd.read_csv(file_path, header=0)

# Filter data to include only 'versicolor' and 'virginica'
filtered_data = df[df['species'].isin(['versicolor', 'virginica'])]

X = filtered_data[[filtered_data.columns[2], filtered_data.columns[3]]].values
X_full = filtered_data.iloc[:, :-1].values

def sigmoid(x):
        #applying the sigmoid function
     return 1 / (1 + np.exp(-np.array(x)))

def simple_neural_network(input_vector, weight_matrix, bias_vector):
    """Simple neural network with one output node."""
    linear_output = np.dot(weight_matrix, input_vector) + bias_vector
    activation_output = sigmoid(np.array(linear_output).astype(float))
    return activation_output

def plot_2nd_3rd(X, y):
    """Plot decision boundary and overlay sigmoid non-linearity."""
    
    # Plot filtered data
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='Set1', legend='full')

def plot_decision_boundary(X,y,weight_vector,bias):
    plot_2nd_3rd(X,y)
    decision_boundary = np.dot(X, weight_vector) + bias
    sigmoid(decision_boundary)
    plt.axline((0, -bias/weight_vector[1]), slope=-weight_vector[0]/weight_vector[1], color='red', linestyle='--', label='Decision Boundary')



def logistic_regression_hypothesis(X, weight_vector, bias):
    return sigmoid(np.dot(X, weight_vector) + bias)

def plot_logistic_regression_hypothesis(X, weight_vector, bias):
    # Generate a grid of points in the input space for the two features of interest
    feature1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    feature2_vals = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    feature1_grid, feature2_grid = np.meshgrid(feature1_vals, feature2_vals)
    
    # Create input space with two features of interest
    input_space = np.c_[feature1_grid.ravel(), feature2_grid.ravel()]

    # Calculate logistic regression hypothesis over the entire input space
    hypothesis = logistic_regression_hypothesis(input_space, weight_vector, bias)

    # Reshape hypothesis to match the shape of feature1_grid
    hypothesis = hypothesis.reshape(feature1_grid.shape)

    # Plot the 3D surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the logistic regression hypothesis as a 3D surface
    ax.plot_surface(feature1_grid, feature2_grid, hypothesis, cmap='viridis', alpha=0.8)

    ax.set_title('Logistic Regression Hypothesis')
    ax.set_xlabel('Petal length')
    ax.set_ylabel('Petal width')
    ax.set_zlabel('Hypothesis Output')

    plt.show()

def show_classifier_output(X, y, weight_vector, bias):
    # Use the plot_decision_boundary function directly
    decision_boundary = np.dot(X, weight_vector) + bias

    # Apply sigmoid function
    sigmoid_output = sigmoid(decision_boundary)
    plt.axline((0, -bias/weight_vector[1]), slope=-weight_vector[0]/weight_vector[1], color='red', linestyle='--', label='Decision Boundary')

    # Convert sigmoid output to predicted classes (0 or 1)
    predicted_classes = (sigmoid_output >= 0.5).astype(int)

    # Plot the classifier's output
    plt.scatter(X[predicted_classes == 0, 0], X[predicted_classes == 0, 1], c='blue', marker='o', label='Predicted Class 2nd ')
    plt.scatter(X[predicted_classes == 1, 0], X[predicted_classes == 1, 1], c='green', marker='^', label='Predicted Class 3rd')

    # Mark points with actual class labels
    # Plot points with actual class labels
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='black', marker='x', label='Actual Class 2nd')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', marker='x', label='Actual Class 3rd')


    plt.title('Classifier Output and Decision Boundary')
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend()
    plt.show()
def compute_gradient(X, y, weight_vector, bias):
    # Calculate decision boundary
    decision_boundary = np.dot(X, weight_vector) + bias
    
    # Apply sigmoid function
    sigmoid_output = sigmoid(decision_boundary)
    
    # Compute gradient using vectorized expression
    gradient = np.dot(X.T, (sigmoid_output - y) * sigmoid_output * (1 - sigmoid_output))
    gradient_bias = np.sum((sigmoid_output - y) * sigmoid_output * (1 - sigmoid_output))
    
    return gradient, gradient_bias

def update_weights(weight_vector, gradient, learning_rate):
    # Update weights
    weight_vector -= learning_rate * gradient
    
    return weight_vector

def update_small_step(X,y,weight_vector,gradient):
    learning_rate = 0.01
    # Plot initial decision boundary
    plt.figure(figsize=(8, 6))
    plot_decision_boundary(X, y, weight_vector, bias)
    plt.title('Initial Decision Boundary')
    plt.show()

    # Perform one update step and plot the updated decision boundary
    gradient, gradient_bias = compute_gradient(X, y, weight_vector, bias)
    weight_vector = update_weights(weight_vector, gradient, learning_rate)

    plt.figure(figsize=(8, 6))
    plot_decision_boundary(X, y, weight_vector, bias)
    plt.title('Updated Decision Boundary')
    plt.show()

if __name__ == "__main__":

    # Set handcrafted weights and bias for the decision boundary
    weight_vector = np.array([0.05, 0.36])
    bias = -0.86  # Adjust this value
    y = np.where(filtered_data['species'] == 'versicolor', 0, 1)
    # Assume input_vector has shape (2,) in this case
    input_vector = X[0, :]  # Using the first data point for illustration
    print("Input of the neural network:", input_vector)

    output = simple_neural_network(input_vector, weight_vector, bias)
    print("Output of the neural network:", output)

    # Scatter plot of Iris classes
    sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=filtered_data, palette='Set1')
    plt.title('Scatter Plot of Iris Classes (versicolor and virginica)')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    # Plot decision boundary
    plt.show()
    plot_decision_boundary(X,y,weight_vector,bias)
    plot_logistic_regression_hypothesis(X,weight_vector,bias)
    # Manually selected points for testing
    test_points = np.array([[5.3, 2.3], [6.9, 2.3], [6.4, 2.0], [3.9, 1.2], [3.5, 1.0], [3.8, 1.1], [5.1, 1.6], [5.1, 1.9]])
    actual_labels = np.array([1, 1, 1, 0, 0, 0, 0, 1])  # 1 for Virginica, 0 for Versicolor

    show_classifier_output(test_points, actual_labels, weight_vector, bias)
   
    
    update_small_step(X,y,weight_vector,bias)

