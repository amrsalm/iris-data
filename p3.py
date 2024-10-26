import numpy as np
import pandas as pd
import p2 

def calculate_mse(X, weight_vector, bias, y): 
    # Calculate decision boundary
    decision_boundary = np.dot(X, weight_vector) + bias

    # Apply sigmoid function
    sigmoid_output = p2.sigmoid(decision_boundary)

  
    # Convert sigmoid output to predicted classes (0 or 1)
    predicted_classes = (sigmoid_output >= 0.5).astype(int)

    # Calculate mean squared error
    mse = np.mean((predicted_classes - y) ** 2)

    return mse
# Load the Iris dataset
file_path = 'irisdata.csv'
df = pd.read_csv(file_path, header=0)

# Filter data to include only 'versicolor' and 'virginica'
filtered_data = df[df['species'].isin(['versicolor', 'virginica'])]
X = filtered_data[[filtered_data.columns[2], filtered_data.columns[3]]].values
y = np.where(filtered_data['species'] == 'versicolor', 0, 1)

if __name__ == "__main__":
    # X = filtered_data[[filtered_data.columns[2], filtered_data.columns[3]]].values
    # y = labels for the classes
        
    weight_vector = np.array([0.03, 0.2])
    bias = -0.86

    # Calculate Mean Squared Error
    mse = calculate_mse(X, weight_vector, bias, y)

    print(f"Mean Squared Error: {mse}")