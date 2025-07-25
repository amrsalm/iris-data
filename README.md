# Iris Species Clustering, Classification, and Neural Network Project

This project explores clustering, classification, and neural network models using the classic Iris dataset. Key techniques implemented include K-means clustering, logistic regression, and a neural network classifier for multi-class species classification.

## Project Structure

### Main Algorithms Implemented

1. **K-means Clustering**
   - A custom implementation of K-means clustering to group data based on petal dimensions.
   - Functions include:
     - Random centroid initialization.
     - Cluster assignment and centroid recalculation.
     - Distortion value computation to monitor convergence.

2. **Binary Classification Using Logistic Regression**
   - A simple neural network implementation to classify between `versicolor` and `virginica` species.
   - Functions include:
     - `sigmoid`: Calculates the sigmoid function for activation.
     - `simple_neural_network`: A minimal neural network function to produce binary classifications.
     - `logistic_regression_hypothesis`: Produces logistic regression output.
   - Evaluated with mean squared error (MSE) and distortion tracking.

3. **Multi-Layer Perceptron (MLP) Classifier Using Scikit-Learn**
   - Uses an **MLPClassifier** to classify all three Iris species (`setosa`, `versicolor`, and `virginica`) based on all four feature dimensions.
   - Functions include:
     - `MLPClassifier` model creation with one hidden layer.
     - Train-test splitting, training, and prediction on the test set.
     - Model evaluation with accuracy score and confusion matrix visualization.

4. **Decision Boundary and Learning Process Visualization**
   - Uses **Matplotlib** and **Seaborn** to create:
     - Decision boundaries for K-means clustering and logistic regression.
     - A confusion matrix heatmap for the MLPClassifier’s performance.

### Core Functions and Flow
- **Data Visualization**: `plot_data` and `plot_decision_boundaries` visualize initial data points and decision boundaries.
- **Distortion Calculation**: `distortion` monitors clustering progress by calculating squared error differences.
- **Classifier Training**: `MLPClassifier` is trained on a training subset of Iris data to evaluate on unseen test data.

### Example Execution
1. **K-means Clustering**:
   - Clusters petal length and width dimensions and visualizes distortion values during iterations.
   - Plots final clusters and centroids.

2. **Binary Logistic Regression**:
   - Uses logistic regression to separate `versicolor` and `virginica`.
   - Plots decision boundaries and evaluates MSE.

3. **Neural Network Classification**:
   - Trains an MLPClassifier using the complete feature set to predict all three Iris species.
   - Displays the accuracy of the model and examples of actual vs. predicted classes.
   - Visualizes classification results using a confusion matrix heatmap.

### Usage
1. Clone this repository.
2. Run `main.py` to observe clustering and neural network classifications.
3. The `irisdata.csv` file must be in the same directory to load the dataset.

### Dependencies
- **Python** 3.x
- **Numpy** and **Pandas** for data handling and matrix operations.
- **Matplotlib** and **Seaborn** for plotting.
- **Scikit-Learn** for the MLPClassifier.

## Example Output
- Accuracy on the test set from the MLPClassifier.
- Distortion values during K-means clustering.
- Scatter plots showing clusters and decision boundaries.
- Confusion matrix heatmap displaying true vs. predicted species labels.

### Additional Notes
- Adjust the neural network parameters in `MLPClassifier` to experiment with hidden layers or epochs.
- The `irisdata.csv` file should contain the complete Iris dataset with the four feature columns and species labels.

## Future Enhancements
- Implement PCA to reduce dimensions before clustering.
- Integrate cross-validation for improved model evaluation.
- Extend binary logistic regression to multi-class classification using softmax.

