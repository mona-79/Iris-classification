# Iris Classification Project

## Overview

This project demonstrates the classification of the Iris dataset using a K-Nearest Neighbors (KNN) classifier. It includes data visualization, hyperparameter tuning, cross-validation, and performance evaluation.

## Dataset

The Iris dataset contains 150 samples of iris flowers, with each sample having four features: sepal length, sepal width, petal length, and petal width. The samples belong to one of three classes: setosa, versicolor, or virginica.

You can find the dataset on the UCI Machine Learning Repository: [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/Iris).

## Installation

To run the code, you need to have Python installed along with the following libraries:

- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:

```bash
pip install numpy matplotlib seaborn scikit-learn

##Code

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Visualize the dataset
sns.pairplot(sns.load_dataset('iris'), hue='species')
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 25)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X_train, y_train)

# Best parameters
print(f'Best parameters: {knn_cv.best_params_}')

# Train the K-Nearest Neighbors classifier with the best parameters
knn_best = KNeighborsClassifier(n_neighbors=knn_cv.best_params_['n_neighbors'])
knn_best.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_best.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Cross-validation
cv_scores = cross_val_score(knn_best, X, y, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean():.2f}')

# Predict class for a new sample
sample = [[5.0, 3.5, 1.6, 0.3]]
sample_scaled = scaler.transform(sample)
prediction = knn_best.predict(sample_scaled)
print(f'Predicted class for the sample: {iris.target_names[prediction][0]}')

##Usage
Load the Dataset: The Iris dataset is loaded and visualized to understand feature relationships.
Split and Standardize: The dataset is split into training and testing sets, and features are standardized.
Hyperparameter Tuning: The best n_neighbors value is found using GridSearchCV.
Model Training: A K-Nearest Neighbors classifier is trained with the best hyperparameters.
Evaluation: Model performance is evaluated using accuracy, classification report, and confusion matrix. Cross-validation scores are also computed.
Prediction: The classifier predicts the class for a new sample.

##Future Work
For further improvement, consider the following:

Explore other classification algorithms (e.g., SVM, Decision Trees).
Perform additional hyperparameter tuning.
Implement more advanced data visualization techniques.
Enhance model robustness with techniques like feature selection or engineering.

##License

This `README.md` provides detailed information about the project, including installation instructions, code explanation, usage steps, and suggestions for future work. It ensures that anyone reviewing or running the project can understand its components and objectives.

