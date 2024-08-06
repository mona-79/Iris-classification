# Iris Dataset Classification

This project demonstrates basic data analysis and classification on the Iris dataset using Logistic Regression and Decision Tree classifiers.

## Dataset

The Iris dataset is a classic dataset in machine learning, containing information about three different species of Iris flowers: Iris-setosa, Iris-versicolor, and Iris-virginica. The dataset includes 150 samples, with 50 samples for each species. Each sample has four features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.x installed
- Required Python libraries installed: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`

You can install the required libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. **Load the Dataset**

    ```python
    import pandas as pd
    df = pd.read_csv('Iris.csv')
    ```

2. **Data Preprocessing**

    - Drop the `Id` column
    - Display basic information about the dataset
    - Check for missing values
    - Plot histograms for each feature
    - Scatter plots for different feature pairs
    - Plot correlation heatmap
    - Label Encoding for the target variable

    ```python
    df = df.drop(columns=['Id'])
    print(df.head())
    print(df.describe())
    print(df.info())
    print(df['Species'].value_counts())
    print(df.isnull().sum())

    import matplotlib.pyplot as plt
    df['SepalLengthCm'].hist()
    plt.title('Sepal Length')
    plt.show()
    df['SepalWidthCm'].hist()
    plt.title('Sepal Width')
    plt.show()
    df['PetalLengthCm'].hist()
    plt.title('Petal Length')
    plt.show()
    df['PetalWidthCm'].hist()
    plt.title('Petal Width')
    plt.show()

    import seaborn as sns
    colors = ['red', 'orange', 'blue']
    species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']

    for i in range(3):
        x = df[df['Species'] == species[i]]
        plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c=colors[i], label=species[i])
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()
    plt.show()

    for i in range(3):
        x = df[df['Species'] == species[i]]
        plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c=colors[i], label=species[i])
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend()
    plt.show()

    for i in range(3):
        x = df[df['Species'] == species[i]]
        plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c=colors[i], label=species[i])
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')
    plt.legend()
    plt.show()

    for i in range(3):
        x = df[df['Species'] == species[i]]
        plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c=colors[i], label=species[i])
    plt.xlabel('Sepal Width')
    plt.ylabel('Petal Width')
    plt.legend()
    plt.show()

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')
    plt.show()

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['Species'] = le.fit_transform(df['Species'])
    print(df.head())
    ```

3. **Train/Test Split**

    Split the dataset into training and testing sets:

    ```python
    from sklearn.model_selection import train_test_split
    x = df.drop(columns=['Species'])
    y = df['Species']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    ```

4. **Logistic Regression**

    Train and evaluate a Logistic Regression model:

    ```python
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    print('Logistic Regression Accuracy:', accuracy)

    coefficients = model.coef_
    print('Coefficients:', coefficients)
    ```

5. **Decision Tree Classifier**

    Train and evaluate a Decision Tree Classifier model:

    ```python
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    print('Decision Tree Accuracy:', accuracy)
    ```

## Results

After running the code, you should see the accuracy of both Logistic Regression and Decision Tree models printed out, along with the coefficients of the Logistic Regression model.

## License

This project is licensed under the MIT License.
