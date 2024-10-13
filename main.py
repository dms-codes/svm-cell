import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools

# Constants
CELL_SAMPLES_FILE = "cell_samples.csv"

def load_data(file_path):
    """
    Load the dataset and display basic information.
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    df = pd.read_csv(file_path)
    print("\nDataset Information:")
    df.info()
    print("\nFirst 5 Rows of the Dataset:")
    print(df.head())
    print("\nStatistical Summary:")
    print(df.describe())
    return df

def clean_data(df):
    """
    Clean the dataset by ensuring numeric values in relevant columns.
    :param df: DataFrame with the raw data
    :return: Cleaned DataFrame
    """
    # Filter out rows with non-numeric values in specific columns
    numeric_columns = ['Clump', 'UnifSize', 'BareNuc']
    for col in numeric_columns:
        df = df[df[col].apply(lambda x: str(x).isdigit())]
    
    # Convert the filtered columns to numeric type
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    return df

def label_classes(df):
    """
    Convert numerical class labels (2 and 4) to descriptive labels (Benign and Malignant).
    :param df: DataFrame with class labels
    :return: Updated DataFrame with labeled classes
    """
    df['Class'] = df['Class'].replace({2: 'Benign', 4: 'Malignant'})
    return df

def plot_distribution(df):
    """
    Visualize the distribution of 'Clump' vs 'UnifSize', colored by 'Class'.
    :param df: Cleaned DataFrame
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Clump', y='UnifSize', hue='Class',
                    palette={'Benign': 'blue', 'Malignant': 'red'}, 
                    alpha=0.7, s=100)
    plt.title("Distribution of Clump Thickness vs Uniformity of Cell Size")
    plt.xlabel("Clump Thickness")
    plt.ylabel("Uniformity of Cell Size")
    plt.legend(title="Class")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def split_data(df):
    """
    Split the dataset into training and test sets.
    :param df: Cleaned DataFrame
    :return: Train and test sets
    """
    features = df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 
                   'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
    X = np.asarray(features)
    y = df['Class'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    print(f'Train set: {X_train.shape}, {y_train.shape}')
    print(f'Test set: {X_test.shape}, {y_test.shape}')
    return X_train, X_test, y_train, y_test

def train_svm_model(X_train, y_train):
    """
    Train an SVM model with the RBF kernel.
    :param X_train: Training features
    :param y_train: Training labels
    :return: Trained SVM model
    """
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    """
    Evaluate the trained SVM model and display the classification report.
    :param clf: Trained model
    :param X_test: Test features
    :param y_test: True test labels
    :return: Predicted labels
    """
    yhat = clf.predict(X_test)
    print(classification_report(y_test, yhat))
    return yhat

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot the confusion matrix.
    :param cm: Confusion matrix
    :param classes: List of class names
    :param normalize: Whether to normalize values
    :param title: Title of the plot
    :param cmap: Color map
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def main():
    # Load, clean, and prepare data
    df = load_data(CELL_SAMPLES_FILE)
    df = clean_data(df)
    df = label_classes(df)
    
    # Visualize data distribution
    plot_distribution(df)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train the SVM model
    clf = train_svm_model(X_train, y_train)
    
    # Evaluate the model
    yhat = evaluate_model(clf, X_test, y_test)
    
    # Compute and plot confusion matrix
    cm = confusion_matrix(y_test, yhat, labels=['Benign', 'Malignant'])
    plot_confusion_matrix(cm, classes=['Benign', 'Malignant'], normalize=False)

if __name__ == '__main__':
    main()
