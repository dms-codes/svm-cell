
# Cell Samples Classification using SVM

This project implements a machine learning pipeline to classify cell samples as either *Benign* or *Malignant* based on features such as Clump Thickness, Uniformity of Cell Size, and Bare Nuclei. The classification is performed using a **Support Vector Machine (SVM)** model with an RBF kernel. The project uses Python and common data science libraries such as **Pandas**, **Seaborn**, **Matplotlib**, and **Scikit-learn**.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Features](#features)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Installation

To run the project, ensure you have Python installed (preferably version 3.7 or above). You can install the required libraries using:

```bash
pip install -r requirements.txt
```

### Required Libraries
- `pandas`
- `matplotlib`
- `seaborn`
- `numpy`
- `scikit-learn`

You can create a `requirements.txt` file with the following:

```txt
pandas
matplotlib
seaborn
numpy
scikit-learn
```

## Project Structure

The project includes the following files:
```
├── cell_samples.csv        # Dataset
├── script.py               # Main script with data processing, SVM training, and evaluation
├── README.md               # Project overview and instructions
└── requirements.txt        # Required libraries
```

## Dataset

The dataset `cell_samples.csv` contains several features for each sample:
- **Clump Thickness**
- **Uniformity of Cell Size**
- **Uniformity of Cell Shape**
- **Marginal Adhesion**
- **Single Epithelial Cell Size**
- **Bare Nuclei**
- **Bland Chromatin**
- **Normal Nucleoli**
- **Mitoses**

The target variable (`Class`) indicates whether the cell is:
- **2** - *Benign*
- **4** - *Malignant*

## Features

- **Data Cleaning**: Non-numeric values in important columns (e.g., `Bare Nuclei`) are removed, and the columns are converted to numeric types for analysis.
- **Data Visualization**: A scatter plot visualizes the distribution of the samples based on Clump Thickness and Uniformity of Cell Size, colored by class.
- **Train-Test Split**: The data is split into training and testing sets.
- **SVM Model**: A Support Vector Machine (SVM) classifier with an RBF kernel is trained on the features to predict the class (Benign or Malignant).
- **Model Evaluation**: The performance of the model is evaluated using a confusion matrix and a classification report.

## Usage

1. **Load Data**: The dataset is loaded and basic exploratory analysis is performed.
2. **Clean Data**: Non-numeric rows in key columns are removed.
3. **Label Classes**: The `Class` column is mapped to descriptive labels (`Benign` and `Malignant`).
4. **Visualize Data**: The scatter plot shows the distribution of the data based on key features.
5. **Train Model**: An SVM model is trained on the training set.
6. **Evaluate Model**: The model is evaluated using a confusion matrix and classification report.

To run the script, execute:

```bash
python main.py
```

### Example Output

```
Train set: (558, 9), (558,)
Test set: (140, 9), (140,)
              precision    recall  f1-score   support

       Benign       0.97      0.99      0.98        90
    Malignant       0.98      0.95      0.97        50

    accuracy                           0.98       140
   macro avg       0.98      0.97      0.98       140
weighted avg       0.98      0.98      0.98       140
```

The script will also display the confusion matrix as a plot.

## Results

- The SVM model achieves a high accuracy in predicting whether cell samples are benign or malignant.
- The confusion matrix and classification report show excellent precision, recall, and F1 scores.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```
