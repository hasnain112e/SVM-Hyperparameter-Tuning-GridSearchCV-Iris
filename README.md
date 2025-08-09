# SVM-Hyperparameter-Tuning-GridSearchCV-Iris
# 🔍 Hyperparameter Tuning for SVM with GridSearchCV - Iris Dataset

## 📌 Project Overview
This project demonstrates how to use **GridSearchCV** to systematically find the best hyperparameters for a **Support Vector Machine (SVM)** classifier.  
The **Iris dataset** is used, and the model is optimized by tuning critical parameters like `C`, `gamma`, and `kernel`.

## 📊 Dataset
- **Source:** `load_iris()` from scikit-learn
- **Samples:** 150
- **Features:** Sepal length, sepal width, petal length, petal width
- **Target Classes:** Setosa, Versicolor, Virginica

## 🚀 Steps Performed
1. Loaded and explored the Iris dataset
2. Visualized data distribution with seaborn.pairplot
3. Performed 80-20 train-test split (`random_state=42`)
4. Defined hyperparameter grid:
   - `C`: [0.1, 1, 10, 100]
   - `gamma`: [1, 0.1, 0.01, 0.001]
   - `kernel`: ['rbf']
5. Applied **GridSearchCV** with:
   - 5-fold cross-validation
   - `n_jobs=-1` for parallel execution
6. Found the best parameters and best cross-validation score
7. Evaluated model on the test set:
   - Accuracy score
   - Classification report
   - Confusion matrix visualization

## 🛠 Technologies Used
- Python
- scikit-learn
- Pandas
- Matplotlib
- Seaborn

## 📈 Output
- **Best Parameters:** Example: `{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}`
- **Best CV Score:** ~XX%
- **Test Accuracy:** ~XX%
- Confusion matrix heatmap
- Pairplot visualization

## 🔧 How to Run
```bash
# Clone the repository
git clone https://github.com/yourusername/SVM-Hyperparameter-Tuning-GridSearchCV-Iris.git
cd SVM-Hyperparameter-Tuning-GridSearchCV-Iris

# Install dependencies
pip install scikit-learn pandas matplotlib seaborn

# Run the notebook
jupyter notebook svm_gridsearchcv_iris.ipynb
