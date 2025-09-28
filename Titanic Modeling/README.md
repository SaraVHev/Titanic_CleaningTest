# Titanic Survival Prediction – Modeling

## Project Overview
This notebook (`Titanic_Modeling.ipynb`) continues from the Titanic Cleaning project. It uses the cleaned dataset to predict passenger survival and compare model performance. The main focus is on:

- Preparing the features for modeling
- Training and evaluating machine learning models
- Comparing metrics and visualizing results

## Steps Performed
1. **Load Cleaned Data**  
   The cleaned dataset (`titanic_cleaned.csv`) is loaded from the same folder as this notebook.

2. **Feature Scaling**  
   Numeric features are standardized using `StandardScaler` to improve model performance.

3. **Train-Test Split**  
   Data is split into training (80%) and test (20%) sets using `train_test_split`.

4. **Modeling**  
   - **Random Forest Classifier**  
   - **Logistic Regression** (with `max_iter` increased to ensure convergence)

5. **Evaluation**  
   Metrics calculated for both models: Accuracy, Precision, Recall, F1-Score.  
   Additional analysis includes:
   - Feature importance (Random Forest)  
   - ROC curve comparison (both models)  
   - Confusion matrices for both models

## Results
The notebook produces:

- A table comparing performance metrics for both models
- Bar chart of feature importances (Random Forest)
- ROC curve comparison
- Confusion matrices for Random Forest and Logistic Regression

## How to Run
1. Make sure `titanic_cleaned.csv` is in the same folder as the notebook.  
2. Open `Titanic_Modeling.ipynb` in Jupyter Notebook.  
3. Run all cells to reproduce the results and visualizations.  

## Folder Structure
Titanic_Cleaning/ <- Notebook and scripts for data cleaning
Titanic_Modeling/ <- Notebook and scripts for modeling
├─ Titanic_Modeling.ipynb
├─ titanic_cleaned.csv <- Cleaned dataset used for modeling
└─ README.md <- This file


## Dependencies
- Python 3.13.4
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn