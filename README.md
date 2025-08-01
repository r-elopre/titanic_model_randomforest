# Titanic Survival Prediction with Random Forest

## Overview
This project uses a Random Forest Classifier to predict passenger survival on the Titanic based on a preprocessed dataset (`train_scaled.csv`). The dataset has been cleaned, encoded, and scaled, and the script trains a model, evaluates its performance, visualizes results, and saves predictions.

## Dataset
The dataset (`train_scaled.csv`) is located in the `data` directory and contains 891 passenger records with the following columns:
- **PassengerId**: Unique identifier for each passenger.
- **Survived**: Target variable (0 = Not Survived, 1 = Survived).
- **Pclass**: Passenger class (1, 2, or 3).
- **Age**: Scaled age of the passenger.
- **SibSp**: Scaled number of siblings/spouses aboard.
- **Parch**: Scaled number of parents/children aboard.
- **Fare**: Scaled ticket fare.
- **Sex_male**: Binary indicator (True = Male, False = Female).
- **Embarked_q**, **Embarked_s**: One-hot encoded embarkation ports (Q, S; C is the reference category).
- **Deck_B**, **Deck_C**, **Deck_D**, **Deck_E**, **Deck_F**, **Deck_G**, **Deck_T**, **Deck_Unknown**: One-hot encoded cabin deck information.

The dataset is preprocessed, with numerical features scaled and categorical features encoded.

## Script Description
The script (`titanic_model_randomforest.py`) is located in the `titanic_model_randomforest` directory and performs the following steps:
1. **Load Data**: Reads `train_scaled.csv` from `../data/train_scaled.csv` into a Pandas DataFrame.
2. **Set Target and Features**: Uses `Survived` as the target variable and all other columns except `PassengerId` as features.
3. **Train-Test Split**: Splits the data into 80% training and 20% testing sets (179 passengers in the test set).
4. **Model Training**: Fits a `RandomForestClassifier` with default parameters (random_state=42).
5. **Prediction**: Generates predictions on the test set.
6. **Evaluation**: Computes and prints:
   - Accuracy score.
   - Classification report (precision, recall, F1-score).
   - Confusion matrix.
7. **Visualization**: Plots the confusion matrix as a heatmap using Seaborn.
8. **Save Predictions**: Saves test set predictions to `titanic_rf_predictions.csv` with columns `PassengerId`, `Survived` (actual), and `Survived_Predicted`.

### Dependencies
- Python 3.x
- Libraries (install via `pip install pandas scikit-learn seaborn matplotlib`):
  - pandas
  - scikit-learn
  - seaborn
  - matplotlib

## Project Structure
```
titanic_model_randomforest/
├── data/
│   └── train_scaled.csv
├── titanic_model_randomforest/
│   ├── titanic_model_randomforest.py
│   ├── titanic_rf_predictions.csv (generated after running the script)
│   └── README.md
└── venv/ (virtual environment)
```

## How to Run
1. **Set Up Environment**:
   - Create and activate a virtual environment:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate  # Windows
     ```
   - Install dependencies:
     ```bash
     pip install pandas scikit-learn seaborn matplotlib
     ```

2. **Verify Dataset**:
   - Ensure `train_scaled.csv` is in the `data` directory (`C:\Users\ri\OneDrive\ai project\model\titanic_model_randomforest\data\`).

3. **Run the Script**:
   - Navigate to the script directory:
     ```bash
     cd C:\Users\ri\OneDrive\ai project\model\titanic_model_randomforest\titanic_model_randomforest
     ```
   - Execute the script:
     ```bash
     python titanic_model_randomforest.py
     ```

4. **Output**:
   - Console output includes accuracy, classification report, and confusion matrix.
   - A confusion matrix heatmap is displayed.
   - Predictions are saved to `titanic_rf_predictions.csv` in the script directory.

## Output Details
The script was run successfully, producing the following results:

### Console Output
```
Accuracy: 0.7932960893854749

Classification Report:
               precision recall f1-score support
           0       0.82      0.83      0.82       105
           1       0.75      0.74      0.75        74
    accuracy                           0.79       179
   macro avg       0.79      0.79      0.79       179
weighted avg       0.79      0.79      0.79       179

Confusion Matrix:
[[87 18]
 [19 55]]
```

- **Accuracy**: 79.33% (correctly predicted survival for ~79% of test set passengers).
- **Classification Report**:
  - **Class 0 (Not Survived)**: Precision = 0.82, Recall = 0.83, F1 = 0.82 (105 passengers).
  - **Class 1 (Survived)**: Precision = 0.75, Recall = 0.74, F1 = 0.75 (74 passengers).
- **Confusion Matrix**:
  - True Negatives (TN): 87 (correctly predicted "Not Survived").
  - False Positives (FP): 18 (incorrectly predicted "Survived").
  - False Negatives (FN): 19 (incorrectly predicted "Not Survived").
  - True Positives (TP): 55 (correctly predicted "Survived").

### Visualization
A heatmap of the confusion matrix was displayed, showing the counts of TN, FP, FN, and TP with labels "Not Survived" and "Survived".

### Saved File
- `titanic_rf_predictions.csv` contains 179 rows with:
  - `PassengerId`: Unique ID of test set passengers.
  - `Survived`: Actual survival status (0 or 1).
  - `Survived_Predicted`: Predicted survival status (0 or 1).

## Analysis
- The model performs reasonably well with ~79% accuracy, but there’s room for improvement, especially for Class 1 (Survived) where precision and recall are lower.
- The confusion matrix shows more errors in predicting survival (18 FP, 19 FN) than non-survival, possibly due to class imbalance (105 vs. 74 in the test set).
- The preprocessed dataset (scaled and encoded) likely contributed to the model’s performance.

## Potential Improvements
- **Hyperparameter Tuning**: Use `GridSearchCV` to optimize Random Forest parameters (e.g., `n_estimators`, `max_depth`).
- **Feature Engineering**: Analyze feature importance (`rf_model.feature_importances_`) to identify key predictors.
- **Cross-Validation**: Implement k-fold cross-validation for robust performance estimates.
- **Handle Imbalance**: Apply techniques like SMOTE or class weights to address the slight class imbalance.
- **Additional Metrics**: Plot ROC curves or compute AUC for deeper evaluation.

## Troubleshooting
- **FileNotFoundError**: Ensure `train_scaled.csv` is in `../data/`. Update the path in the script if the dataset is moved.
- **Library Issues**: Verify all dependencies are installed in the virtual environment.
- **Visualization Issues**: If the heatmap doesn’t display, check Matplotlib/Seaborn installation or backend settings.

## Author
This project was developed as part of an AI project to demonstrate machine learning with the Titanic dataset.