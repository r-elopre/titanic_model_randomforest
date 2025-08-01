import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the train_scaled.csv into a Pandas DataFrame
df = pd.read_csv('../data/train_scaled.csv')  # Updated path to point to the data directory

# 2. Set the target column as Survived
y = df['Survived']

# 3. Use the rest of the columns as features, but exclude PassengerId
X = df.drop(columns=['PassengerId', 'Survived'])

# 4. Split the data into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Fit a RandomForestClassifier (with default params)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 6. Predict on the test set
y_pred = rf_model.predict(X_test)

# 7. Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8. Visualize the confusion matrix using seaborn.heatmap()
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 9. Save the predictions to titanic_rf_predictions.csv
predictions_df = pd.DataFrame({
    'PassengerId': df.loc[X_test.index, 'PassengerId'],
    'Survived': y_test,
    'Survived_Predicted': y_pred
})
predictions_df.to_csv('titanic_rf_predictions.csv', index=False)