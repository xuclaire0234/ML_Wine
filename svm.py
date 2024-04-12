
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
wine_data = datasets.load_wine()
X = wine_data.data
y = wine_data.target

# Split the dataset into training, validation, and testing sets
x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Define the SVM classifier and hyperparameters
svm = SVC(random_state=42)
params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto']
}

# Use GridSearchCV to find the best hyperparameters
grid = GridSearchCV(svm, param_grid=params, cv=5, scoring='accuracy')
grid.fit(x_train, y_train)

print(f"Best Hyperparameters: {grid.best_params_}")
print(f"Best Accuracy: {grid.best_score_:.2f}")

# Evaluate the best model on the validation set
y_val_pred = grid.predict(x_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_acc:.2f}")

best_svm = grid.best_estimator_
best_svm.fit(x_train, y_train)

# Evaluate the final model on the test set
y_test_pred = best_svm.predict(x_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_acc:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=wine_data.target_names))

# print("\nConfusion Matrix:")
# cm = confusion_matrix(y_test, y_test_pred)
# sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix for SVM Classifier')
# plt.show()
