
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Wine dataset 
wine_data = datasets.load_wine()
X = wine_data.data
y = wine_data.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Determine class distribution in the training set
unique_classes, class_counts = np.unique(y_train, return_counts=True)
class_probabilities = class_counts / len(y_train)

y_pred = np.random.choice(unique_classes, size=len(y_test), p=class_probabilities)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# print("\nConfusion Matrix:")
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix for Baseline')
# plt.show()
