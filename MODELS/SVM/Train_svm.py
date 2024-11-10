import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve, auc, precision_recall_fscore_support, f1_score, average_precision_score,classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import seaborn as sns
from model_def import svm_model
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
import joblib

# Suppress Matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = r"FEATURES" # Path where Features extracted are stored
CLASSES_LIST = os.listdir(DATA_PATH)
print(CLASSES_LIST)

# Load and preprocess the keypoints data
features = []
labels = []

for label_id, sign_name in enumerate(CLASSES_LIST):
    sign_path = os.path.join(DATA_PATH, sign_name)
    np_data = np.load(os.path.join(sign_path, sign_name + ".npy"))
    print("Extracting", sign_name)
    features.extend(np_data)
    samples = len(os.listdir(os.path.join(r"DATASET", sign_name)))  # Set number of samples per sign. "Dataset" is path where Dataset is stored
    print(samples)
    labels.extend([label_id] * samples)

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Flatten the features
features = features.reshape(features.shape[0], -1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True)

param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly','sigmoid'],  # Kernel type
    'degree': [1,2, 3, 4,5,6]  # Degree of the polynomial kernel (only for poly kernel)
}

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(svm_model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the SVM model with the best hyperparameters
best_svm_model = SVC(**best_params)

best_svm_model.fit(X_train, y_train)

# Predict using the trained model
y_pred = best_svm_model.predict(X_test)
res = best_svm_model.decision_function(X_test)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate overall accuracy using confusion matrix
overall_accuracy = (conf_matrix.diagonal().sum()) / conf_matrix.sum()
print(f"Overall Accuracy (from confusion matrix): {overall_accuracy * 100:.2f}%")

# Calculate and print accuracy for each sign
sign_accuracies = []
for i in range(len(CLASSES_LIST)):
    sign_accuracy = conf_matrix[i, i] / np.sum(conf_matrix[i, :])
    sign_accuracies.append(sign_accuracy)
    print(f"Accuracy for {CLASSES_LIST[i]}: {sign_accuracy * 100:.2f}%")

# Calculate mean average precision
mean_average_precision = average_precision_score(y_test, res, average='micro')
print(f'Mean Average Precision: {mean_average_precision * 100:.2f}%')

# Calculate average precision for each sign
average_precisions = {}
for i, sign_name in enumerate(CLASSES_LIST):
    precision, recall, _ = precision_recall_curve(y_test == i, res[:, i])
    average_precisions[sign_name] = auc(recall, precision)

# Calculate and print average precision for each sign
for sign_name, ap in average_precisions.items():
    print(f'Average Precision for {sign_name}: {ap * 100:.2f}%')

# Calculate precision, recall, and F1-score for each sign
precisions, recalls, f1_scores, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

# Calculate average precision, recall, and F1-score
average_precision = sum(precisions) / len(precisions)
average_recall = sum(recalls) / len(recalls)
average_f1_score = f1_score(y_test, y_pred, average='weighted')

# Print precision, recall, and F1-score for each sign
for i, sign_name in enumerate(CLASSES_LIST):
    print(f"Metrics for {sign_name}:")
    print(f"Precision: {precisions[i] * 100:.2f}%")
    print(f"Recall: {recalls[i] * 100:.2f}%")
    print(f"F1-Score: {f1_scores[i] * 100:.2f}%")
    print("-----------------------------")

# Print average ,Accuracy,precision, recall, and F1-score
print("\nAverage Metrics:")
print(f'Average Accuracy: {overall_accuracy * 100:.2f}%')
# print(f"Mean Average Precision: {mean_average_precision * 100:.2f}%")
print(f"Average Precision: {average_precision * 100:.2f}%")
print(f"Average Recall: {average_recall * 100:.2f}%")
print(f"Average F1-Score: {average_f1_score * 100:.2f}%")

# # Predict using the trained model
# y_pred = best_svm_model.predict(X_test)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')
plt.show()

mean_average_precision = average_precision_score(label_binarize(y_test, classes=list(range(len(CLASSES_LIST)))), res, average='micro')
print(f'Mean Average Precision: {mean_average_precision * 100:.2f}%')

# Save precision-recall curve for mean average precision
precision, recall, _ = precision_recall_curve(label_binarize(y_test, classes=list(range(len(CLASSES_LIST)))).ravel(), res.ravel())
plt.figure(figsize=(10, 6))
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('mean_average_precision.png')
plt.show()

# # Calculate and print accuracy for each sign
# sign_accuracies = []
# for i in range(len(CLASSES_LIST)):
#     sign_accuracy = conf_matrix[i, i] / np.sum(conf_matrix[i, :])
#     sign_accuracies.append(sign_accuracy)
#     print(f"Accuracy for {CLASSES_LIST[i]}: {sign_accuracy * 100:.2f}%")

# # Calculate overall accuracy
# overall_accuracy = (conf_matrix.diagonal().sum()) / conf_matrix.sum()
# print(f"Overall Accuracy (from confusion matrix): {overall_accuracy * 100:.2f}%")
# Print classification report

classification_rep = classification_report(y_test, y_pred, target_names=CLASSES_LIST)
print("\nClassification Report:\n", classification_rep)

# Save the SVM model
joblib.dump(best_svm_model, 'svm_model.pkl')

