import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, precision_recall_fscore_support, f1_score, average_precision_score, classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from model_def import rf_model
import warnings

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

rf_model.fit(X_train, y_train)

# Predict using the trained model
y_pred = rf_model.predict(X_test)
res = rf_model.predict_proba(X_test)

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

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')
plt.show()

# Calculate mean average precision
# mean_average_precision = sum(average_precisions.values()) / len(average_precisions)

# Plot bar graph for mean average precision
plt.figure(figsize=(10, 6))
plt.bar(average_precisions.keys(), average_precisions.values(), color='darkblue')
plt.xlabel('Signs')
plt.ylabel('Average Precision')
plt.title('Average Precision for Each Sign')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('mean_average_precision_bar.png')
plt.show()

# Calculate precision, recall, and F1-score for each sign
precisions, recalls, f1_scores, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

# Calculate average precision, recall, and F1-score
average_precision = sum(precisions) / len(precisions)
average_recall = sum(recalls) / len(recalls)
average_f1_score = f1_score(y_test, y_pred, average='weighted')

# Print precision, recall, and F1-score for each sign
# for i, sign_name in enumerate(CLASSES_LIST):
#     print(f"Metrics for {sign_name}:")
#     print(f"Precision: {precisions[i] * 100:.2f}%")
#     print(f"Recall: {recalls[i] * 100:.2f}%")
#     print(f"F1-Score: {f1_scores[i] * 100:.2f}%")
#     print("-----------------------------")

# Print average ,Accuracy,precision, recall, and F1-score
print("\nAverage Metrics:")
print(f'Average Accuracy: {overall_accuracy * 100:.2f}%')
print(f"Mean Average Precision: {mean_average_precision * 100:.2f}%")
print(f"Average Precision: {average_precision * 100:.2f}%")
print(f"Average Recall: {average_recall * 100:.2f}%")
print(f"Average F1-Score: {average_f1_score * 100:.2f}%")

# Predict using the trained model
# y_pred = rf_model.predict(X_test)
# Print classification report
classification_rep = classification_report(y_test, y_pred, target_names=CLASSES_LIST)
print("\nClassification Report:\n", classification_rep)


# mean_average_precision = average_precision_score(y_test, res, average='micro')
# print(f'Mean Average Precision: {mean_average_precision * 100:.2f}%')

# # Save precision-recall curve for mean average precision
# precision, recall, _ = precision_recall_curve(y_test.ravel(), res.ravel())
# plt.figure(figsize=(10, 8))
# plt.plot(recall, precision, color='darkorange', lw=2)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.savefig('precision_recall_curve.png')
# plt.show()

# # Calculate and print accuracy for each sign
# sign_accuracies = []
# for i in range(len(CLASSES_LIST)):
#     sign_accuracy = conf_matrix[i, i] / np.sum(conf_matrix[i, :])
#     sign_accuracies.append(sign_accuracy)
#     print(f"Accuracy for {CLASSES_LIST[i]}: {sign_accuracy * 100:.2f}%")

# # Calculate overall accuracy
# overall_accuracy = accuracy_score(y_test, y_pred)
# print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

# Save the SVM model
import joblib
joblib.dump(rf_model, 'rf_model.pkl')
