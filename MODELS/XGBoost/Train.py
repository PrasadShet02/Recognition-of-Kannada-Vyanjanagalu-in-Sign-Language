import numpy as np
import os
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve, auc, precision_recall_fscore_support, f1_score, average_precision_score,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score

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
# Reshape the features to be 2D
features = features.reshape(features.shape[0], -1)

# Training
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, shuffle=True)

# Create an XGBoost classifier
clf = xgb.XGBClassifier(
    n_estimators=100,  # Number of boosting rounds (you can adjust this)
    max_depth=3,  # Maximum depth of each tree (you can adjust this)
    learning_rate=0.1,  # Step size shrinkage to prevent overfitting
    objective='multi:softmax',  # For multiclass classification
    num_class=len(CLASSES_LIST)  # Number of classes in your dataset
)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix_xgboost.png')
plt.show()

# Calculate and print accuracy for each sign
sign_accuracies = []
for i in range(len(CLASSES_LIST)):
    sign_accuracy = conf_matrix[i, i] / np.sum(conf_matrix[i, :])
    sign_accuracies.append(sign_accuracy)
    print(f"Accuracy for {CLASSES_LIST[i]}: {sign_accuracy * 100:.2f}%")

# Plot mean average precision for each class
class_labels = np.array(CLASSES_LIST)
mean_avg_precision = np.zeros(len(CLASSES_LIST))
y_prob = clf.predict_proba(X_test)
for i in range(len(CLASSES_LIST)):
    mean_avg_precision[i] = average_precision_score(label_binarize(y_test, classes=[i]), y_prob[:, i])

plt.figure(figsize=(10, 6))
plt.bar(class_labels, mean_avg_precision, color='darkblue')
plt.xlabel('Class Labels')
plt.ylabel('Mean Average Precision')
plt.title('Mean Average Precision for Each Class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('mean_average_precision_xgboost.png')
plt.show()

# Calculate overall accuracy
overall_accuracy = accuracy_score(y_test, y_pred)


# Calculate mean average precision
y_prob = clf.predict_proba(X_test)
mean_average_precision = average_precision_score(label_binarize(y_test, classes=list(range(len(CLASSES_LIST)))), y_prob, average='micro')
print(f'Mean Average Precision: {mean_average_precision * 100:.2f}%')

# Save precision-recall curve for mean average precision
precision, recall, _ = precision_recall_curve(label_binarize(y_test, classes=list(range(len(CLASSES_LIST)))).ravel(), y_prob.ravel())

# Calculate and print average of recall, precision, and F1-score
average_recall = sum(recall)/len(recall)
average_precision = sum(precision)/len(precision)
average_f1_score = np.mean(f1_score(y_test, y_pred, average=None))

print("----------------------------------------")

print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
print(f"Average Recall: {average_recall * 100:.2f}%")
print(f"Average Precision: {average_precision * 100:.2f}%")
print(f"Average F1-Score: {average_f1_score * 100:.2f}%")


classification_rep = classification_report(y_test, y_pred, target_names=CLASSES_LIST)
print("\nClassification Report:\n", classification_rep)

# Save the XGBoost model
joblib.dump(clf, 'XGboost_model.pkl')
