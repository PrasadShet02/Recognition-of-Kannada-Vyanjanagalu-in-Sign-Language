import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score,precision_recall_curve,auc,average_precision_score,precision_recall_fscore_support,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from grucnn import model

DATA_PATH = r"FEATURES" # Path where Features extracted are stored
CLASSES_LIST = os.listdir(DATA_PATH)
print(CLASSES_LIST)

# Load and preprocess the keypoints data
features = []
labels = []

class_accuracies = {}  # Dictionary to store accuracies for each sign

for label_id, sign_name in enumerate(CLASSES_LIST):
    sign_path = os.path.join(DATA_PATH, sign_name)
    np_data = np.load(os.path.join(sign_path, sign_name + ".npy"))
    print("Extracting", sign_name)
    features.extend(np_data)
    samples = len(os.listdir(os.path.join(r"DATASET", sign_name)))  # Set number of samples per sign. "Dataset" is path where Dataset is stored
    print(samples)
    labels.extend([label_id] * samples)

    # Store the number of samples for each class
    class_accuracies[sign_name] = samples

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(to_categorical(labels).astype(int))

# Training
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True)

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=125)  # Adjust the number of epochs as needed

res = model.predict(X_test)
# Generate confusion matrix
y_pred = np.argmax(res, axis=1)
y_true = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred)

# Calculate accuracy for each sign
accuracies = {}
for i, sign_name in enumerate(CLASSES_LIST):
    sign_accuracy = accuracy_score(y_true[i::len(CLASSES_LIST)], y_pred[i::len(CLASSES_LIST)])
    accuracies[sign_name] = sign_accuracy

# Calculate average accuracy
average_accuracy = sum(accuracies.values()) / len(accuracies)


# Print accuracies for each sign
for sign_name, accuracy in accuracies.items():
    print(f'Accuracy for {sign_name}: {accuracy:.2%}')

# Generate confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')
plt.show()

#Mean average precision
average_precisions = {}
for i, sign_name in enumerate(CLASSES_LIST):
    precision, recall, _ = precision_recall_curve(y_test[:, i], res[:, i])
    average_precisions[sign_name] = auc(recall, precision)

# Calculate mean average precision
mean_average_precision = sum(average_precisions.values()) / len(average_precisions)
# Print average precision for each sign
for sign_name, ap in average_precisions.items():
    print(f'Average Precision for {sign_name}: {ap:.2%}')

# Print mean average precision
print(f'Mean Average Precision: {mean_average_precision:.2%}')

# Generate bar plot for average precision
plt.figure(figsize=(10, 6))
plt.bar(average_precisions.keys(), average_precisions.values())
plt.xlabel('Signs')
plt.ylabel('Average Precision')
plt.title('Average Precision for Each Sign')
plt.savefig('average_precision.png')
plt.show()

# Calculate precision, recall, and F1-score for each sign
precisions, recalls, f1_scores, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

# Calculate average precision, recall, and F1-score
average_precision = sum(precisions) / len(precisions)
average_recall = sum(recalls) / len(recalls)
average_f1_score = sum(f1_scores) / len(f1_scores)

# # Print precision, recall, and F1-score for each sign
# for i, sign_name in enumerate(CLASSES_LIST):
#     print(f"Metrics for {sign_name}:")
#     print(f"Precision: {precisions[i]:.2%}")
#     print(f"Recall: {recalls[i]:.2%}")
#     print(f"F1-Score: {f1_scores[i]:.2%}")

print("-----------------------------")

# Print average ,Accuracy,precision, recall, and F1-score
print("Average Metrics:")
print(f'Average Accuracy: {average_accuracy:.2%}')
print(f"Average Precision: {average_precision:.2%}")
print(f"Average Recall: {average_recall:.2%}")
print(f"Average F1-Score: {average_f1_score:.2%}")

# Print classification report
classification_rep = classification_report(y_true, y_pred, target_names=CLASSES_LIST)
print("\nClassification Report:\n", classification_rep)

# Save the model
model.save('gruCNN_model.keras')
model.summary()
