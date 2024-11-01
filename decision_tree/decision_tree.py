import numpy as np
import pandas as pd
from collections import Counter

def entropy(y):
    """ Entropy is a measure of the uncertainty or randomness in a set of labels. 
    Used to evaluate the distribution of classes in the data."""
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def split_data(X, y, feature_idx, threshold):
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def best_split(X, y):
    best_gain = 0
    best_split = None
    current_entropy = entropy(y)
    n_features = X.shape[1]

    for feature_idx in range(n_features):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_data(X, y, feature_idx, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            left_entropy = entropy(y_left)
            right_entropy = entropy(y_right)
            n_left, n_right = len(y_left), len(y_right)
            weighted_entropy = (n_left * left_entropy + n_right * right_entropy) / len(y)

            gain = current_entropy - weighted_entropy
            if gain > best_gain:
                best_gain = gain
                best_split = {"feature_idx": feature_idx, "threshold": threshold}

    return best_gain, best_split

class DecisionTreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(X, y, depth=0, max_depth=5):
    n_samples, n_features = X.shape
    n_labels = len(np.unique(y))

    if depth >= max_depth or n_labels == 1:
        leaf_value = Counter(y).most_common(1)[0][0]
        return DecisionTreeNode(value=leaf_value)
    
    gain, split = best_split(X, y)
    if gain == 0:
        leaf_value = Counter(y).most_common(1)[0][0]
        return DecisionTreeNode(value=leaf_value)
    
    # Split the data andd create child nodes
    X_left, y_left, X_right, y_right = split_data(X, y, split["feature_idx"], split["threshold"])
    left_child = build_tree(X_left, y_left, depth + 1, max_depth)
    right_child = build_tree(X_right, y_right, depth + 1, max_depth)
    return DecisionTreeNode(split["feature_idx"], split["threshold"], left_child, right_child)

def predict_sample(node, sample):
    if node.value is not None:
        return node.value
    if sample[node.feature_idx] <= node.threshold:
        return predict_sample(node.left, sample)
    else:
        return predict_sample(node.right, sample)

def predict(tree, X):
    return [predict_sample(tree, sample) for sample in X]

from sklearn.metrics import confusion_matrix
# def evaluate_metrics(y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred)
#     tp = cm[1, 1] if cm.shape == (2, 2) else 0 # True Positive
#     tn = cm[0, 0] if cm.shape == (2, 2) else 0 # True Negative
#     fp = cm[0, 1] if cm.shape == (2, 2) else 0 # False Positive   
#     fn = cm[1, 0] if cm.shape == (2, 2) else 0 # False Negative

#     accuracy = (tp + tn) / (tp + tn + fp + fn)
#     precision = tp / (tp + fp) if (tp + fp) != 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) != 0 else 0
#     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
#     return accuracy, precision, recall, f1_score

def evaluate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    
    # Đối với bài toán đa lớp, tính precision, recall, f1-score cho mỗi lớp và trung bình cộng kết quả
    precision_per_class = []
    recall_per_class = []
    f1_score_per_class = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_score_per_class.append(f1_score)

    # Tính độ chính xác tổng thể
    accuracy = np.trace(cm) / np.sum(cm)
    
    # Trung bình macro cho precision, recall, f1-score
    precision = np.mean(precision_per_class)
    recall = np.mean(recall_per_class)
    f1_score = np.mean(f1_score_per_class)
    
    return accuracy, precision, recall, f1_score

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    tree = build_tree(X_train, y_train)
    predictions = predict(tree, X_test)

    accuracy, precision, recall, f1_score = evaluate_metrics(y_test, predictions)

    with open("decision_tree\metrics.txt", "w") as file:
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Precision: {precision:.2f}\n")
        file.write(f"Recall: {recall:.2f}\n")
        file.write(f"F1-Score: {f1_score:.2f}\n")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")
