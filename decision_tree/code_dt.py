import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def entropy(y: np.ndarray)->float:
    """
    Calculates the entropy of a label distribution.

    Args:
        y (array): Array of labels.

    Returns:
        float: Entropy of the label distribution.
    """
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def split_data(X: np.ndarray, y: np.ndarray, feature_idx: int, threshold: float)->tuple:
    """
    Splits the dataset based on a feature and a threshold.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels array.
        feature_idx (int): Index of the feature to split on.
        threshold (float): Threshold value to split the feature on.

    Returns:
        tuple: Split datasets (X_left, y_left, x_right, y_right).
    """
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def best_split(X: np.ndarray, y: np.ndarray)->tuple:
    """
    Find the best feature and threshold to split the data to maximize information gain.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels array.

    Returns:
        tuple: Best information gain and the best split (feature index and threshold).
    """
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
    """
    A class representing a node in a decision tree.

    Attributes:
        feature_idx (int): Index of the feature to split on.
        threshold (float): Threshold value to split the feature.
        left (DecisionTreeNode): Left child node.
        right (DecisionTreeNode): Right child node.
        value (int): Class label if it is a leaf node.
    """
    def __init__(self, feature_idx: int=None, threshold: float=None, left: 'DecisionTreeNode'=None, right: 'DecisionTreeNode'=None, *, value: int=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Check if the node is a leaf node.

        Returns:
            bool: True if the node is a leaf node, otherwise False.
        """
        return self.value is not None

def build_tree(X: np.ndarray, y: np.ndarray, depth=0, max_depth=5)->'DecisionTreeNode':
    """
    Builds a decision tree using recursive splitting.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels array.
        depth (int, optional): Current depth of the tree. Defaults to 0.
        max_depth (int, optional): Maximum depth of the tree. Defaults to 5.

    Returns:
        DecisionTreeNode: Root node of the decision tree.
    """
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

def predict_sample(node: 'DecisionTreeNode', sample: np.ndarray)->int:
    """
    Predicts the class label for a single sample by traversing the decision tree.

    Args:
        node (DecisionTreeNode): Root node of the decision tree.
        sample (np.ndarray): A single data sample.

    Returns:
        int: Predicted class label.
    """
    if node.value is not None:
        return node.value
    if sample[node.feature_idx] <= node.threshold:
        return predict_sample(node.left, sample)
    else:
        return predict_sample(node.right, sample)

def predict(tree: 'DecisionTreeNode', X: np.ndarray)->list:
    """
    Predicts class labels for multiple samples using a trained decision tree.

    Args:
        tree (DecisionTreeNode): Root node of the decision tree.
        X (np.ndarray): Feature matrix of samples.

    Returns:
        list: Predicted class labels for all samples.
    """
    return [predict_sample(tree, sample) for sample in X]

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray)->tuple:
    """
    Evaluates the model performance using confusion matrix and computes accuracy, precision, recall and F1 score.

    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.

    Returns:
        tuple: Accuracy, precision, recall and F1 score of the model.
    """
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    
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

    accuracy = np.trace(cm) / np.sum(cm)
    
    precision = np.mean(precision_per_class)
    recall = np.mean(recall_per_class)
    f1_score = np.mean(f1_score_per_class)
    
    return accuracy, precision, recall, f1_score

if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    tree = build_tree(X_train, y_train)
    predictions = predict(tree, X_test)

    accuracy, precision, recall, f1_score = evaluate_metrics(y_test, predictions)

    with open("test\data-generator\decision_tree\decision_tree_metrics.txt", "w") as file:
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Precision: {precision:.2f}\n")
        file.write(f"Recall: {recall:.2f}\n")
        file.write(f"F1-Score: {f1_score:.2f}\n")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")
