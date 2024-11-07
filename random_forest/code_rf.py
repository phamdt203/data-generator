import numpy as np
from collections import Counter
from decision_tree.code_dt import DecisionTreeNode
import random
from typing import Any, List
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def gini_impurity(y: np.ndarray)->float:

    counts = Counter(y)
    impurity = 1 - sum((count / len(y)) ** 2 for count in counts.values())
    return impurity

def split_data(X: np.ndarray, y: np.ndarray, feature_idx: int, threshold: float)->tuple:
    """
    Split data based on a feature threshold.

    Args:
        X (np.ndarray): Input data.
        y (np.ndarray): Data labels.
        feature_idx (int): Index of the feature for splitting.
        threshold (float): Splitting threhold.

    Returns:
        tuple: (X_left, y_left, X_right, y_right) data and labels after splitting.
    """
    left_mask = X[:, feature_idx] <= threshold
    right_mask =X[:, feature_idx] > threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def build_tree(X: np.ndarray, y: np.ndarray, depth: int = 0, max_depth: int = 5)-> DecisionTreeNode:
    """
    Build a decision tree for the input data.

    Args:
        X (np.ndarray): Input data.
        y (np.ndarray): Data labels.
        depth (int): Current depth of the tree.
        max_depth (int): Maximum depth of the tree.

    Returns:
        DecisionTreeNode: Root node of the decision tree.
    """
    n_samples, n_features = X.shape
    n_labels = len(set(y))

    # Stopping condition
    if depth >= max_depth or n_labels == 1 or n_samples < 2:
        leaf_value = Counter(y).most_common(1)[0][0]
        return DecisionTreeNode(value=leaf_value)

    # Find the best split
    best_feature, best_threshold = None, None
    min_impurity = float("inf")
    
    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_data(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            # Calculate impurity after splitting
            impurity = (len(y_left) / len(y)) * gini_impurity(y_left) + (len(y_right) / len(y)) * gini_impurity(y_right)
            if impurity < min_impurity:
                min_impurity = impurity
                best_feature = feature_index
                best_threshold = threshold

    # Split data
    X_left, y_left, X_right, y_right = split_data(X, y, best_feature, best_threshold)
    left_child = build_tree(X_left, y_left, depth + 1, max_depth)
    right_child = build_tree(X_right, y_right, depth + 1, max_depth)
    return DecisionTreeNode(feature_idx=best_feature, threshold=best_threshold, left=left_child, right=right_child)

def predict_tree(node: DecisionTreeNode, X: np.ndarray)->Any:
    """
    Predict the label of a single input sample using the decision tree.

    Args:
        node (DecisionTreeNode): Root node of the decision tree.
        X (np.ndarray): Input sample

    Returns:
        any: Predicted label of the sample
    """
    if node.is_leaf_node():
        return node.value
    if X[node.feature_idx] <= node.threshold:
        return predict_tree(node.left, X)
    return predict_tree(node.right, X)

class RandomForest:
    """
    Class representing a Random Forest model.

    Attributes:
        n_trees (int): Number of trees in the forest.
        max_depth (int): Maximum depth of each tree.
        trees (list): List of decision trees in the forest.
    """
    def __init__(self, n_trees: int = 10, max_depth: int = 5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees: List[DecisionTreeNode] = []

    def bootstrap_sample(self, X: np.ndarray, y: np.ndarray)->tuple:
        """
        Generate a bootstrap sample from the datase.

        Args:
            X (np.ndarray): Input data
            y (np.ndarray): Data labels

        Returns:
            tuple: (X_sample, y_sample) bootstrapped data and labels.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the Random Forest on the input data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Data labels.
        """
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = build_tree(X_sample, y_sample, max_depth=self.max_depth)
            self.trees.append(tree)

    def predict(self, X: np.ndarray, y: np.ndarray):

        tree_preds = np.array([predict_tree(tree, x) for tree in self.trees for x in X])
        tree_preds = tree_preds.reshape(self.n_trees, len(X))
        y_pred = [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(len(X))]
        return np.array(y_pred)
    
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
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    forest = RandomForest(n_trees=10, max_depth=5)
    forest.fit(X_train, y_train)

    y_pred = forest.predict(X_test, y_test)

    accuracy, precision, recall, f1_score = evaluate_metrics(y_test, y_pred)

    with open(r"test\data-generator\random_forest\random_forest_metrics.txt", "w") as file:
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Precision: {precision:.2f}\n")
        file.write(f"Recall: {recall:.2f}\n")
        file.write(f"F1-Score: {f1_score:.2f}\n")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")
