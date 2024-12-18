import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def split_data(data: pd.DataFrame, target: str, test_size: float=0.3, seed: int=42):
    """Split the data into training and testing sets.
        
    Parameters:
        - data: DataFrame
        - target: str, the name of the target column
        - test_size: float, default=0.3 corresponds to 30% of the data
        - seed: int, default=42

    Returns:
        - X_train, X_test, y_train, test_y: DataFrames
    """
    X = data.drop(columns=target)
    y = data[target]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=seed)
    return train_x, test_x, train_y, test_y

def sampled_range(mini, maxi, num):
    if not num:
        return []
    lmini = math.log(mini)
    lmaxi = math.log(maxi)
    ldelta = (lmaxi - lmini) / (num - 1)
    out = [x for x in set([int(math.exp(lmini + i * ldelta)) for i in range(num)])]
    out.sort()
    return out

def find_best_k(train_x, train_y):
    """Find the best k for the KNN model.

    Parameters:
        - train_x: DataFrame
        - train_y: DataFrame

    Returns:
        - best_k: int
    """
    best_k = 0
    best_score = 0
    candidate_k = sampled_range(1, 100, 10)
    for k in candidate_k:
        model = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_validation(train_x, train_y, model)
        if cv_scores['accuracy']['mean'] > best_score:
            best_score = cv_scores['accuracy']['mean']
            best_k = k
    return best_k
        

def create_classifier(model_type: str, n_neighbors: int=5):
    """Create a classifier based on the model type.
    
    Parameters:
        - model_type: str, the type of model to create
        - n_neighbors: int, default=5
    Returns:
        - model: the model object created. 
            Can be None if the model_type is not recognized.
    """
    if model_type == 'logistic':
        return LogisticRegression()
    elif model_type == 'decision_tree':
        return DecisionTreeClassifier()
    elif model_type == 'random_forest':
        return RandomForestClassifier()
    elif model_type == 'svm':
        return SVC()
    elif model_type == 'knn':
        return KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        return None

def eval_disease_classifier(test_x, test_y, classifier, scorings: list=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
    """Evaluate the disease classifier with different metrics.

    Parameters:
        - test_x: DataFrame, the test data
        - test_y: DataFrame, the test target
        - classifier: the classifier object
        - scorings: list, default=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    Returns:
        - error_rate: float
        - precision: float
        - recall: float
        - f1: float
        - auc: float
    """
    y_pred = classifier.predict(test_x)
    scores = {}
    for scoring in scorings:
        if scoring == 'accuracy':
            scores[scoring] = accuracy_score(test_y, y_pred)
        elif scoring == 'precision':
            scores[scoring] = precision_score(test_y, y_pred)
        elif scoring == 'recall':
            scores[scoring] = recall_score(test_y, y_pred)
        elif scoring == 'f1':
            scores[scoring] = f1_score(test_y, y_pred)
    return scores

def cross_validation(train_x: pd.DataFrame, train_y: pd.DataFrame, model, k: int=5, scorings: list=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
    """Perform k-fold cross validation on the data.

    Parameters:
        - data: DataFrame
        - model_type: str, the type of model to create
        - k: int, default=5
        - scorings: list, default=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    Returns:
        - scores: dict containing the scores for each metric
    """
    scores = {}
    for scoring in scorings:
        cv_scores = cross_val_score(model, train_x, train_y, cv=k, scoring=scoring)
        scores[scoring] = {
                "mean": cv_scores.mean(),
                "std": cv_scores.std(),
                "scores": cv_scores
            }
    return scores

def get_roc_metrics(test_y, y_pred):
    """Get the ROC curve.

    Parameters:
        - test_y: DataFrame
        - y_pred: DataFrame

    Returns:
        - fpr: array
        - tpr: array
    """
    fpr, tpr, threshold = roc_curve(test_y, y_pred)
    auc = roc_auc_score(test_y, y_pred)
    return fpr, tpr, threshold, auc

def permutation_features(train_x: pd.DataFrame, train_y: pd.DataFrame, model, n_shuffle, seed=42):
    """Compute the permutation importance of each feature.
    
    Parameters:
        - train_x: DataFrame
        - train_y: DataFrame
        - model: the model object
        - n_shuffle: int, the number of times to shuffle the data
        - seed: int, default=42
    
    Returns:
        - error_permutation: dict
    """
    error_permutation = permutation_importance(model, train_x, train_y, n_repeats=n_shuffle, random_state=seed)
    return error_permutation
