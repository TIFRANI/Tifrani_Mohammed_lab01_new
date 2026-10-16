import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_iris

# =====================================================================
# TASK 1: DATA PREPROCESSING (Breast Cancer via ucimlrepo)
# =====================================================================

def preprocess_breast_cancer(breast_cancer_obj):
    """
    Extracts DataFrames from the ucimlrepo object, imputes missing values ('?')
    using column mode, calculates level counts per feature (v), and prepends
    the level vector as Row 0 of matrix X (Task 3 Modification 1).
    """
    X_df = breast_cancer_obj.data.features.copy()
    y_df = breast_cancer_obj.data.targets.copy()

    df = pd.concat([y_df, X_df], axis=1)

    # 1. Mode Imputation for Missing Values ('?' or NaN)
    for col in df.columns:
        df[col] = df[col].replace('?', np.nan)
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    X_data = df.iloc[:, 1:].astype(str).values
    y_data = df.iloc[:, 0].astype(str).values

    # 2. Count unique categorical levels (v) for each attribute column
    num_levels = [len(np.unique(X_data[:, j])) for j in range(X_data.shape[1])]

    # 3. Task 3 Modification 1: Prepend feature levels vector as Row 0 of matrix X
    v_row = np.array([num_levels], dtype=str)
    X_with_levels = np.vstack([v_row, X_data])
    y_with_levels = np.insert(y_data, 0, 'LEVELS_HEADER')

    print(f"--- Task 1: Dataset Preprocessing ---")
    print(f"Breast Cancer Samples: {X_data.shape[0]} rows | Features: {X_data.shape[1]} columns")
    print(f"Levels per attribute column (v): {num_levels}\n")

    return X_with_levels, y_with_levels

def train_test_split_with_levels(X_with_levels, y_with_levels, train_ratio=0.7, seed=42):
    """
    Splits dataset into train/test subsets while preserving the level counts
    row (Row 0) at index 0 of both feature matrices.
    """
    levels_X = X_with_levels[0:1]
    levels_y = y_with_levels[0:1]
    
    data_X = X_with_levels[1:]
    data_y = y_with_levels[1:]
    
    np.random.seed(seed)
    n_samples = len(data_X)
    shuffled_idx = np.random.permutation(n_samples)
    
    train_size = int(n_samples * train_ratio)
    train_idx = shuffled_idx[:train_size]
    test_idx = shuffled_idx[train_size:]
    
    X_train = np.vstack([levels_X, data_X[train_idx]])
    y_train = np.concatenate([levels_y, data_y[train_idx]])
    
    X_test = np.vstack([levels_X, data_X[test_idx]])
    y_test = np.concatenate([levels_y, data_y[test_idx]])
    
    return X_train, X_test, y_train, y_test


# =====================================================================
# TASKS 2, 3 & THINK FURTHER (Part 1): NAIVE BAYES CLASSIFIER
# =====================================================================

class Nbayes:
    trained = False

    def __init__(self, alpha=1.0, use_log_probs=True):
        """
        Parameters:
        -----------
        alpha : float
            Smoothing parameter 'a'.
            a = 0.0 : Standard Naive Bayes (Task 2)
            a = 1.0 : Standard Laplace Smoothing (Task 3)
            a > 1.0 / a < 1.0 : Custom prior weighting
        use_log_probs : bool, default=True
            Think Further Part 1: Uses log probabilities (summation) to prevent 
            floating-point underflow when working with multiple attributes.
        """
        self.alpha = float(alpha)
        self.use_log_probs = use_log_probs
        self.v = None
        self.d = 0
        self.classes = []
        self.priors = {}
        self.likelihoods = {}
        self.known_values = []

    def fit(self, X_train, y_train):
        X = np.array(X_train)
        y = np.array(y_train)

        # Extract levels row from index 0 (Modification 1)
        self.v = X[0, :].astype(int)
        
        # Strip header row from training data
        X_data = X[1:, :]
        y_data = y[1:]

        n_samples, self.d = X_data.shape
        self.classes, counts = np.unique(y_data, return_counts=True)

        # 1. Compute Class Priors P(c)
        self.priors = {c: count / n_samples for c, count in zip(self.classes, counts)}

        # Track unique feature levels seen during training
        self.known_values = [set() for _ in range(self.d)]
        for j in range(self.d):
            self.known_values[j].update(np.unique(X_data[:, j]))

        # 2. Compute Likelihoods P(x_j = y | c)
        self.likelihoods = {c: {} for c in self.classes}

        for c in self.classes:
            X_c = X_data[y_data == c]
            n_c = len(X_c)

            for j in range(self.d):
                self.likelihoods[c][j] = {}
                v_j = self.v[j]

                for val in self.known_values[j]:
                    count_y = np.sum(X_c[:, j] == val)

                    if self.alpha > 0:
                        # Task 3: Laplace additive smoothing formula
                        prob = (count_y + self.alpha) / (n_c + self.alpha * v_j)
                    else:
                        # Task 2: Raw frequency formula
                        prob = count_y / float(n_c) if n_c > 0 else 0.0

                    self.likelihoods[c][j][val] = prob

        self.trained = True

    def predict(self, X_test):
        if not self.trained:
            raise ValueError("Model is not trained yet. Call 'fit' first.")

        X = np.array(X_test)
        
        # Strip levels header row if present in X_test
        if len(X) > 0 and np.array_equal(X[0, :].astype(int), self.v):
            X_data = X[1:, :]
        else:
            X_data = X

        y_predict = []

        for row_idx, instance in enumerate(X_data):
            # Check for unseen feature levels not in training set
            unseen = False
            for j in range(self.d):
                if instance[j] not in self.known_values[j]:
                    unseen = True
                    break

            if unseen:
                print(f"Error: Row {row_idx} contains an unseen attribute level. Discarding pattern.")
                y_predict.append(None)
                continue

            class_scores = {}

            if self.use_log_probs:
                # THINK FURTHER (PART 1): Log-space summation to avoid numerical underflow
                for c in self.classes:
                    log_prob = np.log(self.priors[c])
                    for j in range(self.d):
                        val = instance[j]
                        likelihood = self.likelihoods[c][j].get(val, 0.0)
                        
                        if likelihood == 0.0:
                            log_prob = -np.inf
                            break
                        else:
                            log_prob += np.log(likelihood)
                            
                    class_scores[c] = log_prob
            else:
                # Standard probability multiplication
                for c in self.classes:
                    prob = self.priors[c]
                    for j in range(self.d):
                        val = instance[j]
                        prob *= self.likelihoods[c][j].get(val, 0.0)
                    class_scores[c] = prob

            # Select class with maximum posterior score
            best_class = max(class_scores, key=class_scores.get)
            y_predict.append(best_class)

        return np.array(y_predict)

    def test(self, X_test, y_test):
        if not self.trained:
            raise ValueError("Model is not trained yet. Call 'fit' first.")

        y_test_arr = np.array(y_test)
        
        if y_test_arr[0] == 'LEVELS_HEADER':
            y_test_arr = y_test_arr[1:]

        y_predict = self.predict(X_test)

        valid_mask = y_predict != None
        if np.sum(valid_mask) == 0:
            return 0.0

        return (y_test_arr[valid_mask] == y_predict[valid_mask]).sum() / len(y_test_arr[valid_mask])


# =====================================================================
# THINK FURTHER (PART 2 & 3): CONTINUOUS FEATURE EXPERIMENT (IRIS)
# =====================================================================

def run_continuous_iris_experiment():
    """
    Think Further Parts 2 & 3:
    Demonstrates handling continuous variables by binarizing the Iris dataset 
    using column medians and evaluating it with our Nbayes classifier.
    """
    iris = load_iris()
    X_continuous = iris.data
    y_labels = iris.target.astype(str)

    # Convert continuous features to binary strings ('0' or '1') using feature medians
    X_binary = np.zeros_like(X_continuous, dtype=str)
    for j in range(X_continuous.shape[1]):
        median_val = np.median(X_continuous[:, j])
        X_binary[:, j] = np.where(X_continuous[:, j] >= median_val, '1', '0')

    # Binarized features have v = 2 levels per attribute
    num_levels_iris = [2, 2, 2, 2]
    v_row = np.array([num_levels_iris], dtype=str)
    
    X_with_levels = np.vstack([v_row, X_binary])
    y_with_levels = np.insert(y_labels, 0, 'LEVELS_HEADER')

    X_tr, X_te, y_tr, y_te = train_test_split_with_levels(X_with_levels, y_with_levels, train_ratio=0.7)

    clf_iris = Nbayes(alpha=1.0, use_log_probs=True)
    clf_iris.fit(X_tr, y_tr)
    accuracy = clf_iris.test(X_te, y_te)

    print("--- Think Further Part 3: Iris Dataset Experiment ---")
    print(f"Continuous features discretized to binary via Median Split.")
    print(f"Binarized Iris Accuracy (Laplace Smoothing, a = 1.0): {accuracy * 100:.2f}%\n")


# =====================================================================
# MAIN SCRIPT EXECUTION
# =====================================================================

if __name__ == "__main__":
    # 1. Fetch UCI Breast Cancer Dataset
    breast_cancer = fetch_ucirepo(id=14)

    # 2. Execute Task 1 Preprocessing
    X_data, y_data = preprocess_breast_cancer(breast_cancer)

    # 3. Train/Test Split (70% Train, 30% Test)
    X_train, X_test, y_train, y_test = train_test_split_with_levels(X_data, y_data, train_ratio=0.7)

    print("======================================================")
    print("UCI BREAST CANCER DATASET EVALUATION")
    print("======================================================")

    # Task 2: Raw Frequency (a = 0.0)
    clf_t2 = Nbayes(alpha=0.0, use_log_probs=True)
    clf_t2.fit(X_train, y_train)
    acc_t2 = clf_t2.test(X_test, y_test)
    print(f"Task 2 (Raw Frequency, a = 0.0): Accuracy = {acc_t2 * 100:.2f}%")

    # Task 3: Laplace Smoothing (a = 1.0)
    clf_t3 = Nbayes(alpha=1.0, use_log_probs=True)
    clf_t3.fit(X_train, y_train)
    acc_t3 = clf_t3.test(X_test, y_test)
    print(f"Task 3 (Laplace Smoothing, a = 1.0): Accuracy = {acc_t3 * 100:.2f}%")

    # Task 3 Refinements
    clf_a2 = Nbayes(alpha=2.0, use_log_probs=True)
    clf_a2.fit(X_train, y_train)
    print(f"Task 3 Refinement (a = 2.0):          Accuracy = {clf_a2.test(X_test, y_test) * 100:.2f}%")

    clf_a05 = Nbayes(alpha=0.5, use_log_probs=True)
    clf_a05.fit(X_train, y_train)
    print(f"Task 3 Refinement (a = 0.5):          Accuracy = {clf_a05.test(X_test, y_test) * 100:.2f}%\n")

    # 4. Execute Think Further Section
    print("======================================================")
    print("THINK FURTHER EXPERIMENTS")
    print("======================================================")
    run_continuous_iris_experiment()