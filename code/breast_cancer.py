import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

# =====================================================================
# TASK 1: DATA PREPROCESSING (Using ucimlrepo)
# =====================================================================

def preprocess_breast_cancer(breast_cancer_obj):
    """
    Extracts DataFrames from the ucimlrepo object, imputes missing values
    ('?') using column mode, calculates level counts per feature, and 
    prepends the level vector as Row 0 of matrix X (Task 3 Modification 1).
    """
    # Extract feature matrix X and target matrix y as DataFrames
    X_df = breast_cancer_obj.data.features.copy()
    y_df = breast_cancer_obj.data.targets.copy()

    # Combine temporarily for missing value handling
    df = pd.concat([y_df, X_df], axis=1)

    # Replace '?' or NaN missing values with the column mode
    for col in df.columns:
        df[col] = df[col].replace('?', np.nan)
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    # Extract target y (column 0) and features X (columns 1..9) as string arrays
    X_data = df.iloc[:, 1:].astype(str).values
    y_data = df.iloc[:, 0].astype(str).values

    # Count unique levels (v) for each of the 9 categorical features
    num_levels = [len(np.unique(X_data[:, j])) for j in range(X_data.shape[1])]

    # Prepend the level counts as Row 0 (Task 3 Modification 1)
    v_row = np.array([num_levels], dtype=str)
    X_with_levels = np.vstack([v_row, X_data])
    y_with_levels = np.insert(y_data, 0, 'LEVELS_HEADER')

    print(f"Dataset Loaded via ucimlrepo: {X_data.shape[0]} observations, {X_data.shape[1]} features")
    print(f"Attribute Levels per Column (v): {num_levels}\n")

    return X_with_levels, y_with_levels

def train_test_split_with_levels(X_with_levels, y_with_levels, train_ratio=0.7, seed=42):
    """
    Splits data into train and test subsets while preserving the 
    levels header (Row 0) at index 0 of both feature matrices.
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
# TASK 2 & TASK 3: NAIVE BAYES CLASSIFIER CLASS TEMPLATE
# =====================================================================

class Nbayes:
    trained = False

    def __init__(self, alpha=1.0):
        """
        Parameters:
        -----------
        alpha : float
            Smoothing parameter 'a'.
            a = 0.0 : Standard Naive Bayes (Task 2)
            a = 1.0 : Standard Laplace Smoothing (Task 3)
            a > 1.0 / a < 1.0 : Prior weighting experiments
        """
        self.alpha = float(alpha)
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
        
        # Strip level header row from training data
        X_data = X[1:, :]
        y_data = y[1:]

        n_samples, self.d = X_data.shape
        self.classes, counts = np.unique(y_data, return_counts=True)

        # 1. Compute Class Priors P(c)
        self.priors = {c: count / n_samples for c, count in zip(self.classes, counts)}

        # Track unique attribute levels seen in training data
        self.known_values = [set() for _ in range(self.d)]
        for j in range(self.d):
            self.known_values[j].update(np.unique(X_data[:, j]))

        # 2. Compute Likelihoods P(x_j = y | c)
        self.likelihoods = {c: {} for c in self.classes}

        for c in self.classes:
            X_c = X_data[y_data == c]
            n_c = len(X_c)  # Observations in class c

            for j in range(self.d):
                self.likelihoods[c][j] = {}
                v_j = self.v[j]  # Total levels for attribute j

                for val in self.known_values[j]:
                    count_y = np.sum(X_c[:, j] == val)

                    if self.alpha > 0:
                        # Task 3 Formula: ((count) + a) / ((n_c) + a * v_j)
                        prob = (count_y + self.alpha) / (n_c + self.alpha * v_j)
                    else:
                        # Task 2 Formula: count / n_c
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
            # Check for unseen attribute values not in training set
            unseen = False
            for j in range(self.d):
                if instance[j] not in self.known_values[j]:
                    unseen = True
                    break

            # Handle unseen feature values: issue error message and discard pattern
            if unseen:
                print(f"Error: Row {row_idx} contains unseen attribute value. Discarding pattern.")
                y_predict.append(None)
                continue

            # Calculate class posteriors
            class_posteriors = {}
            for c in self.classes:
                prob = self.priors[c]
                for j in range(self.d):
                    val = instance[j]
                    likelihood = self.likelihoods[c][j].get(val, 0.0)
                    prob *= likelihood

                class_posteriors[c] = prob

            # Select class with maximum posterior probability
            best_class = max(class_posteriors, key=class_posteriors.get)
            y_predict.append(best_class)

        return np.array(y_predict)

    def test(self, X_test, y_test):
        if not self.trained:
            raise ValueError("Model is not trained yet. Call 'fit' first.")

        y_test_arr = np.array(y_test)
        
        # Strip header row from targets if present
        if y_test_arr[0] == 'LEVELS_HEADER':
            y_test_arr = y_test_arr[1:]

        y_predict = self.predict(X_test)

        valid_mask = y_predict != None
        if np.sum(valid_mask) == 0:
            return 0.0

        return (y_test_arr[valid_mask] == y_predict[valid_mask]).sum() / len(y_test_arr[valid_mask])


# =====================================================================
# MAIN EXPERIMENTAL SCRIPT
# =====================================================================

if __name__ == "__main__":
    # 1. Fetch Dataset using ucimlrepo
    breast_cancer = fetch_ucirepo(id=14)

    # 2. Preprocess Data (Task 1)
    X_data, y_data = preprocess_breast_cancer(breast_cancer)

    # 3. Train/Test Split (70-30 split)
    X_train, X_test, y_train, y_test = train_test_split_with_levels(X_data, y_data, train_ratio=0.7)

    print("======================================================")
    print("EXPERIMENTAL EVALUATION ON BREAST CANCER DATASET")
    print("======================================================\n")

    # Task 2 Evaluation (a = 0.0)
    clf_task2 = Nbayes(alpha=0.0)
    clf_task2.fit(X_train, y_train)
    acc_task2 = clf_task2.test(X_test, y_test)
    print(f"Task 2 (Raw Frequency, a = 0.0): Accuracy = {acc_task2 * 100:.2f}%")

    # Task 3 Evaluation (a = 1.0)
    clf_task3 = Nbayes(alpha=1.0)
    clf_task3.fit(X_train, y_train)
    acc_task3 = clf_task3.test(X_test, y_test)
    print(f"Task 3 (Laplace Smoothing, a = 1.0): Accuracy = {acc_task3 * 100:.2f}%")

    # Task 3 Refinement (a = 2.0)
    clf_a2 = Nbayes(alpha=2.0)
    clf_a2.fit(X_train, y_train)
    acc_a2 = clf_a2.test(X_test, y_test)
    print(f"Task 3 Refinement (a = 2.0):          Accuracy = {acc_a2 * 100:.2f}%")

    # Task 3 Refinement (a = 0.5)
    clf_a05 = Nbayes(alpha=0.5)
    clf_a05.fit(X_train, y_train)
    acc_a05 = clf_a05.test(X_test, y_test)
    print(f"Task 3 Refinement (a = 0.5):          Accuracy = {acc_a05 * 100:.2f}%")