import pandas as pd
import numpy as np

# =====================================================================
# TASK 1 & 3 DATA PREPARATION: Adding Number of Levels as First Row
# =====================================================================

def prepare_weather_dataset_with_levels(filepath='Weather_data_set.txt'):
    """
    Reads the weather dataset, calculates feature levels, and prepends
    the level counts as the first row as specified in Modification 1.
    """
    df = pd.read_csv(filepath, sep=r'\s+', comment='#', header=None,
                     names=['Outlook', 'Temperature', 'Humidity', 'Windy', 'Play'])
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Weather feature levels: Outlook (3), Temperature (3), Humidity (2), Windy (2)
    # Stored as string to match the categorical matrix dtype
    v_levels = np.array([[3, 3, 2, 2]], dtype=str)
    
    # Prepend the level vector as the first row of X
    X_with_levels = np.vstack([v_levels, X])
    
    # Add dummy target label for the levels row (will be stripped during fit)
    y_with_levels = np.insert(y, 0, 'LEVELS_HEADER')
    
    return X_with_levels, y_with_levels

def train_test_split_with_levels(X_with_levels, y_with_levels, train_ratio=0.7, seed=42):
    """
    Splits data into train and test sets while ensuring the first row 
    (number of levels) is preserved at index 0 of both sets.
    """
    # Separate header row from actual dataset
    levels_row_X = X_with_levels[0:1]
    levels_row_y = y_with_levels[0:1]
    
    data_X = X_with_levels[1:]
    data_y = y_with_levels[1:]
    
    np.random.seed(seed)
    n_samples = len(data_X)
    shuffled_idx = np.random.permutation(n_samples)
    
    train_size = int(n_samples * train_ratio)
    train_idx = shuffled_idx[:train_size]
    test_idx = shuffled_idx[train_size:]
    
    # Re-attach the levels row as row 0 for both training and test sets
    X_train = np.vstack([levels_row_X, data_X[train_idx]])
    y_train = np.concatenate([levels_row_y, data_y[train_idx]])
    
    X_test = np.vstack([levels_row_X, data_X[test_idx]])
    y_test = np.concatenate([levels_row_y, data_y[test_idx]])
    
    return X_train, X_test, y_train, y_test


# =====================================================================
# TASK 3: Naive Bayes Classifier with Laplace (Additive) Smoothing
# =====================================================================

class Nbayes:
    trained = False

    def __init__(self, alpha=1.0):
        """
        Parameters:
        -----------
        alpha : float, default=1.0
            Smoothing parameter 'a'.
            a = 1.0 : Standard Laplace smoothing
            a > 1.0 : Higher trust in prior belief
            a < 1.0 : Lower trust in prior belief
            a = 0.0 : Raw frequency (no smoothing)
        """
        self.alpha = float(alpha)
        self.v = None            # Vector storing number of levels per feature
        self.d = 0               # Dimensionality (number of features)
        self.classes = []
        self.priors = {}
        self.likelihoods = {}
        self.known_values = []

    def fit(self, X_train, y_train):
        X = np.array(X_train)
        y = np.array(y_train)

        # -------------------------------------------------------------
        # Modification 1: Extract and store the levels row from row 0
        # -------------------------------------------------------------
        self.v = X[0, :].astype(int)
        
        # Strip the level header row from training data
        X_data = X[1:, :]
        y_data = y[1:]

        n_samples, self.d = X_data.shape
        self.classes, counts = np.unique(y_data, return_counts=True)

        # Compute Class Priors P(c)
        self.priors = {c: count / n_samples for c, count in zip(self.classes, counts)}

        # Track unique feature levels seen in training
        self.known_values = [set() for _ in range(self.d)]
        for j in range(self.d):
            self.known_values[j].update(np.unique(X_data[:, j]))

        # -------------------------------------------------------------
        # Modification 2: Compute Likelihoods with Laplace Smoothing
        # P(x_j = y | class z) = (count + a) / (N_z + a * v_j)
        # -------------------------------------------------------------
        self.likelihoods = {c: {} for c in self.classes}

        for c in self.classes:
            X_c = X_data[y_data == c]  # Sub-matrix for class c
            n_c = len(X_c)            # Number of observations in class c (N_z)

            for j in range(self.d):
                self.likelihoods[c][j] = {}
                v_j = self.v[j]  # Number of possible values (v) for column j

                for val in self.known_values[j]:
                    # Number of observations in class c where attribute x_j has value y
                    count_y = np.sum(X_c[:, j] == val)

                    if self.alpha > 0:
                        # Task 3 Formula: ((count) + a) / ((n_c) + a * v_j)
                        prob = (count_y + self.alpha) / (n_c + self.alpha * v_j)
                    else:
                        # Task 2 Raw Frequency: count / n_c
                        prob = count_y / float(n_c) if n_c > 0 else 0.0

                    self.likelihoods[c][j][val] = prob

        self.trained = True

    def predict(self, X_test):
        if not self.trained:
            raise ValueError("Model is not trained yet. Call 'fit' first.")

        X = np.array(X_test)
        
        # Strip levels row from test set if present as first row
        if len(X) > 0 and np.array_equal(X[0, :].astype(int), self.v):
            X_data = X[1:, :]
        else:
            X_data = X

        y_predict = []

        for row_idx, instance in enumerate(X_data):
            # Check for unseen feature levels not present in training set
            unseen = False
            for j in range(self.d):
                if instance[j] not in self.known_values[j]:
                    unseen = True
                    break

            if unseen:
                print(f"Error: Row {row_idx} has unseen feature value. Discarding pattern.")
                y_predict.append(None)
                continue

            # Posterior calculation using Laplace-smoothed probabilities
            class_posteriors = {}
            for c in self.classes:
                prob = self.priors[c]
                for j in range(self.d):
                    val = instance[j]
                    
                    if val in self.likelihoods[c][j]:
                        prob *= self.likelihoods[c][j][val]
                    else:
                        # Fallback prior belief 1/v if value was never observed in class c
                        prob *= (self.alpha / (len(X_data) + self.alpha * self.v[j]))

                class_posteriors[c] = prob

            best_class = max(class_posteriors, key=class_posteriors.get)
            y_predict.append(best_class)

        return np.array(y_predict)

    def test(self, X_test, y_test):
        if not self.trained:
            raise ValueError("Model is not trained yet. Call 'fit' first.")

        y_test_arr = np.array(y_test)
        
        # Strip header row from y_test if present
        if y_test_arr[0] == 'LEVELS_HEADER':
            y_test_arr = y_test_arr[1:]

        y_predict = self.predict(X_test)

        valid_mask = y_predict != None
        if np.sum(valid_mask) == 0:
            return 0.0

        return (y_test_arr[valid_mask] == y_predict[valid_mask]).sum() / len(y_test_arr[valid_mask])


# =====================================================================
# EXPERIMENTAL SCRIPT & COMPARISON (a = 1, a > 1, a < 1)
# =====================================================================

if __name__ == "__main__":
    # 1. Prepare Weather data with levels header [3, 3, 2, 2]
    X_data, y_data = prepare_weather_dataset_with_levels('Weather_data_set.txt')
    
    # 2. Split train/test maintaining the level header
    X_train, X_test, y_train, y_test = train_test_split_with_levels(X_data, y_data, train_ratio=0.7)

    print("--- Experiment 1: Raw Frequency (Without Smoothing, a = 0) ---")
    clf_no_smooth = Nbayes(alpha=0.0)
    clf_no_smooth.fit(X_train, y_train)
    acc_no_smooth = clf_no_smooth.test(X_test, y_test)
    print(f"Accuracy: {acc_no_smooth * 100:.2f}%\n")

    print("--- Experiment 2: Standard Laplace Smoothing (a = 1.0) ---")
    clf_laplace = Nbayes(alpha=1.0)
    clf_laplace.fit(X_train, y_train)
    acc_laplace = clf_laplace.test(X_test, y_test)
    print(f"Accuracy: {acc_laplace * 100:.2f}%\n")

    print("--- Experiment 3: Strong Prior Belief (a = 2.0) ---")
    clf_high_a = Nbayes(alpha=2.0)
    clf_high_a.fit(X_train, y_train)
    acc_high_a = clf_high_a.test(X_test, y_test)
    print(f"Accuracy: {acc_high_a * 100:.2f}%\n")

    print("--- Experiment 4: Weak Prior Belief (a = 0.5) ---")
    clf_low_a = Nbayes(alpha=0.5)
    clf_low_a.fit(X_train, y_train)
    acc_low_a = clf_low_a.test(X_test, y_test)
    print(f"Accuracy: {acc_low_a * 100:.2f}%")