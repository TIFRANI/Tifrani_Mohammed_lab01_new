import pandas as pd
import numpy as np

# =====================================================================
# TASK 1 CONTINUATION: Preprocessing Utilities
# =====================================================================

def load_and_preprocess_weather(filepath='Weather_data_set.txt'):
    """Reads and prepares the Weather dataset."""
    df_weather = pd.read_csv(filepath, sep=r'\s+', comment='#', header=None,
                             names=['Outlook', 'Temperature', 'Humidity', 'Windy', 'Play'])
    X_weather = df_weather.iloc[:, :-1].values
    y_weather = df_weather.iloc[:, -1].values
    return X_weather, y_weather

def train_test_split_custom(X, y, train_ratio=0.7, seed=42):
    """Splits data into training and testing subsets."""
    np.random.seed(seed)
    n_samples = len(X)
    shuffled_indices = np.random.permutation(n_samples)
    
    train_size = int(n_samples * train_ratio)
    train_idx = shuffled_indices[:train_size]
    test_idx = shuffled_indices[train_size:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# =====================================================================
# TASK 2: Naive Bayes Classifier Implementation
# =====================================================================

class Nbayes:
    trained = False

    def fit(self, X_train, y_train):
        # Convert inputs to numpy arrays
        X = np.array(X_train)
        y = np.array(y_train)

        n_samples, self.d = X.shape
        self.classes, counts = np.unique(y, return_counts=True)

        # 1. Store Class Priors P(c) = (count of class c) / (total samples)
        self.priors = {}
        for c, count in zip(self.classes, counts):
            self.priors[c] = count / n_samples

        # 2. Store Class Likelihoods P(x_j = v | c) using pseudocode specification
        # Structure: self.likelihoods[class][feature_idx][value] = likelihood
        self.likelihoods = {c: {} for c in self.classes}
        self.known_values = [set() for _ in range(self.d)]

        # Track all unique values observed in training set per attribute
        for j in range(self.d):
            self.known_values[j].update(np.unique(X[:, j]))

        # Calculate likelihoods: for each class c, variable x_j, and value v
        for c in self.classes:
            X_c = X[y == c]  # Sub-matrix of training samples belonging to class c
            n_c = len(X_c)

            for j in range(self.d):
                self.likelihoods[c][j] = {}
                for v in self.known_values[j]:
                    # Instances of class c where attribute x_j == v, divided by total instances of class c
                    count_v = np.sum(X_c[:, j] == v)
                    self.likelihoods[c][j][v] = count_v / n_c

        self.trained = True

    def predict(self, X_test):
        if not self.trained:
            raise ValueError("Model is not trained yet. Call 'fit' first.")

        X = np.array(X_test)
        y_predict = []

        for row_idx, instance in enumerate(X):
            # Check for unseen feature values in test set not present in training set
            unseen_found = False
            for j in range(self.d):
                if instance[j] not in self.known_values[j]:
                    unseen_found = True
                    break

            # Handle unseen feature values: issue error message and discard pattern
            if unseen_found:
                print(f"Error: Sample at index {row_idx} contains an unseen attribute value. Discarding pattern.")
                y_predict.append(None)
                continue

            # Compute posterior scores P(c) * PROD(P(x_j | c))
            class_posteriors = {}
            for c in self.classes:
                prob = self.priors[c]
                for j in range(self.d):
                    val = instance[j]
                    # Fetch stored probability mass function value
                    prob *= self.likelihoods[c][j].get(val, 0.0)
                class_posteriors[c] = prob

            # Select class with maximum posterior probability
            best_class = max(class_posteriors, key=class_posteriors.get)
            y_predict.append(best_class)

        return np.array(y_predict)

    def test(self, X_test, y_test):
        if not self.trained:
            raise ValueError("Model is not trained yet. Call 'fit' first.")
        y_predict = self.predict(X_test)
        
        # Filter out discarded (None) predictions before calculating accuracy
        valid_mask = y_predict != None
        if np.sum(valid_mask) == 0:
            return 0.0
            
        return (np.array(y_test)[valid_mask] == np.array(y_predict)[valid_mask]).sum() / len(np.array(y_test)[valid_mask])


# =====================================================================
# TESTING SCRIPT
# =====================================================================

if __name__ == "__main__":
    # 1. Load data from Task 1
    X_weather, y_weather = load_and_preprocess_weather('Weather_data_set.txt')
    
    # 2. Split dataset into 70% Train and 30% Test
    X_train, X_test, y_train, y_test = train_test_split_custom(X_weather, y_weather, train_ratio=0.7)
    
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # 3. Instantiate, train, and test classifier
    clf = Nbayes()
    clf.fit(X_train, y_train)
    
    print("\n--- Predictions on Test Set ---")
    predictions = clf.predict(X_test)
    print(f"True Labels:      {y_test}")
    print(f"Predicted Labels: {predictions}")
    
    # 4. Compute Test Accuracy
    accuracy = clf.test(X_test, y_test)
    print(f"\nAccuracy (Standard Naive Bayes): {accuracy * 100:.2f}%")

    