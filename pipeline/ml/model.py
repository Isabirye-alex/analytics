import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler, OneHotEncoder


class ChurnModel:
    """
    Trains a churn prediction model safely, using Pandas sample to split train/test.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """
        Args:
            X (pd.DataFrame): Numeric feature matrix
            y (pd.Series): Target variable (Churn)
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self.X = X.copy()
        self.y = y.copy()
        self.pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                         random_state=42, class_weight="balanced"
                    ),
                ),
            ]
        )

    def train(self, test_frac: float = 0.2):
        """
        Train the Random Forest model and print evaluation metrics.
        Uses Pandas sample for splitting.

        Args:
            test_frac (float): Fraction of data to hold out for testing
        """

        #Sample test set
        # Get 20% test set with similar class distribution
        test_indices = (
            self.X.groupby(self.y).sample(frac=test_frac, random_state=42).index
        )
        X_test = self.X.loc[test_indices]
        y_test = self.y.loc[test_indices]

        #Remaining rows become training set
        X_train = self.X.drop(test_indices)
        y_train = self.y.drop(test_indices)

        # Fit the model
        self.pipe.fit(X_train, y_train)

        #Predict on test set
        preds = self.pipe.predict(X_test)

        #Print evaluation
        print("Classification Report:")
        print(classification_report(y_test, preds))

        return self.pipe
