import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict, Any
from reusables.reusable_functions import ReusableFunctions
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression


class ChurnModel:
    """
    ChurnModel encapsulates training and evaluation of a churn classifier.

    Fixes applied:
        - Proper feature/target separation
        - Leakage removed (Recency dropped)
        - Clean train/test split with stratification
        - Correct pipeline usage
    """

    # Columns that should NEVER be used as features
    DROP_COLUMNS = [
        "CustomerNo",
        "Segment",
        "M_SCORE",
        "R_SCORE",
        "F_SCORE",
        "Recency",  
    ]

    TARGET_COLUMN = "Churn"

    def __init__(
        self,
        dataframe: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize model.

        Args:
            dataframe (pd.DataFrame): Full dataset including features + target
            test_size (float): Test split ratio
            random_state (int): Seed for reproducibility
        """

        self.df = dataframe.copy()
        self.test_size = test_size
        self.random_state = random_state
        self.metrics: Dict[str, Any] = {}

        self.logger = ReusableFunctions.setup_logger(self.__class__.__name__)

        # Define pipeline
        self.pipe = Pipeline(
            [
                (
                    "scaler", StandardScaler()),
                    ('imputer', SimpleImputer(strategy='median')),
                    ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
                (
                    "model",
                    LogisticRegression(
                        random_state=42,
                        class_weight="balanced",
                        max_iter=100,
                        C=0.1,

                    ),
                ),
            ]
        )

    def _prepare_features(self):
        """
        Separate features and target safely.

        Returns:
            X (pd.DataFrame): Clean feature matrix
            y (pd.Series): Target labels
        """

        if self.TARGET_COLUMN not in self.df.columns:
            raise ValueError("Churn column missing from dataset")

        # Separate target
        y = self.df[self.TARGET_COLUMN]

        # Drop leakage + unwanted columns
        X = self.df.drop(columns=self.DROP_COLUMNS + [self.TARGET_COLUMN])

        self.logger.info(f"Feature shape: {X.shape}, Target shape: {y.shape}")

        return X, y

    def _split_data(self, X, y):
        """
        Perform stratified train/test split.

        Returns:
            X_train, X_test, y_train, y_test
        """

        self.logger.info(
            f"Splitting data (test_size={self.test_size}, random_state={self.random_state})"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,  #mportant for imbalance
        )

        self.logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def _evaluate(self, y_test, preds):
        """
        Evaluate model performance.
        """

        accuracy = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        self.metrics["accuracy"] = round(accuracy, 4)
        self.metrics["classification_report"] = report

        self.logger.info(f"Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, preds))

    def get_feature_importance(self, x):
        classifier = self.pipe.named_steps['model']

        importance_df = pd.DataFrame(
            {
                'Feature': x.columns.tolist(),
                'Coefficient': classifier.coef_[0],
                'AbsCoefficient': abs(classifier.coef_[0]),

            }
        ).sort_values('AbsCoefficient', ascending=False)
        print(importance_df)

    def train(self) -> Dict[str, Any]:
        """
        Full training pipeline.

        Steps:
            1. Prepare features
            2. Split data
            3. Train model
            4. Evaluate model
        """

        self.logger.info("Starting training pipeline")

        # Step 1: Prepare data
        X, y = self._prepare_features()

        # Step 2: Split
        X_train, X_test, y_train, y_test = self._split_data(X, y)

        # Step 3: Train
        self.logger.info("Training model")
        self.pipe.fit(X_train, y_train)

        # Step 4: Predict
        preds = self.pipe.predict_proba(X_test)[:, 1]
        preds = (preds > 0.65).astype(int)
        

        # Step 5: Evaluate
        self._evaluate(y_test, preds)
        # self.get_feature_importance(X_test)
        self.get_feature_importance(X_train)
        self.logger.info("Training completed")

        return self.metrics
