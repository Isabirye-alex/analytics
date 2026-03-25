import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional, Callable
from reusables.reusable_functions import ReusableFunctions


class ChurnModel:
    """
    ChurnModel encapsulates the full lifecycle of a churn classification model.

    Responsibilities:
        - Feature preparation and leakage prevention
        - Stratified train/test splitting
        - Pipeline training (GBM by default, swappable)
        - Threshold-based prediction
        - Model evaluation and cross-validation
        - Feature importance extraction
        - Scoring all customers with churn probability
        - Persisting and loading trained pipelines

    Attributes:
        df (pd.DataFrame): Full dataset including features and target.
        test_size (float): Proportion of data reserved for evaluation.
        random_state (int): Seed for reproducibility across all operations.
        pipe (Pipeline): sklearn Pipeline containing preprocessor + classifier.
        metrics (Dict): Evaluation metrics populated after training.

    Usage:
        model = ChurnModel(dataset)
        model.train(threshold=0.4)
        model.cross_validate_model()
        model.get_feature_importance()
        scores = model.predict(threshold=0.4)
        model.save("churn_model.pkl")
    """

    # Columns excluded from the feature matrix — leakage or identifiers
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
        pipe: Optional[Pipeline] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize ChurnModel.

        Args:
            dataframe (pd.DataFrame): Full dataset including features and target.
            pipe (Pipeline, optional): Custom sklearn Pipeline. Defaults to
                GradientBoostingClassifier pipeline if not provided.
            test_size (float): Fraction of data held out for evaluation. Default 0.2.
            random_state (int): Reproducibility seed. Default 42.
        """

        self.df = dataframe.copy()
        self.test_size = test_size
        self.random_state = random_state
        self.metrics: Dict[str, Any] = {}
        self.logger = ReusableFunctions.setup_logger(self.__class__.__name__)

        # Use injected pipeline or fall back to GBM default
        self.pipe = pipe if pipe is not None else self._default_pipeline()

        self.logger.info(
            f"Initialized with classifier: "
            f"{type(self.pipe.named_steps['classifier']).__name__}"
        )

        self.pipeline_steps: List[Callable] = [
            self._prepare_features,
            self._split_data,
            self.train,
            self._evaluate,
            self.cross_validate_model,
            self.get_feature_importance,
            self.predict,
            self.save,
            self.load
        ]

    # Pipeline Definition

    def _default_pipeline(self) -> Pipeline:
        """
        Build the default GradientBoosting pipeline.

        Returns:
            Pipeline: Imputer → Scaler → GradientBoostingClassifier
        """

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    # Data Preparation

    def _prepare_features(self):
        """
        Separate feature matrix from target label.

        Drops leakage columns and the target column from X.
        Silently skips DROP_COLUMNS that are not present in the dataset.

        Returns:
            X (pd.DataFrame): Clean feature matrix.
            y (pd.Series): Binary churn labels.

        Raises:
            ValueError: If TARGET_COLUMN is absent from the dataset.
        """

        if self.TARGET_COLUMN not in self.df.columns:
            raise ValueError(
                f"Target column '{self.TARGET_COLUMN}' missing from dataset."
            )

        y = self.df[self.TARGET_COLUMN]

        cols_to_drop = [
            col
            for col in self.DROP_COLUMNS + [self.TARGET_COLUMN]
            if col in self.df.columns
        ]
        X = self.df.drop(columns=cols_to_drop)

        self.logger.info(f"Features: {X.columns.tolist()}")
        self.logger.info(f"Feature shape: {X.shape} | Target shape: {y.shape}")

        return X, y

    def _split_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Perform stratified train/test split.

        Stratification preserves the churn ratio in each split,
        preventing accidental class imbalance in the test set.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target labels.

        Returns:
            X_train, X_test, y_train, y_test
        """

        self.logger.info(
            f"Splitting data — test_size={self.test_size}, "
            f"random_state={self.random_state}"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,  # preserves class ratio across splits
        )

        self.logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    # Evaluate
    def _evaluate(self, y_test: pd.Series, preds) -> None:
        """
        Compute and store evaluation metrics.

        Captures accuracy and the full per-class classification report.
        Stores results in self.metrics for downstream access.

        Args:
        y_test (pd.Series): Ground truth labels from the test set.
        preds: Predicted labels after threshold application.
        """

        accuracy = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        self.metrics["accuracy"] = round(accuracy, 4)
        self.metrics["classification_report"] = report

        self.logger.info(f"Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, preds))

    # Training

    def train(self, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Execute the full training pipeline.

        Steps:
            1. Prepare features and target
            2. Stratified train/test split
            3. Fit pipeline on training data
            4. Generate threshold-based predictions on test data
            5. Evaluate and store metrics

        Args:
            threshold (float): Decision threshold for churn classification.
                Lower values increase recall at the cost of precision.
                Default 0.5. Recommended 0.4 based on validation results.

        Returns:
            Dict[str, Any]: Evaluation metrics including accuracy,
                classification report, and threshold used.

        Raises:
            ValueError: If required columns are missing.
        """

        self.logger.info("Starting training pipeline")

        # Step 1: Prepare
        X, y = self._prepare_features()

        # Step 2: Split
        X_train, X_test, y_train, y_test = self._split_data(X, y)

        # Step 3: Fit
        self.logger.info("Fitting pipeline on training data")
        self.pipe.fit(X_train, y_train)

        # Step 4: Predict with threshold
        proba = self.pipe.predict_proba(X_test)[:, 1]
        preds = (proba > threshold).astype(int)

        # Step 5: Evaluate
        self._evaluate(y_test, preds)
        self.metrics["threshold"] = threshold

        self.logger.info("Training completed")

        return self.metrics

    # Evaluation

    def cross_validate_model(self, n_splits: int = 5) -> pd.Series:
        """
        Run stratified k-fold cross-validation on the full dataset.

        Validates that single test split results are not a lucky outcome.
        Should be called after train() to confirm model stability.

        Args:
            n_splits (int): Number of cross-validation folds. Default 5.

        Returns:
            pd.Series: Mean F1, precision, and recall across all folds.

        Raises:
            ValueError: If model has not been trained yet.
        """

        if not self.metrics:
            raise ValueError("Model has not been trained. Call train() first.")

        self.logger.info(f"Running {n_splits}-fold cross-validation")

        X, y = self._prepare_features()

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        scores = cross_validate(
            self.pipe,
            X,
            y,
            cv=cv,
            scoring=["f1", "precision", "recall"],
        )

        results = pd.DataFrame(scores)[
            ["test_f1", "test_precision", "test_recall"]
        ].mean()

        self.logger.info(f"Cross-validation results: {results.to_dict()}")

        print("\nCross-Validation Results:")
        print(results)

        return results

    # Feature Importance

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract and rank feature importances from the trained classifier.

        Handles both tree-based models (feature_importances_) and
        linear models (coef_) automatically.

        Returns:
            pd.DataFrame: Features ranked by importance or absolute
                coefficient value, descending.

        Raises:
            ValueError: If model has not been trained yet.
            ValueError: If classifier does not expose importances or coefficients.
        """

        if not self.metrics:
            raise ValueError("Model has not been trained. Call train() first.")

        X, _ = self._prepare_features()
        classifier = self.pipe.named_steps["classifier"]

        # Tree-based: GBM, RandomForest, XGBoost
        if hasattr(classifier, "feature_importances_"):
            importance_df = (
                pd.DataFrame(
                    {
                        "Feature": X.columns.tolist(),
                        "Importance": classifier.feature_importances_,
                    }
                )
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )

        # Linear: LogisticRegression
        elif hasattr(classifier, "coef_"):
            importance_df = (
                pd.DataFrame(
                    {
                        "Feature": X.columns.tolist(),
                        "Coefficient": classifier.coef_[0],
                        "AbsCoefficient": abs(classifier.coef_[0]),
                    }
                )
                .sort_values("AbsCoefficient", ascending=False)
                .reset_index(drop=True)
            )

        else:
            raise ValueError(
                f"{type(classifier).__name__} does not expose "
                f"feature importances or coefficients."
            )

        self.logger.info("Feature importances extracted successfully")

        print("\nFeature Importances:")
        print(importance_df)

        return importance_df

    # Scoring

    def predict(self, threshold: float = 0.4) -> pd.DataFrame:
        """
        Score all customers on churn probability using the trained model.

        Unlike _evaluate which scores only the test set, this method
        scores every customer in the full dataset.

        Args:
            threshold (float): Decision threshold for ChurnPrediction flag.
                Default 0.4. Should match threshold used in train().

        Returns:
            pd.DataFrame: All customers with ChurnProbability and
                ChurnPrediction columns, sorted by probability descending.

        Raises:
            ValueError: If model has not been trained yet.
        """

        if not self.metrics:
            raise ValueError("Model has not been trained. Call train() first.")

        X, _ = self._prepare_features()
        proba = self.pipe.predict_proba(X)[:, 1]

        scores = (
            pd.DataFrame(
                {
                    "CustomerNo": self.df["CustomerNo"].values,
                    "ChurnProbability": proba,
                    "ChurnPrediction": (proba > threshold).astype(int),
                }
            )
            .sort_values("ChurnProbability", ascending=False)
            .reset_index(drop=True)
        )

        self.logger.info(
            f"Scored {len(scores)} customers | "
            f"Predicted churners: {scores['ChurnPrediction'].sum()}"
        )

        return scores

    # Persistence

    def save(self, path: str) -> None:
        """
        Persist the trained pipeline to disk using joblib.

        Args:
            path (str): Destination file path (e.g. 'pipeline/ml/churn_model.pkl').

        Raises:
            ValueError: If model has not been trained yet.
        """

        if not self.metrics:
            raise ValueError("Model has not been trained. Call train() first.")

        joblib.dump(self.pipe, path)
        self.logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: str, dataframe: pd.DataFrame) -> "ChurnModel":
        """
        Load a previously saved pipeline and attach it to a new dataset.

        Bypasses training — the loaded pipeline is ready to call predict()
        immediately without retraining.

        Args:
            path (str): Path to the saved .pkl pipeline file.
            dataframe (pd.DataFrame): Dataset to score against.

        Returns:
            ChurnModel: Fully initialised instance ready for prediction.
        """

        pipe = joblib.load(path)
        instance = cls(dataframe=dataframe, pipe=pipe)
        instance.metrics = {"loaded": True}  # Bypass the untrained guard
        instance.logger.info(f"Pipeline loaded from {path}")

        return instance
    
    # Run entire pipeline
    def run_pipeline(self, threshold: float = 0.4) -> dict:
        """
        Execute the full model lifecycle in sequence.

        Steps:
            1. Train and evaluate
            2. Cross-validate
            3. Extract feature importances
            4. Score all customers

        Returns:
            dict: Results from each stage.
        """

        metrics = self.train(threshold=threshold)
        cv_results = self.cross_validate_model()
        importances = self.get_feature_importance()
        scores = self.predict(threshold=threshold)

        return {
            "metrics": metrics,
            "cv_results": cv_results,
            "feature_importance": importances,
            "churn_scores": scores,
        }
