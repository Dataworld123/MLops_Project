import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")
# Suppress MLflow artifact download warnings
# os.environ["MLFLOW_DISABLE_ARTIFACTS_DOWNLOAD"] = "1"

# Set MLflow Tracking URI & DAGsHub integration
MLFLOW_TRACKING_URI = "https://dagshub.com/nishantsingh801102/MLops_Project.mlflow"
dagshub.init(repo_owner="nishantsingh801102", repo_name="MLops_Project", mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("LoR Hyperparameter Tuning")


# ==========================
# Text Preprocessing Functions
# ==========================
def preprocess_text(text):
    """Applies basic text preprocessing steps without NLTK dependencies."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Basic stopwords removal (most common English stopwords)
    basic_stopwords = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
                      'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
                      'that', 'the', 'to', 'was', 'were', 'will', 'with'}
    
    words = text.split()
    words = [word for word in words if word not in basic_stopwords]
    
    return ' '.join(words).strip()


# ==========================
# Load & Prepare Data
# ==========================
def load_and_prepare_data(filepath):
    """Loads, preprocesses, and vectorizes the dataset."""
    df = pd.read_csv(filepath)
    
    # Apply text preprocessing
    df["review"] = df["review"].astype(str).apply(preprocess_text)
    
    # Filter for binary classification
    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})
    
    # Convert text data to TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Limit features to reduce complexity
        min_df=5,          # Ignore terms that appear in less than 5 documents
        max_df=0.95        # Ignore terms that appear in more than 95% of documents
    )
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer


# ==========================
# Train & Log Model
# ==========================
def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):
    """Trains a Logistic Regression model with GridSearch and logs results to MLflow."""
    
    param_grid = {
        "C": [0.1, 1, 15],
        "penalty": ["l2"],  # Simplified to just l2 penalty
        "solver": ["liblinear"]
    }
    
    with mlflow.start_run():
        grid_search = GridSearchCV(
            LogisticRegression(max_iter=1000), 
            param_grid, 
            cv=10, 
            scoring="f1", 
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Log all hyperparameter tuning runs
        for params, mean_score, std_score in zip(
            grid_search.cv_results_["params"], 
            grid_search.cv_results_["mean_test_score"], 
            grid_search.cv_results_["std_test_score"]
        ):
            with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
                model = LogisticRegression(**params, max_iter=1000)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }
                
                # Log parameters & metrics
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        # Log the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"\nBest Params: {best_params} | Best F1 Score: {best_f1:.4f}")


# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data("notebooks/data.csv")
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)

