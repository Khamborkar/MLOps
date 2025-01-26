import re
import subprocess
import joblib
import nltk
import pandas as pd
import mlflow
import mlflow.keras
import optuna
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Download required NLTK data
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

def clean_text(text):
    """Clean text data by removing unwanted characters and formatting."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

stemmer = PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words("english"))

def preprocess_text(text):
    """Preprocess text by stemming and removing stopwords."""
    words = text.split()
    processed_words = [
        stemmer.stem(word) for word in words if word not in stop_words
    ]
    return " ".join(processed_words)

def build_model(embedding_dim, lstm_units_1, lstm_units_2, dropout_rate):
    """Build and compile the LSTM model."""
    model = models.Sequential([
        Embedding(input_dim=10000, output_dim=embedding_dim, input_length=100),
        layers.LSTM(lstm_units_1, return_sequences=True),
        layers.LSTM(lstm_units_2),
        layers.Dropout(dropout_rate),
        Dense(3, activation="softmax")
    ])
    model.compile(
        optimizer = "adam",
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"]
    )
    return model

def objective(trial):
    """Optuna objective function to optimize hyperparameters."""
    embedding_dim = trial.suggest_int("embedding_dim", 64, 256)
    lstm_units_1 = trial.suggest_int("lstm_units_1", 64, 256)
    lstm_units_2 = trial.suggest_int("lstm_units_2", 64, 128)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = 3  # Fixed epochs for quick trials
    
    # Prepare data
    df = pd.read_csv("Tweets.csv")
    df["processed_text"] = (
        df["text"]
        .apply(clean_text)
        .apply(preprocess_text)
    )

    df["airline_sentiment_value"] = df["airline_sentiment"].map({
        "positive": 1,
        "negative": 0,
        "neutral": 2
    })

    X_train, X_temp, y_train, y_temp = train_test_split(
        df["processed_text"],
        df["airline_sentiment_value"],
        test_size = 0.3,
        random_state = 42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size = 0.5, random_state = 42
    )

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train)

    max_sequence_length = 100
    X_train = pad_sequences(
        tokenizer.texts_to_sequences(X_train),
        maxlen = max_sequence_length
    )
    X_val = pad_sequences(
        tokenizer.texts_to_sequences(X_val),
        maxlen = max_sequence_length
    )
    X_test = pad_sequences(
        tokenizer.texts_to_sequences(X_test),
        maxlen = max_sequence_length
    )

    model = build_model(embedding_dim, lstm_units_1, lstm_units_2, dropout_rate)
    early_stopping = EarlyStopping(
        monitor = "val_loss", patience=3,
        restore_best_weights=  True
    )

    # Train the model
    model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping]
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    # Log hyperparameters and metrics
    mlflow.log_param("embedding_dim", embedding_dim)
    mlflow.log_param("lstm_units_1", lstm_units_1)
    mlflow.log_param("lstm_units_2", lstm_units_2)
    mlflow.log_param("dropout_rate", dropout_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_metric("test_accuracy", test_accuracy)

    return test_accuracy

if __name__ == "__main__":
    # Start MLflow experiment
    mlflow.set_experiment("Sentiment Analysis")

    with mlflow.start_run():
        # Run Optuna for hyperparameter tuning
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)

        best_params = study.best_params
        print(f"Best hyperparameters: {best_params}")

        # Re-train the model with the best hyperparameters
        df = pd.read_csv("Tweets.csv")
        df["processed_text"] = (
            df["text"]
            .apply(clean_text)
            .apply(preprocess_text)
        )

        df["airline_sentiment_value"] = df["airline_sentiment"].map({
            "positive": 1,
            "negative": 0,
            "neutral": 2
        })

        X_train, X_temp, y_train, y_temp = train_test_split(
            df["processed_text"],
            df["airline_sentiment_value"],
            test_size = 0.3,
            random_state = 42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size = 0.5, random_state = 42
        )

        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(X_train)

        max_sequence_length = 100
        X_train = pad_sequences(
            tokenizer.texts_to_sequences(X_train),
            maxlen = max_sequence_length
        )
        X_val = pad_sequences(
            tokenizer.texts_to_sequences(X_val),
            maxlen = max_sequence_length
        )
        X_test = pad_sequences(
            tokenizer.texts_to_sequences(X_test),
            maxlen = max_sequence_length
        )

        # Build and train the model with best parameters
        model = build_model(
            best_params["embedding_dim"],
            best_params["lstm_units_1"],
            best_params["lstm_units_2"],
            best_params["dropout_rate"]
        )

        early_stopping = EarlyStopping(
            monitor = "val_loss", patience=3, restore_best_weights = True
        )

        history = model.fit(
            X_train, y_train,
            validation_data = (X_val, y_val),
            epochs=best_params["epochs"],
            batch_size = best_params["batch_size"],
            callbacks = [early_stopping]
        )

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        # Save model and tokenizer
        model.save("model.keras")
        joblib.dump(tokenizer, "tokenizer.pkl")

        # Log artifacts to MLflow
        mlflow.keras.log_model(model, "model")
        mlflow.log_artifact("tokenizer.pkl")

        # Save model versioning with DVC
        subprocess.run(["dvc", "add", "model.keras"], check=True)
        subprocess.run(["dvc", "add", "tokenizer.pkl"], check=True)
        subprocess.run(["git", "add",
                        "model.keras.dvc", "tokenizer.pkl.dvc"],
                       check=True)
        subprocess.run(["git",
                        "commit", "-m",
                        "Save best model and tokenizer with DVC"],
                       check=True)

        print("DVC and MLflow run completed.")
