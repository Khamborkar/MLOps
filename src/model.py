import subprocess
import pandas as pd
import optuna
import mlflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


# Prepare text data for the model
def preprocess_text(data, max_words=10000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    X = pad_sequences(sequences, maxlen=max_len)
    return X, tokenizer


# Build the model
def build_model(embedding_dim, lstm_units_1, lstm_units_2, dropout_rate):
    model = Sequential()
    model.add(Embedding(10000, embedding_dim))  # Vocabulary size is set to 10k
    model.add(LSTM(lstm_units_1, return_sequences=True))
    model.add(LSTM(lstm_units_2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Optuna objective function for hyperparameter tuning
def objective(trial):
    embedding_dim = trial.suggest_int("embedding_dim", 64, 256)
    lstm_units_1 = trial.suggest_int("lstm_units_1", 64, 256)
    lstm_units_2 = trial.suggest_int("lstm_units_2", 64, 128)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = 3  # Fixed number of epochs for quick trials
    
    # Split the data into training and validation sets
    # Load the dataset
    data = load_data("data/dataset.csv")
    X, tokenizer = preprocess_text(data['text'])
    y = data['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train the model
    model = build_model(embedding_dim, lstm_units_1, lstm_units_2, dropout_rate)
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate the model on validation data
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    # Log the hyperparameters and the results to MLflow
    mlflow.log_param("embedding_dim", embedding_dim)
    mlflow.log_param("lstm_units_1", lstm_units_1)
    mlflow.log_param("lstm_units_2", lstm_units_2)
    mlflow.log_param("dropout_rate", dropout_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_metric("val_accuracy", val_accuracy)
    
    return val_accuracy


# Main function to load data, run Optuna, and save model
def main():
    # Set up MLflow
    mlflow.start_run()

    # Load the dataset
    data = load_data("data/dataset.csv")
    X, tokenizer = preprocess_text(data['text'])
    y = data['label']  # Assuming 'label' is the target variable

    # Run Optuna for hyperparameter tuning
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    # Build and train the final model
    final_model = build_model(
        best_params['embedding_dim'],
        best_params['lstm_units_1'],
        best_params['lstm_units_2'],
        best_params['dropout_rate']
    )

    # Train the model with best hyperparameters
    final_model.fit(X, y, epochs=3, batch_size=best_params['batch_size'])
    
    # Save the trained model using DVC
    final_model.save("model.keras")
    subprocess.run(["dvc", "add", "model.keras"], check=True)
    subprocess.run(["git", "add", "model.keras.dvc"], check=True)
    subprocess.run(["git", "commit", "-m", "Save best model after Optuna tuning"], check=True)
    
    # Log the model artifact in MLflow
    mlflow.keras.log_model(final_model, "model")

    mlflow.end_run()


if __name__ == "__main__":
    main()
