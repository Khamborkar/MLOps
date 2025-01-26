import os
import re
import subprocess
import joblib
import nltk
import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
import optuna  # Import Optuna for hyperparameter optimization

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
    """Build and compile the LSTM model with hyperparameters."""
    model = models.Sequential([
        Embedding(input_dim=10000, output_dim=embedding_dim, input_length=100),
        layers.LSTM(lstm_units_1, return_sequences=True),
        layers.LSTM(lstm_units_2),
        layers.Dropout(dropout_rate),
        Dense(3, activation="softmax")
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def objective(trial):
    """Objective function for Optuna optimization."""
    # Hyperparameters to optimize
    embedding_dim = trial.suggest_int('embedding_dim', 64, 128)
    lstm_units_1 = trial.suggest_int('lstm_units_1', 64, 128)
    lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 64)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    
    model = build_model(embedding_dim, lstm_units_1, lstm_units_2, dropout_rate)
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
    
    # Get validation accuracy to maximize
    val_accuracy = history.history['val_accuracy'][-1]
    
    # Log parameters and metrics
    mlflow.log_param('embedding_dim', embedding_dim)
    mlflow.log_param('lstm_units_1', lstm_units_1)
    mlflow.log_param('lstm_units_2', lstm_units_2)
    mlflow.log_param('dropout_rate', dropout_rate)
    mlflow.log_metric('val_accuracy', val_accuracy)
    
    return val_accuracy

# Create an Optuna study to optimize the hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Best hyperparameters found
print(f"Best trial: {study.best_trial.params}")
