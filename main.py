import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import optuna

# Load the dataset
df = pd.read_csv('genres_v2.csv')
df = df.drop(['title', 'song_name', 'analysis_url', 'track_href', 'uri', 'id', 'Unnamed: 0', 'type'], axis=1)
le = LabelEncoder()
df['genre'] = le.fit_transform(df['genre'])
X = df.drop(['genre'], axis=1)
y = df['genre']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'random_state': 42
    }

    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters and retrain the model
best_params = study.best_params
model = GradientBoostingClassifier(**best_params)
model.fit(X_train, y_train)

# Save the model, scaler, and encoder as pickle files
with open('genre_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('encoder.pkl', 'wb') as file:
    pickle.dump(le, file)

# Print the best accuracy achieved during optimization
best_accuracy = study.best_value
print("Best Accuracy: ", best_accuracy)
