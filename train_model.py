import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load Data
data = pd.read_csv('hand_data.csv')

# 2. Separate Features and Labels
X = data.iloc[:, 1:].values   # landmark coordinates
y = data.iloc[:, 0].values   # labels (A, B, C, space...)

# 3. Encode Labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 5. Build Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train Model
print("Training model...")
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# 7. Save Model
model.save('sign_language_model.keras')
print("✅ Model saved successfully!")