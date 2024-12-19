# training.py
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Parameters
max_features = 20000  # Number of words to consider as features
maxlen = 200          # Cut reviews after 200 words

# Load IMDb dataset
print("Loading IMDb dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to the same length
print("Padding sequences...")
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Check the shapes
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")

# Build the model with Dropout layers
print("Building the model...")
model = Sequential([
    Embedding(input_dim=max_features, output_dim=128, input_length=maxlen),
    LSTM(64, return_sequences=True),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))
])

# Compile the model
print("Compiling the model...")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Display the model's architecture
model.summary()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)

# Train the model with callbacks
print("Starting training...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=512,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the model on the test set
print("Evaluating the model on test data...")
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the trained model
print("Saving the model to 'best_model.h5'...")
model.save('best_model.h5')
print("Model saved successfully!")
