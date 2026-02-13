import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import har_utils
# 1. Load Data
X, y, le = har_utils.load_and_preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. Build LSTM Model
def create_lstm(input_shape, num_classes):
    model = Sequential([
        # LSTM Layer: 64 units
        # return_sequences=False because we only need the final output for classification
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.5),

        # Dense Classification Layers
        Dense(64, activation='relu'),

        # Output
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model_lstm = create_lstm(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=y.shape[1])

# 3. Train
print("Training LSTM...")
history_lstm = model_lstm.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 4. Evaluate & Visualize
har_utils.plot_metrics(model_lstm, history_lstm, X_test, y_test, le.classes_, "LSTM")

# 5. Export
model_lstm.save("LSTM_model.h5")
model_lstm.export("LSTM_tf_model")
print("LSTM Model Saved.")