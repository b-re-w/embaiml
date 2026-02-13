import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import har_utils
# 1. Load Data
X, y, le = har_utils.load_and_preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. Build CNN Model
def create_cnn(input_shape, num_classes):
    model = Sequential([
        # Conv Layer 1: Learn 64 different features (filters)
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),

        # Conv Layer 2
        Conv1D(filters=64, kernel_size=3, activation='relu'),

        # Pooling: Downsample to reduce complexity
        MaxPooling1D(pool_size=2),
        Dropout(0.5),

        # Flatten for classification
        Flatten(),
        Dense(100, activation='relu'),

        # Output
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model_cnn = create_cnn(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=y.shape[1])

# 3. Train
print("Training CNN...")
history_cnn = model_cnn.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 4. Evaluate & Visualize
har_utils.plot_metrics(model_cnn, history_cnn, X_test, y_test, le.classes_, "CNN")

# 5. Export
model_cnn.save("CNN_model.h5")
model_cnn.export("CNN_tf_model")
print("CNN Model Saved.")