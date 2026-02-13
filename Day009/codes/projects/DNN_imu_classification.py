import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
import har_utils
# 1. Load Data (Using helper from Part A)
X, y, le = har_utils.load_and_preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. Build DNN Model
def create_dnn(input_shape, num_classes):
    model = Sequential([
        # Flatten: Converts (50, 4) -> (200,)
        Flatten(input_shape=input_shape),

        # Dense Layers (Fully Connected)
        Dense(128, activation='relu'),
        Dropout(0.3),  # Regularization
        Dense(64, activation='relu'),

        # Output Layer
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model_dnn = create_dnn(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=y.shape[1])

# 3. Train
print("Training DNN...")
history_dnn = model_dnn.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 4. Evaluate & Visualize
har_utils.plot_metrics(model_dnn, history_dnn, X_test, y_test, le.classes_, "DNN")

# 5. Export
model_dnn.save("DNN_model.h5")
model_dnn.export("DNN_tf_model")  # Tensorflow SavedModel format
print("DNN Model Saved.")