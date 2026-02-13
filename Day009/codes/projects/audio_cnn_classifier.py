import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import prepare_data
# 1. Load Data
# (Assuming you run the load_data() function from Part 1)
X, y, classes = prepare_data.load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN
X_train_cnn = prepare_data.prepare_for_cnn(X_train)
X_test_cnn = prepare_data.prepare_for_cnn(X_test)
input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2], 1)

# 2. Build CNN
model_cnn = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. Train
print("Training CNN...")
history_cnn = model_cnn.fit(X_train_cnn, y_train, epochs=20, batch_size=32, validation_data=(X_test_cnn, y_test))

# 4. Save
model_cnn.save("speech_cnn_model.h5")

# Execute Plots
prepare_data.plot_evaluation(model_cnn, X_test_cnn, y_test, history_cnn, "CNN")