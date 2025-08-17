from sklearn.model_selection import train_test_split
from preprocess import load_and_preprocess
from create_sequences import make_sequences
from model import build_model
import matplotlib.pyplot as plt

# Load data
X, y, le = load_and_preprocess("/content/dataset")

# Make sequences
X_seq, y_seq = make_sequences(X, y, seq_len=5)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build model
model = build_model(num_classes=y_seq.shape[1])

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=16
)

# Save model
model.save("fruit_shelf_life_model.h5")

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("CNN+LSTM Training")
plt.show()
