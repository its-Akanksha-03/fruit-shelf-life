from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess
from create_sequences import make_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Load data
X, y, le = load_and_preprocess("/content/dataset")
X_seq, y_seq = make_sequences(X, y, seq_len=5)

X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Load trained model
model = load_model("fruit_shelf_life_model.h5")

# Evaluate
loss, acc = model.evaluate(X_val, y_val, verbose=1)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {acc:.4f}")

# Classification report
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

print(classification_report(y_true_classes, y_pred_classes, target_names=le.classes_))
