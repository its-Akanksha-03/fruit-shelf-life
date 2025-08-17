import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_and_preprocess(base_path="/content/dataset"):
    X, y = [], []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                img = cv2.resize(img, (128,128))
                img = img.astype("float32") / 255.0
                X.append(img)

                label = os.path.basename(root)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    return X, y_categorical, le
