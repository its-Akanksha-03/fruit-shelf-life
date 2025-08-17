from tensorflow.keras import layers, models

def build_model(num_classes):
    model = models.Sequential([
        layers.TimeDistributed(layers.Conv2D(32,(3,3),activation='relu'), input_shape=(5,128,128,3)),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Conv2D(64,(3,3),activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Flatten()),

        layers.LSTM(64, return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
