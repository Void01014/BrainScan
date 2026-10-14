from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import plot_model


def build_model(input_shape=(224, 224, 3), num_classes=4, lr=1e-3):
    """Build and compile the baseline CNN for the MRI classes."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_architecture(model, path="docs/model_architecture.png"):
    """Save a diagram of the model; requires pydot and system Graphviz."""
    plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)
