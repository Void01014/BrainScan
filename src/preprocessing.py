from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def encode_labels(labels):
    """str labels -> ints (for stratify) + LabelEncoder (for decoding at inference)."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, le


def split_data(images, y, test_size=0.15, val_size=0.15, seed=42):
    """Stratified train/val/test split. Split BEFORE normalizing or augmenting."""
    X_train, X_test, y_train, y_test = train_test_split(
        images, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size / (1 - test_size),
        stratify=y_train, random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize(x):
    """uint8 [0,255] -> float32 [0,1]. Reused as-is at inference."""
    return x.astype("float32") / 255.0


def one_hot(y, num_classes):
    return to_categorical(y, num_classes=num_classes)