from sklearn.model_selection import train_test_split
from cnn_model import build_url_cnn_model
import numpy as np

# X = tes données (séquences)
# y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = build_url_cnn_model(max_words=5000, max_len=200)

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

# Évaluation finale
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# Sauvegarde du modèle
model.save("cnn_model.h5")