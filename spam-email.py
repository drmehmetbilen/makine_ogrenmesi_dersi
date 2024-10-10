from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Sahte veri kümesi oluştur
X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeli oluştur
model = LogisticRegression()

# Modeli eğit
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Modelin doğruluğunu ölç
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy}")
