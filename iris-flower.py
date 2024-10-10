from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Veri kümesini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeli oluştur (k = 5)
model = KNeighborsClassifier(n_neighbors=5)

# Modeli eğit
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Modelin doğruluğunu ölç
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy}")

# Karmaşıklık matrisini hesapla
conf_matrix = confusion_matrix(y_test, y_pred)

# Karmaşıklık matrisini görselleştir
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title("Karmaşıklık Matrisi")
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.show()
