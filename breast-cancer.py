from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Veri kümesini yükle
data = load_breast_cancer()
X = data.data
y = data.target

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeli oluştur (kernel=linear ile düz bir SVM kullanıyoruz)
model = SVC(kernel='linear')

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
