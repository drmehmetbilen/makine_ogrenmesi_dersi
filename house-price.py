from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# Sahte veri kümesi oluştur (özellik sayısı 1 olacak şekilde)
X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeli oluştur
model = LinearRegression()

# Modeli eğit
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Ortalama karesel hatayı hesapla
mse = mean_squared_error(y_test, y_pred)
print(f"Ortalama Karesel Hata: {mse}")