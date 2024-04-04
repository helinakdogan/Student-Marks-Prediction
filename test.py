from sklearn.linear_model import LinearRegression
import pandas as pd

# Verileri yükle
df = pd.read_csv('student.csv')

# Veri çerçevesinin başını göster
print(df.head(3))

# Veri çerçevesinin genel bilgilerini göster
print(df.info())

# Bağımsız değişkenler (x) ve bağımlı değişken (y) seçimi
y = df[['Marks']]
x = df[['number_courses', 'time_study']]

# Modeli eğit
l = LinearRegression()
model = l.fit(x, y)

# Tahmin yap ve sonuçları göster
prediction = model.predict([[4, 4]])
print("Tahmin sonucu:", prediction)
