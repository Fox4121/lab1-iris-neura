# 1. Імпортуємо необхідні бібліотеки
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Завантажимо дані
iris = pd.read_csv("C:/Users/Сергей/Downloads/iris.csv")

# 3. Перевіримо дані
print(iris.head())
print(iris.info())
print(iris['Species'].value_counts())

# 4. Візуалізація
sns.pairplot(iris.drop(['Id'], axis=1), hue='Species')
plt.show()

# KDE-графік для Iris-setosa (довжина і ширина чашолистка)
sub = iris[iris['Species'] == 'setosa']

sns.kdeplot(
    x=sub['SepalLengthCm'],
    y=sub['SepalWidthCm'],
    cmap="plasma",
    fill=True,
    thresh= 0.01  # можна підвищити, якщо значень мало
)

plt.title('Iris-setosa Sepal')
plt.xlabel('Sepal Length Cm')
plt.ylabel('Sepal Width Cm')
plt.show()


# 5. Підготовка даних
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Видаляємо непотрібні стовпці
X = iris.drop(['Id', 'Species'], axis=1)
y = iris['Species']

# Перетворюємо мітки на числові значення
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Розділення на train і test
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# 6. Створення моделі нейронної мережі
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 класи

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 7. Навчання моделі
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=1)

# 8. Оцінка моделі
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Точність моделі на тестовій вибірці: {accuracy * 100:.2f}%')

# 9. Побудова графіку точності
plt.plot(history.history['accuracy'], label='Тренувальна точність')
plt.plot(history.history['val_accuracy'], label='Валідаційна точність')
plt.title('Точність навчання моделі')
plt.xlabel('Епохи')
plt.ylabel('Точність')
plt.legend()
plt.show()

# 10. Перевірка на нових даних
example = np.array([[5.1, 3.5, 1.4, 0.2]])  # Значення для прикладу
prediction = model.predict(example)
predicted_class = le.inverse_transform([np.argmax(prediction)])
print(f"Прогнозована категорія: {predicted_class[0]}")
