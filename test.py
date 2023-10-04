import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

# Создаем модель
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Компилируем модель
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Выводим архитектуру модели
model.summary()

# Подготовка данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
    'путь_к_папке_с_тренировочными_изображениями',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'путь_к_папке_с_тестовыми_изображениями',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Обучение модели
model.fit(
    train_set,
    epochs=10,
    validation_data=test_set
)

# Загрузка и предобработка тестового изображения
test_image = image.load_img('путь_к_тестовому_изображению', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image /= 255.

# Предсказание пола
result = model.predict(test_image)

if result[0][0] > 0.5:
    prediction = 'Женский'
else:
    prediction = 'Мужской'

print(f'Результат: {prediction}')
