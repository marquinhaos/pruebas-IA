import tensorflow as tf
import numpy as np

# Crear datos de entrenamiento
x_train = np.array([-1, 0, 1, 2, 3, 4], dtype=float)
y_train = np.array([1, 0, 1, 4, 9, 16], dtype=float)

# Crear modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

# Compilar modelo
model.compile(optimizer=tf.optimizers.Adam(1),
              loss='mean_squared_error')

# Entrenar modelo
model.fit(x_train, y_train, epochs=500)

# Hacer predicciones
print(model.predict([5]))