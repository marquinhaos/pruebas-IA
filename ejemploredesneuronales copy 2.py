import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
#-------------------------------------------------------------#
#DATOS PARA EL ENTRENAMIENTO#
#-------------------------------------------------------------#
length=200; lenth=int(length)
ci=[0] * length
rs=[0] * length
for x in range(0, length):
    y=4*x-0.5
    ci[x]= float(x)
    rs[x]= float(y)

print(ci, rs)

#capa = tf.keras.layers.Dense(units=1, input_shape=[1])
#modelo = tf.keras.Sequential([capa])

#-------------------------------------------------------------#
#MODELO#
#-------------------------------------------------------------#
t1=time.time()

oculta1 = tf.keras.layers.Dense(units=50, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

t2=time.time()
print("MODELO EJECUTADO EN ")
print((t2-t1))
#-------------------------------------------------------------#
#ENTRENAMIENTO#
#-------------------------------------------------------------#
t1=time.time()
print("ENTRENAMIENTO INICIADO")
historial = modelo.fit(ci, rs, epochs=1000, verbose=False)
t2=time.time()

print("ENTRENAMIENTO FINALIZADO EN ")
t2=time.time()
print(t2-t1)
#-------------------------------------------------------------#
#GRAFICAMOS ERROR#
#-------------------------------------------------------------#
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

#-------------------------------------------------------------#
#PREDICCION#
#-------------------------------------------------------------#
while True:
    print("VARIABLE DE ENTRADA:")
    resultado=input(); resultado=float(resultado)
    if input=="OUT":
        raise ValueError('a estudiaaaaa')
    prediccion = modelo.predict([resultado])
    print("RESULTADO RED")
    print(str(prediccion))
    print("RESULTADO ANALITICO")
    anl=resultado*2+resultado*4+12
    print(anl)
    print("ERROR")
    print(anl-prediccion)

#print("Variables internas del modelo")
#print(capa.get_weights())
#print(oculta1.get_weights())
#print(oculta2.get_weights())
#print(salida.get_weights())