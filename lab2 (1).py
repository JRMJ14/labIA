#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ruta_del_archivo = 'https://raw.githubusercontent.com/JRMJ14/labIA/main/mod_credit.csv'
datos = pd.read_csv(ruta_del_archivo)



datos.dropna(inplace=True)


X = datos.iloc[:,0:] 
y = datos['Class'] 


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costo(theta, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta))
    J = -1/m * (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h)))
    return J


def descenso_gradiente(X, y, theta, alpha, iteraciones):
    m = len(y)
    costo_historia = []

    for _ in range(iteraciones):
        gradiente = 1/m * X.T.dot(sigmoid(X.dot(theta)) - y)
        theta -= alpha * gradiente
        costo_historia.append(costo(theta, X, y))

    return theta, costo_historia


theta = np.zeros(X_train.shape[1])


alpha = 0.01  
iteraciones = 1000


theta_optimo, costo_historia = descenso_gradiente(X_train, y_train, theta, alpha, iteraciones)


plt.plot(range(iteraciones), costo_historia)
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.title('Gráfico de costo')
plt.show()


predicciones_train = sigmoid(X_train.dot(theta_optimo))
predicciones_train[predicciones_train >= 0.5] = 1
predicciones_train[predicciones_train < 0.5] = 0
precision_train = np.mean(predicciones_train == y_train)
print("Precisión en el conjunto de entrenamiento:", precision_train)


predicciones_test = sigmoid(X_test.dot(theta_optimo))
predicciones_test[predicciones_test >= 0.5] = 1
predicciones_test[predicciones_test < 0.5] = 0
precision_test = np.mean(predicciones_test == y_test)
print("Precisión en el conjunto de prueba:", precision_test)

mapeo = {1: "si", 0: "no"}


datos['respuesta'] = datos['Class'].map(mapeo)

respuestafinal = datos[['respuesta']]

print(respuestafinal.head(10))


# In[ ]:




