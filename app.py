#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-NN para predecir enfermedad cardíaca usando edad y colesterol
"""

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# Importar el data set
dataset = pd.read_csv('heart.csv')
# Seleccionamos edad, colesterol y enfermedad cardíaca
X = dataset[['Age', 'Cholesterol']].values
y = dataset['HeartDisease'].values

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2)
classifier.fit(X_train, y_train)

# Predicción de un nuevo paciente (ejemplo: 45 años, colesterol 200)
nuevo_paciente = sc_X.transform([[45, 200]])
prediccion = classifier.predict(nuevo_paciente)
print(f"Predicción para paciente de 45 años con colesterol 200: {'Enfermedad cardíaca' if prediccion[0] == 1 else 'No tiene enfermedad'}")

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusión:")
print(cm)

# Calcular la precisión del modelo
from sklearn.metrics import accuracy_score
precision = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {precision:.2f}")

# Métricas de evaluación
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1])  # Para KNN es válido, retorna probabilidades

print(f"Precisión (precision): {precision:.2f}")
print(f"Recall (sensibilidad): {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"Área bajo la curva ROC (AUC): {auc:.2f}")


# Representación gráfica de los resultados del algoritmo en el Conjunto de Entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = sc_X.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 0.5),
                     np.arange(start = X_set[:, 1].min() - 10, stop = X_set[:, 1].max() + 10, step = 5))
plt.contourf(X1, X2, classifier.predict(sc_X.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = f"Enfermedad: {j}")
plt.title('K-NN para Enfermedad Cardíaca (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Colesterol')
plt.legend()
plt.show()

# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
X_set, y_set = sc_X.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 0.5),
                     np.arange(start = X_set[:, 1].min() - 10, stop = X_set[:, 1].max() + 10, step = 5))
plt.contourf(X1, X2, classifier.predict(sc_X.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = f"Enfermedad: {j}")
plt.title('K-NN para Enfermedad Cardíaca (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Colesterol')
plt.legend()
plt.show()