# %% [markdown]
"""
# Ejercicio Módulo 5 - Dataset Swiss
**Inteligencia Artificial - CEIA - FIUBA**

**INSERTE AQUÍ SU NOMBRE**

Para aprender sobre regresión, vamos a utilizar un dataset clásico llamado Swiss, que proviene originalmente del lenguaje R. Este dataset contiene datos socioeconómicos de 47 provincias suizas a fines del siglo XIX. Cada fila representa una provincia, y las variables reflejan características demográficas y sociales relevantes para ese contexto histórico.

## Variables

- `Location`: Provincia donde se midieron los datos.
- `Fertility`: Tasa de fertilidad (número promedio de hijos por mujer)
- `Agriculture`:` Porcentaje de hombres ocupados en agricultura
- `Examination`: Porcentaje de hombres que completaron exámenes de educación superior
- `Education`: Nivel promedio de educación (escala arbitraria)
- `Catholic`: Porcentaje de población católica
- `Infant.Mortality`: Tasa de mortalidad infantil (por cada 1000 nacidos vivos)

## Que queremos predecir?

Vamos a utilizar este dataset para predecir la tasa de fertilidad en cada provincia mediante diferentes métodos de regresión.

--- 

Siguiendo el procedimiento típico de Machine Learning, vamos a leer los datos y separarlos en los datasets de entrenamiento y testeo utilizando Scikit-Learn...
"""

# %%
import pandas as pd

df = pd.read_csv("swiss.csv")

df.head()

# %%
print(f"Tenemos {df.shape[0]} observaciones")

# %% [markdown]
"""
Obtenemos la variable objetivo (`Fertility`) y, por otro lado, los atributos (quitamos `Location` ya que no es un atributo numérico relevante para la regresión)
"""

# %%
X = df.drop(["Fertility", "Location"], axis=1)
y = df["Fertility"]

# %% [markdown]
"""
Dado que tenemos pocas observaciones, vamos a separar el dataset en un 50% para entrenamiento y 50% para testeo:
"""

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# %% [markdown]
"""
## Regresión lineal múltiple

Arranquemos la primera parte del ejercicio. Para eso, vamos a entrenar un modelo de regresión lineal múltiple usando todos los atributos. Para ello debes:

1. Escalar los atributos usando `StandardScaler`
2. Entrenar el modelo usando el dataset de entrenamiento.
3. Obtener las predicciones sobre el dataset de testeo.
4. Calcular las métricas MAE, MSE  y $R^2$, e imprimir los resultados.
"""

# %%
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

##### COMPLETAR AQUI LO PEDIDO
pipeline_linear = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

pipeline_linear.fit(X_train, y_train)
y_pred = pipeline_linear.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Regresión Lineal Múltiple")
print("-------------------------")
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"R2: {r2:.3f}")

# %% [markdown]
"""
## Modelo con regularización

Para mejorar nuestro modelo, vamos a explorar técnicas de regresión lineal con regularización, que nos permiten controlar el sobreajuste y seleccionar variables relevantes automáticamente.

Existen dos variantes muy populares:

- Una penaliza la suma de los cuadrados de los coeficientes (regularización L2).
- La otra penaliza la suma del valor absoluto de los coeficientes (regularización L1).

Ambas ayudan a mejorar la generalización, pero una de ellas además puede eliminar variables (coeficientes exactamente cero), lo que ayuda a identificar qué atributos son realmente importantes.

Tu tarea:

1. Elegí correctamente cuál de los dos métodos de regularización usar para este problema. 
    - Pista: Queremos que el modelo sea capaz de hacer una selección automática de variables, dejando fuera aquellas que no aportan.
2. Implementá un pipeline que incluya escalado y el modelo elegido.
3. Buscá automáticamente el mejor valor del hiperparámetro de regularización (alpha) usando validación cruzada usando 3-folds.
4. Entrená el modelo con los datos de entrenamiento y obtené las predicciones para el set de testeo.
5. Calcular las métricas MAE, MSE  y $R^2$, e imprimir los resultados.
6. Imprimí los coeficientes resultantes e identificá qué variables fueron eliminadas (coeficiente = 0).
"""

# %%
import numpy as np

from sklearn.linear_model import LassoCV, RidgeCV

# %% [markdown]
"""
Paremos un momento para entender qué hacen LassoCV y RidgeCV antes de continuar con la resolución:

> Tanto `LassoCV` como `RidgeCV` son implementaciones de regresión lineal con regularización que incluyen la búsqueda automática del mejor hiperparámetro alpha mediante validación cruzada.
>
> Ambos métodos prueban distintos valores de alpha y eligen el que minimiza el error del modelo, facilitando el proceso de ajuste sin necesidad de una búsqueda manual.
>
> Internamente, utilizan la métrica del error cuadrático medio (MSE) para evaluar el rendimiento del modelo en cada fold de la validación cruzada.
>
> Por ejemplo, si llamás a RidgeCV(alphas=alphas, cv=5), se hará una validación cruzada de 5 folds utilizando los valores de alpha que vos le pases, y se seleccionará el que obtenga el menor MSE promedio.
> 
> Una vez elegido el mejor alpha, el modelo final se entrena con todos los datos de entrenamiento usando ese valor.

¡Listo! Con todo lo que vimos hasta ahora, ya estás en condiciones de resolver esta parte y completar los 6 puntos propuestos
"""

# %%
alphas = np.logspace(-4, 1, 500)

##### COMPLETAR AQUI LO PEDIDO
pipeline_lasso = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', LassoCV(alphas=alphas, cv=3, random_state=42))
])

pipeline_lasso.fit(X_train, y_train)
y_pred_lasso = pipeline_lasso.predict(X_test)

mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

best_alpha = pipeline_lasso.named_steps['regressor'].alpha_
coefs = pipeline_lasso.named_steps['regressor'].coef_
features = X.columns

print("Modelo Regularizado (Lasso)")
print("---------------------------")
print(f"Mejor valor de alpha: {best_alpha:.4f}")
print(f"MAE: {mae_lasso:.3f}")
print(f"MSE: {mse_lasso:.3f}")
print(f"R2: {r2_lasso:.3f}\n")

print("Coeficientes obtenidos:")
for feat, coef in zip(features, coefs):
    print(f"- {feat}: {coef:.4f}")

eliminadas = [feat for feat, coef in zip(features, coefs) if coef == 0]
print(f"\nVariables eliminadas (coeficiente = 0): {eliminadas}")



# %% [markdown]
"""
## Comparación de modelos y conclusiones

Completá la siguiente tabla con las métricas obtenidas para cada uno de los modelos que entrenaste:

| Modelo                        | MAE   | MSE    | $R^2$ |
| ----------------------------- | ----- | ------ | ----- |
| Regresión Lineal              | 6.094 | 64.457 | 0.574 |
| Modelo Regularizado (Lasso)   | 5.995 | 64.294 | 0.576 |


> ⚠️ Asegurate de cambiar el nombre del modelo `Modelo Regularizado (L1 o L2)` según el modelo que usaste (Lasso o Ridge).

### Justificación

**¿Cuál de los modelos te parece que tuvo un mejor desempeño general?**

Tené en cuenta las tres métricas al responder, y también pensá en la complejidad del modelo (por ejemplo, si eliminó variables innecesarias).

Escribí tu respuesta a continuación:
"""

# %% [markdown]
"""
El modelo Lasso tuvo un mejor desempeño.
Logró reducir el error (tanto el MAE como el MSE) y mejorar ligeramente el
$R^2$.
Además, simplificó el modelo al descartar por completo la variable Agriculture
(dejando su coeficiente en 0), lo que nos da un modelo más sencillo sin perder
precisión.
"""
