# ANÁLISIS DE PICKLE (DILL) Y JOBLIB PARA PERSISTENCIA DE MODELOS

## PICKLE/DILL:

Pickle es el módulo estándar de python para la serialización de objetos Python,
es compatible con casi cualquiera (listas, arrays, clases, modelos).
Dill es una extensión de pickle que permite serializar decoradores/funciones
lambda pero a cambio de menor velocidad

Pros:
-Fácil de usar
-Biblioteca estándar de Python
-Compatible con casi cualquier tipo de objeto Python
Contras:
-Menos eficiente con archivos de gran tamaño (como algunos modelos/dataframes)
-No portable entre versiones muy distintas de sklearn


## JOBLIB:

Joblib es la librería de persistencia de objetos Python creada por los propios
desarrolladores de scikit-learn más enfocada en serializar eficientemente
objetos grandes (sobre todo numpy/pandas).

Pros:
-Más rápido y compacto que Pickle
-API simple
-100% compatible con sklearn
Contras:
-No portable fuera de Python


CONCLUSIÓN:
Joblib es, sin lugar a dudas, la mejor opción para serializar/deserializar los
modelos que se entrenen en nuestra app, pues está diseñada por los mismos 
desarrolladores de scikit y la recomiendan en su documentación oficial.
La experiencia de uso de la API es fluida y simple. Además, en el test que hice,
incluí, a parte de la Pipeline en el paquete .joblib, las features y target de test
para no tener que leer el csv cada vez que se ejecute el código.
Si probáis a ejecutarlo antes y despúes de haberlo serializado veréis la 
diferencia de tiempos al final. 