ANÁLISIS DE SCIKIT-LEARN Y STATSMODELS PARA MODELOS DE REGRESIÓN LINEAL

Scikit-Learn:

Scikit-learn es una de las librerías de machine learning más utilizadas y documentadas en Python. Es ampliamente recomendada para modelos sencillos y que no requieren de un análisis matemático muy detallado o extenderse a otras áreas de la IA (deeplearning, computer vision...) por su intuitiva API, amplia comunidad y compatibilidad con otras librerías.
Pros:
-Gran cantidad de fuentes de informacion y documentación bien explicada
-Compatibilidad con pandas, librería que ya estamos usando
-API sencilla, código limpio
-Cálculos "in situm" rápidos y eficaces, idel para dahsboards interactivas
Contras:
-Menos escalabilidad si queremos modelos/análisis más complejos (no procede para el proyecto)


Statsmodels:

Librería enfocada en el análisis y diagnóstico estadístico de un modelo de aprendizaje automático.
Pros:
-Ofrece un repertorio de cálculos y métricas mayor que sklearn
Contras:
-Código ligeramente más complejo que scikit-learn
-Menor velocidad de cómputo
-Serialización más costosa


CONCLUSIÓN:
Teniendo en cuenta que para nuestro proyecto ponderan mucho más la velocidad y la
sencillez sobre la extension del analisis estadístico del modelo, y con vistas
a la serialización en posteriores tareas, Scikit-learn se alza como la mejor opción.
Con aprender unas pocas librerías es muy sencillo preparar todo el pipeline, desde
el preprocesado hasta las predicciones finales, pasando por normalización de features.
Recomendado familiarizarse con numpy y pandas para manejo de datos con sklearn.