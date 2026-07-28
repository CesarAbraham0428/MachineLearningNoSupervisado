# ClusterLab

Proyecto de extracción de conocimiento en bases de datos y clustering.

## Estructura del proyecto

- `views/`: Vistas de la aplicación.
- `services/`: Servicios de lógica de negocio y procesamiento.
- `utils/`: Utilidades generales y validaciones.
- `database/`: Manejo de base de datos.
- `storage/`: Almacenamiento de modelos y reportes.
- `assets/`: Estilos CSS y recursos estáticos.
- `tests/`: Pruebas unitarias.

Descripción de la aplicación

ClusterLab es una aplicación web desarrollada en Python con Streamlit que permite realizar análisis exploratorio de datos y entrenar modelos de aprendizaje no supervisado de forma sencilla e interactiva. Su objetivo es facilitar el proceso de carga, preparación, análisis y agrupamiento de conjuntos de datos sin necesidad de conocimientos avanzados en programación.

La aplicación está orientada a estudiantes, investigadores y usuarios que desean experimentar con técnicas de clustering para descubrir patrones, segmentar datos y obtener información relevante a partir de un conjunto de datos. Todo el flujo de trabajo se realiza desde una única interfaz, donde el usuario puede cargar un archivo, analizarlo, entrenar un modelo, visualizar los resultados y guardar el modelo entrenado para utilizarlo posteriormente.

Resumen de funcionalidades

La aplicación permitirá al usuario realizar las siguientes actividades:

Cargar conjuntos de datos en formato CSV o Excel para su análisis.
Visualizar el contenido del dataset mediante una tabla interactiva.
Filtrar la información por categorías o columnas seleccionadas para trabajar únicamente con un subconjunto de los datos.
Exportar los datos filtrados a un nuevo archivo de Excel.
Generar estadísticas descriptivas del conjunto de datos, como medias, medianas, desviaciones estándar, valores mínimos y máximos, distribución de datos, valores nulos y registros duplicados.
Visualizar gráficos estadísticos que faciliten la interpretación del conjunto de datos antes del entrenamiento.
Entrenar un modelo de aprendizaje no supervisado (K-Means) seleccionando las variables que participarán en el proceso y configurando el número de clústeres.
Visualizar los resultados del entrenamiento, incluyendo la asignación de cada registro a un clúster, métricas de evaluación como la inercia y el índice de Silhouette, así como representaciones gráficas de los grupos encontrados.
Generar un reporte en formato PDF con la información estadística y los resultados obtenidos durante el entrenamiento.
Guardar modelos entrenados, almacenando tanto el modelo como sus metadatos, entre ellos el conjunto de datos utilizado, columnas seleccionadas, parámetros del algoritmo, métricas obtenidas y fecha de creación.
Administrar los modelos almacenados, permitiendo consultar la información de cada modelo y reutilizarlo posteriormente con nuevos conjuntos de datos compatibles.

En conjunto, la aplicación implementa un flujo completo de análisis de datos y aprendizaje no supervisado: desde la carga del dataset hasta la generación de resultados y la persistencia de modelos entrenados, todo integrado en una única interfaz web fácil de utilizar.

#0B1220  Fondo principal
#111827  Fondo secundario
#162033  Tarjetas
#1A2436  Tablas
#1F2A40  Hover
#2A3953  Bordes

#2563EB  Azul principal
#3B82F6  Azul hover
#60A5FA  Links

#A855F7  Morado
#22C55E  Verde
#F59E0B  Naranja
#FBBF24  Amarillo
#EF4444  Rojo

#F8FAFC  Texto principal
#CBD5E1  Texto secundario
#94A3B8  Texto terciario
#64748B  Placeholder