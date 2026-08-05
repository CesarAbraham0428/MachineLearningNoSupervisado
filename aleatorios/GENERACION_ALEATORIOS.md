# Generación de aleatorios Big Five

## Propósito

Crear perfiles sintéticos para experimentar con el flujo de análisis sin modificar las respuestas originales. El resultado se limita a las cinco dimensiones Big Five y no agrega carrera, grupo, fecha ni etiquetas de perfil.

## Datos de entrada

El servicio recibe un `DataFrame` con las cinco dimensiones Big Five. Cada valor debe ser numérico y estar dentro de la escala de 1 a 5. Se requieren al menos dos perfiles originales para calcular la desviación estándar.

## Método de generación

Para cada dimensión se calcula, únicamente con los perfiles originales:

- Media muestral.
- Desviación estándar muestral.

Después se generan los nuevos valores de cada dimensión de manera independiente mediante una distribución normal:

`valor_sintético ~ Normal(media_del_rasgo, desviación_del_rasgo)`

Los valores fuera del intervalo de 1 a 5 se descartan y se vuelven a muestrear. Al final, cada valor se redondea a dos decimales.

## Resultado

El archivo combinado contiene primero los registros sintéticos y después los originales. Solo incluye estas cinco columnas:

- Extraversión
- Estabilidad emocional
- Apertura a la experiencia
- Responsabilidad
- Amabilidad

Los originales no se alteran. La generación no selecciona una fila semilla, no busca perfiles lejanos y no fuerza subconjuntos o clústeres.

## Reproducibilidad técnica

La función admite opcionalmente un número `semilla` para que las pruebas automáticas puedan repetir exactamente una ejecución. No representa un perfil base y no es una opción expuesta en la interfaz.

## Alcance e interpretación

Los datos sintéticos conservan la media y la dispersión observadas por rasgo, pero no preservan necesariamente relaciones entre rasgos. Deben utilizarse para demostraciones y pruebas técnicas; los resultados sobre la encuesta deben basarse en los registros reales.
