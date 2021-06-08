# Multimodal sentiment analysis

El campo de aprendizaje automático se diversifica cada vez más, alcanzando a cubrir nuevas 
áreas como lo es el de la salud mental. Temática que ha cobrado relevancia en tiempos cercanos, 
sobre todo en temas de depresión, ansiedad o ayuda psicológica y psiquiátrica. 
El proyecto busca alcanzar el punto donde se encuentran ambos rubros, teniendo un modelo de 
inteligencia que pueda recibir videos como entrada y que pueda clasificar las emociones que 
se denotan en el mismo, con la idea de que en un futuro se pueda usar probablemente como 
una herramienta para los expertos en salud mental y que puedan hacer diagnósticos con más información. 
Este modelo lo que busca también es tener en consideración todos los aspectos del paciente 
o persona que aparezca en el video, analizando su voz, el contenido de su discurso y sus expresiones faciales; 
Esto con la idea de que las inferencias tengan la mayor cantidad de atributos disponibles al ejecutarse.

## Uso del modelo

El estado actual del proyecto consiste en analizar un video y hacer tres inferencias diferentes
tomando en cuenta las características mencionadas de: audio, texto e imágen. Las inferencias
pueden realizarse de manera independiente, pero dentro del repositorio se facilita hacerlas sobre
un mismo medio.

El resultado de las inferencias sobre un video se resume de la siguiente manera:

- ASR - Transcripción del audio dentro del video.
- Texto - Análisis de la transcripción, luego clasificación.
- Audio - Análisis de las características del audio, luego clasificación.
- Imagen - Análisis de cada cuadro del video, generación de video de salida con los puntos faciales.


### Demo

Para correr el demo se debe ingresar al notebook `ColabCentral.ipynb` y ejecutar todas las celdas. 
El repositorio incluye un video de demostración.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ieGuDje5AgOQ1sgbqv9UvSmGTW_ZTEL0?usp=sharing)

### Contenido personalizado

Si se desea hacer una inferencia sobre un audio o video propio del usuario:

1. Ejecutar todas las celdas hasta la sección de ASR para descargar las dependencias y los pesos que se usarán.
2. Hacer uso de las funciones `import_audio`, `import_video`, según sea el caso y las funciones `transcribe` como lo indica el notebook.
3. En la sección de M11 cambiar dentro de la línea:
    ```python
    path_to_video = Path('../', asr_demo_result[0]['audio_path'])
    ```
    `asr_demo_result` por el nombre de la variable que recibió el resultado de la función `transcribe`
4. Continuar ejecutando las celdas hasta la penúltima dentro de la sección de **Inferencias de texto** donde se debe hacer un cambio similar al anterior:
    ```python
    x = predictions(asr_demo_result[0]['transcription'], glove, fasttext)
    ```
    `asr_demo_result` por el nombre de la variable que recibió el resultado de la función `transcribe`
    
5. Finalmente, dentro de la sección de DETR cambiar el valor de la variable `INPUT_PATH` por:
   `<nombre de la variable que recibió la función transcribe>[0]['video_path']`

***

### Sobre este repositorio

* Se encuentran en carpetas por separado los distintos modelos que se utilizaron para esta implementación conjunta, 
cada uno cuenta con una explicación más detallada en el readme que se encuentra dentro de cada carpeta.

Video usado para el demo:
- https://www.youtube.com/watch?v=IehtMYlOuIk

