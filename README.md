# Multimodal sentiment analysis
El campo de aprendizaje automático se diversifica cada vez más, alcanzando a cubrir nuevas áreas como lo es el de la salud mental. Temática que ha cobrado relevancia en tiempos cercanos, sobre todo en temas de depresión, ansiedad o ayuda psicológica y psiquiátrica. El proyecto busca alcanzar el punto donde se encuentran ambos rubros, teniendo un modelo de inteligencia que pueda recibir videos como entrada y que pueda clasificar las emociones que se denotan en el mismo, con la idea de que en un futuro se pueda usar probablemente como una herramienta para los expertos en salud mental y que puedan hacer diagnósticos con más información. Este modelo lo que busca también es tener en consideración todos los aspectos del paciente o persona que aparezca en el video, analizando su voz, el contenido de su discurso y sus expresiones faciales; Esto con la idea de que las inferencias tengan la mayor cantidad de atributos disponibles al ejecutarse.

## Uso del modelo

### Demo
Para correr el demo se debe ingresar al notebook `ColabCentral.ipynb` y ejecutar todas las celdas.
[![Open In Colab](https://colab.research.google.com/drive/1ieGuDje5AgOQ1sgbqv9UvSmGTW_ZTEL0?usp=sharing)

### Contenido personalizado

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
    
5. Finalmente dentro de la sección de DETR cambiar el valor de la variable `INPUT_PATH` por:
   `<nombre de la variable que recibió la función transcribe>[0]['video_path']`


Video usado para el demo:
- https://www.youtube.com/watch?v=IehtMYlOuIk


