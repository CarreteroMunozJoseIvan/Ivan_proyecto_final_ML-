# Predicción del Precio del Coche

Este repositorio contiene el código y los recursos para el proyecto de predicción del precio del coche.

|-- nombre_proyecto_final_ML
    |-- data
    |   |-- raw
    |        |-- dataset.csv
    |        |-- ...
    |   |-- processed
    |   |-- train
    |   |-- test
    |
    |-- notebooks
    |   |-- 01_EDA.ipynb
    |   |-- 02_Preprocesamiento.ipynb
    |   |-- 03_Entrenamiento_Modelo.ipynb
    |   |-- 04_Evaluacion_Modelo.ipynb
    |   |-- ...
    |
    |-- src
    |   |-- data_processing.py
    |   |-- model.py
    |   |-- evaluation.py
    |   |-- ...
    |
    |-- models
    |   |-- trained_model.pkl
    |   |-- model_config.yaml
    |   |-- ...
    |
    |-- app
    |   |-- app.py
    |   |-- requirements.txt
    |   |-- ...
    |
    |-- docs
    |   |-- negocio.ppt
    |   |-- ds.ppt
    |   |-- documentation.pdf
    |   |-- ...

## Estructura de Directorios

1. **data**: Almacena los datos utilizados en el proyecto.

   - `raw`: Contiene los datos en su formato original, sin procesar.
   - `processed`: Almacena los datos procesados después de realizar las transformaciones necesarias.
   - `train`: Contiene los datos de entrenamiento utilizados para entrenar el modelo.
   - `test`: Almacena los datos de prueba utilizados para evaluar el modelo.

2. **notebooks**: Contiene los archivos Jupyter Notebook que contienen el desarrollo del proyecto.

   - `01_EDA.ipynb`: Análisis exploratorio de datos.
   - `02_Preprocesamiento.ipynb`: Transformaciones y limpiezas, incluyendo el feature engineering.
   - `03_Entrenamiento_Modelo.ipynb`: Entrenamiento de modelos (mínimo 5 modelos supervisados diferentes y al menos 1 no supervisado) junto con su hiperparametrización.
   - `04_Evaluacion_Modelo.ipynb`: Evaluación de los modelos (métricas de evaluación, interpretación de variables, etc.).

3. **src**: Contiene los archivos fuente de Python que implementan las funcionalidades clave del proyecto.

   - `data_processing.py`: Código para procesar los datos de la carpeta `data/raw` y guardar los datos procesados en la carpeta `data/processed`.
   - `model.py`: Código para entrenar y guardar el modelo entrenado con el input de los datos de la carpeta `data/processed`, así como los datasets de `data/train` y `data/test` utilizados en el entrenamiento.
   - `evaluation.py`: Código para evaluar el modelo utilizando los datos de prueba de la carpeta `data/test` y generar métricas de evaluación.

4. **models**: Almacena los archivos relacionados con el modelo entrenado.

   - `trained_model.pkl`: Modelo entrenado guardado en formato pickle.
   - `model_config.yaml`: Archivo con la configuración del modelo (parámetros).

5. **app**: Contiene los archivos necesarios para el despliegue del modelo en Streamlit u otra plataforma similar.

   - `app.py`: Código para la aplicación web que utiliza el modelo entrenado (Streamlit, etc.).
   - `requirements.txt`: Especifica las dependencias del proyecto para poder ejecutar la aplicación.

6. **docs**: Contiene la documentación adicional relacionada con el proyecto.

   - `negocio.ppt`: Presentación del proyecto desde la perspectiva del negocio.
   - `ds.ppt`: Presentación del proyecto desde la perspectiva de los datos y el modelado.
   - `documentation.pdf`: Documentación detallada del proyecto.

