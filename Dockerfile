# Dockerfile: crea una "caja" con la API lista para usar
# Cómo construirla:   docker build -t api-credito .
# Cómo encenderla:    docker run -p 8000:8000 api-credito
FROM python:3.12-slim

WORKDIR /app

# 1. Instalar las librerías (primero, para que Docker las guarde en memoria)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copiar el código, el modelo y los datos
COPY mlops_pipeline ./mlops_pipeline
COPY Base_de_datos.csv .

WORKDIR /app/mlops_pipeline
EXPOSE 8000

# 3. Encender la API
CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
