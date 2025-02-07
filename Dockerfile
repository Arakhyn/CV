# Imagen base con dependencias comunes preinstaladas
FROM python:3.10

# Evita que Python cree archivos .pyc y usa un buffer de salida estándar
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar solo requirements.txt para instalar dependencias antes
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Ejecutar el script
CMD ["python", "Blackjack.py"]
