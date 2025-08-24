FROM python:3.9-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Rodar o aplicativo
COPY app.py .
CMD [ "python", "app.py" ]