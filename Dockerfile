FROM python:3.10-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN pip install --no-cache-dir shiny


COPY src/ ./src/
COPY data/ ./data/
COPY results/ ./results/


EXPOSE 8000


CMD ["shiny", "run", "--host", "0.0.0.0", "--port", "8000", "src/app.py"]

