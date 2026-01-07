FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for matplotlib and other libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install shiny (not in requirements.txt)
RUN pip install --no-cache-dir shiny

# Copy application files
COPY src/ ./src/
COPY data/ ./data/
COPY results/ ./results/

# Expose port
EXPOSE 8000

# Run the application
CMD ["shiny", "run", "--host", "0.0.0.0", "--port", "8000", "src/app.py"]

