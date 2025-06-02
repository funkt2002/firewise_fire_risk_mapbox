# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including GDAL)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    libpq5 \
    gdal-bin \
    libgdal-dev \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for data loading
RUN pip install --no-cache-dir tabulate

# Copy application files
COPY app.py .
COPY templates/ templates/
COPY static/ static/
COPY scripts/ scripts/

# Make scripts executable
RUN chmod +x scripts/*.py

# Expose port
EXPOSE 5000

# Run the application in production mode with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
