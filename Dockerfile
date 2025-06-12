FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    libpq5 \
    gdal-bin \
    libgdal-dev \

    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir tabulate gunicorn

# Remove conflicting CBC solver if it exists
RUN rm -f /usr/local/lib/python3.11/site-packages/pulp/solverdir/cbc/linux/64/cbc

# Copy application files
COPY app.py .
COPY railway_debug.py .
COPY templates/ templates/
COPY static/ static/
COPY scripts/ scripts/

# Make scripts executable
RUN chmod +x scripts/*.py scripts/*.sh railway_debug.py

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Railway uses PORT environment variable
EXPOSE 5000

# Use gunicorn for production and Railway PORT
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 4 --timeout 120 --max-requests 1000 --max-requests-jitter 100 app:app