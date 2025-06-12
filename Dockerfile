# Use the official slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies, including the coinor-cbc solver
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    postgresql-client \
    gdal-bin \
    libgdal-dev \
    coinor-cbc \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements for Python deps
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir tabulate gunicorn

# Remove PuLP's vendored CBC binary so that it will use /usr/bin/cbc instead
RUN rm -f /usr/local/lib/python3.11/site-packages/pulp/solverdir/cbc/linux/64/cbc || true

# Copy your application code
COPY app.py .
COPY railway_debug.py .
COPY templates/ templates/
COPY static/ static/
COPY scripts/ scripts/

# Make your scripts executable
RUN chmod +x railway_debug.py && \
    if [ -d "scripts" ]; then chmod +x scripts/*.py; fi

# Create a non-root user and give them ownership of /app
RUN useradd --create-home --shell /bin/bash app \
 && chown -R app:app /app

# Switch to non-root user
USER app

# Expose the port Railway will route to
EXPOSE 5000

# Launch via Gunicorn (Railway sets $PORT)
CMD gunicorn --bind 0.0.0.0:$PORT --workers 4 --timeout 120 --max-requests 1000 --max-requests-jitter 100 app:app
