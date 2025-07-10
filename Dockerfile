# Use the official slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies, including the CBC solver
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    postgresql-client \
    gdal-bin \
    libgdal-dev \
    coinor-cbc \
 && rm -rf /var/lib/apt/lists/*

# Copy Python dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir tabulate gunicorn

# Remove PuLP's vendored CBC binary so that the system solver is used
RUN rm -rf /usr/local/lib/python3.11/site-packages/pulp/solverdir/cbc

# Sanity checks: ensure system CBC is available and PuLP can see it
RUN which cbc && cbc --version
RUN python3 - <<EOF
import pulp
print("Available solvers:", pulp.listSolvers(onlyAvailable=True))
EOF

# Copy application code
COPY app.py config.py utils.py ./
COPY templates/ templates/
COPY static/ static/

# Create a non-root user and set ownership
RUN useradd --create-home --shell /bin/bash app \
 && chown -R app:app /app

# Switch to non-root user
USER app

# Expose port used by Railway
EXPOSE 5000

# Start the application with Gunicorn, using the $PORT environment variable
CMD gunicorn --bind 0.0.0.0:$PORT --workers 4 --timeout 120 --max-requests 1000 --max-requests-jitter 100 app:app
