#!/bin/bash
# Script to run the Flask app locally with proper settings

echo "Starting Fire Risk Calculator locally..."
echo "=================================="
echo "The app will be available at:"
echo "  http://localhost:5000"
echo "  http://127.0.0.1:5000"
echo "=================================="
echo ""

# Set environment to ensure we're not in deployment mode
unset RAILWAY_ENVIRONMENT
unset DEPLOYED
unset PRODUCTION

# Run the Flask app
python3 app.py