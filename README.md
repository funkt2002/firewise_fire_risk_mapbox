Fire Risk Calculator - GIS Web Application
🔥 Project Overview
The Fire Risk Calculator is a sophisticated GIS web application designed for firefighter planners to visualize, analyze, and prioritize fire-risk mitigation efforts on parcel data. The application uses multiple weighted factors to calculate fire risk scores and helps planners allocate limited budgets effectively.
Key Goals

Risk Visualization: Display parcels as an unclassed choropleth (white→red gradient) based on calculated fire-risk scores
Budget Optimization: Select the top N parcels (based on available budget) for fire mitigation treatment
Interactive Analysis: Allow real-time adjustment of risk factor weights through sliders
Inverse Optimization: Enable planners to draw rectangles around high-risk areas and automatically infer what weight combination would prioritize those parcels
Performance: Maintain sub-2 second response times on ~70,000 parcels

🏗️ Technical Architecture
Backend Stack

Flask: Python web framework for API endpoints
PostgreSQL/PostGIS: Spatial database for storing parcel geometries and attributes
Redis: Caching layer for performance optimization
PuLP: Linear programming library for inverse optimization
Gunicorn: Production WSGI server

Frontend Stack

Mapbox GL JS: Interactive map rendering
Vanilla JavaScript: Dynamic UI interactions
HTML5/CSS3: Modern, responsive interface with dark theme

Data Model
Parcels table with the following risk factors:

qtrmi_s: Distance to nearest structure (quarter mile score)
hwui_s: Wildland-Urban Interface coverage percentage
hagri_s: Agricultural/firebreak coverage
hvhsz_s: Very High Fire Hazard Zone coverage
uphill_s: Mean parcel slope
neigh1d_s: Distance to nearest neighbor

🚀 Local Development Setup
Prerequisites

Docker Desktop installed
Git
Mapbox account with access token
~2GB free disk space

Step-by-Step Local Setup

Clone the repository

bashgit clone <repository-url>
cd fire-risk-calculator

Project structure

fire-risk-calculator/
├── app.py                    # Flask backend
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Multi-container setup
├── .env                     # Environment variables
├── templates/
│   └── index.html          # Frontend interface
├── scripts/
│   └── load_data.py        # Data loading utility
├── data/                    # Place your shapefiles here
│   ├── parcels.shp
│   ├── parcels.shx
│   ├── parcels.dbf
│   └── parcels.prj
└── tests/
    └── test_app.py         # Test suite

Configure environment variables

bash# Create .env file
cat > .env << EOF
DATABASE_URL=postgresql://postgres:postgres@db:5432/firedb
REDIS_URL=redis://redis:6379
MAPBOX_TOKEN=pk.your_mapbox_token_here
EOF

Update docker-compose.yml ports (ensure correct mapping)

yamlservices:
  web:
    ports:
      - "5000:5000"  # Not 5000:8080

Start the application

bash# Build and start all services
docker-compose up --build -d

# Check services are running
docker-compose ps

Initialize database

bash# Create tables
docker-compose exec web python scripts/load_data.py --create-tables

# Load parcel data
docker-compose exec web python scripts/load_data.py \
  --parcels data/parcels.shp \
  --verify

# Verify data loaded
docker-compose exec db psql -U postgres firedb -c "SELECT COUNT(*) FROM parcels;"

Access the application


Open browser to: http://localhost:5000
You should see the map interface with your parcels

Testing the Application

Run unit tests

bashdocker-compose exec web pytest tests/

Test API endpoints

bash# Health check
curl http://localhost:5000/health

# Test score calculation
curl -X POST http://localhost:5000/api/score \
  -H "Content-Type: application/json" \
  -d '{
    "weights": {
      "qtrmi_s": 0.3,
      "hwui_s": 0.1,
      "hagri_s": 0.1,
      "hvhsz_s": 0.3,
      "uphill_s": 0.1,
      "neigh1d_s": 0.1
    },
    "top_n": 500
  }'

Test UI features


Adjust weight sliders and click "Calculate"
Change budget/cost parameters
Toggle filters (exclude parcels by year built)
Draw rectangle and click "Infer Weights"
Export selected parcels as GeoJSON

Debugging Common Issues

Empty map / No parcels showing

bash# Check Flask logs
docker-compose logs -f web

# Check database connection
docker-compose exec web python -c "from app import get_db; print(get_db())"

# Verify geometry column exists
docker-compose exec db psql -U postgres firedb -c "\d parcels"

Port errors

bash# Ensure nothing else is using port 5000
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows

Mapbox token issues

bash# Verify token is set
docker-compose exec web bash -c 'echo $MAPBOX_TOKEN'
🌍 Production Deployment
Option 1: Deploy to Fly.io (Recommended)

Install Fly CLI

bash# macOS
brew install flyctl

# Windows
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"

Initialize Fly app

bashfly launch --generate-name
# Follow prompts, choose region close to users

Set secrets

bashfly secrets set MAPBOX_TOKEN=pk.your_token_here
fly secrets set DATABASE_URL=your_production_db_url

Deploy

bashfly deploy

Scale for production

bashfly scale vm shared-cpu-2x --memory 1024
fly autoscale standard min=2 max=10
Option 2: Deploy to AWS

Using Elastic Beanstalk

bash# Install EB CLI
pip install awsebcli

# Initialize
eb init -p docker fire-risk-calculator

# Create environment
eb create fire-risk-prod

# Set environment variables
eb setenv MAPBOX_TOKEN=pk.your_token DATABASE_URL=your_rds_url

# Deploy
eb deploy

Using ECS/Fargate


Build and push image to ECR
Create ECS task definition
Configure ALB and target groups
Set up RDS PostgreSQL with PostGIS
Configure ElastiCache for Redis

Option 3: Deploy to Heroku

Create app and addons

bashheroku create fire-risk-calculator
heroku addons:create heroku-postgresql:hobby-dev
heroku addons:create heroku-redis:hobby-dev

Enable PostGIS

bashheroku pg:psql
> CREATE EXTENSION postgis;

Deploy

bashgit push heroku main
Production Considerations

Database


Use managed PostgreSQL with PostGIS (AWS RDS, Google Cloud SQL, etc.)
Enable connection pooling
Regular backups
Read replicas for scaling


Caching


Use managed Redis (AWS ElastiCache, Redis Cloud)
Set appropriate TTLs
Monitor memory usage


Performance


Use CDN for static assets
Enable gzip compression
Consider vector tiles for large datasets
Implement pagination for API responses


Security


Use HTTPS everywhere
Implement rate limiting
Validate all inputs
Use environment variables for secrets
Regular security updates


Monitoring


Application monitoring (New Relic, DataDog)
Error tracking (Sentry)
Uptime monitoring
Database performance monitoring

📊 Data Requirements
Your parcel shapefile/GeoJSON must include these fields:

Geometry (Polygon or MultiPolygon)
qtrmi_s, hwui_s, hagri_s, hvhsz_s, uphill_s, neigh1d_s (Float)
yearbuilt (Integer, optional)
Additional fields for popups: qtrmi_cnt, hlfmi_agri, hlfmi_wui, hlfmi_vhsz, num_neighb

🤝 For AI Assistants
When helping with this codebase:

The app uses PostGIS spatial functions extensively
Performance is critical - maintain sub-2s response times
The inverse optimization uses linear programming (PuLP)
Frontend uses vanilla JS, no frameworks
Docker Compose orchestrates PostgreSQL, Redis, and Flask
Geometry data is stored in EPSG:3857 (Web Mercator)
All API responses are GeoJSON format
The UI follows a dark theme design pattern

Key Files to Understand

app.py: Core API logic and database queries
templates/index.html: All frontend code
scripts/load_data.py: Data import logic
docker-compose.yml: Service configuration

Common Modification Requests

Adding new risk factors: Update WEIGHT_VARS in app.py
Changing scoring algorithm: Modify calculate_scores() function
UI theme changes: Update CSS in index.html
Adding new filters: Update both backend query and frontend UI

📝 License
[Your License Here]
🙏 Acknowledgments