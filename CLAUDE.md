# Claude Development Rules for Fire Risk Calculator

## Critical Rule
NEVER LIE TO APPEASE ME OR MAKE MY IDEAS/METHODS work. if something i propose is not a good idea, we should not do it

## Development Rules
1. First think through the problem, read the codebase for relevant files, and write a plan to extras/documentation/todo.md
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan
4. Then, begin working on the todo items, marking them as complete as you go
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity
7. Finally, add a review section to the todo.md file with a summary of the changes you made and any other relevant information

## General Rules
- Dont use emojis
- Dont include 'authored by Claude' in git commits
- DO git commits and pushes in one command, and use short descriptions
- Do not include any reference to claude in git commits 
- If I ask you to make a new script, just make one, not several. Always put outputs or new files in subdirectories unless I say otherwise
- Always organize outputs in subfolders and do not spit out files in root
- ALWAYS use real data from parcels.shp when available, NEVER use synthetic data unless explicitly requested
- When creating test scripts, always load real data first and only fall back to synthetic if real data is unavailable or if I specifically ask for synthetic data

## Project Directory Structure

### Core Web Application Files (Root Directory)
```
/
├── app.py                    # Main Flask application (2,212 lines)
├── config.py                 # Configuration management
├── exceptions.py             # Custom exception classes
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker compose setup
├── railway.json             # Railway deployment config
├── .gitignore              # Git ignore rules
├── README.md               # Main project documentation
└── CLAUDE.md               # This file - development guide
```

### Frontend Assets
```
/static/
├── js/                     # JavaScript modules
│   ├── client-filtering.js  # Client-side filtering logic
│   ├── client-scoring.js    # Client-side scoring calculations
│   ├── float32-converter.js # Float32 array conversions
│   ├── plotting.js         # Plotly chart functions
│   └── shared-data-store.js # Shared data management
└── welcome-popup.js        # Welcome tutorial popup
```

### Templates
```
/templates/
├── index.html              # Main application page (3,245 lines)
└── components/             # Reusable Jinja2 components
    ├── control_panel_header.html
    ├── data_filters.html
    ├── modal.html
    └── weight_slider.html
```

### Data Files
```
/data/                      # Geospatial data files
├── *.shp, *.dbf, *.prj    # Shapefiles for parcels, layers
├── *.geojson              # GeoJSON versions
└── *.mbtiles              # Mapbox tiles
```

### Tests
```
/tests/
├── test_app.py            # Application tests
└── test_phase1.py         # Phase 1 refactoring tests
```

### Extras Directory (Non-Essential Files)
```
/extras/                   # All non-production files (in .gitignore)
├── scripts/               # All utility and test scripts
│   ├── add_precomputed_geometry.py
│   ├── baselp.py
│   ├── baselp_with_imports.py
│   ├── check_columns.py
│   ├── deploy.sh
│   ├── inferweights_lp.py
│   ├── load_data.py
│   ├── print_parcels.py
│   ├── railway_debug.py
│   ├── setup_database.py
│   ├── test_api_key.py
│   ├── test_baselp.py
│   ├── test_separation_contiguous.py
│   └── travel_time_arc.py
├── archive/               # Old code versions
│   └── app_not_refactored.py
├── backups/               # Backup files
│   ├── app_backup.py
│   └── index.html.backup
├── documentation/         # Additional documentation
│   ├── PHASE1_COMPLETE.md
│   ├── refactor-plan.md
│   ├── refactoring_plan.md
│   └── todo.md
├── analysis/              # All analysis and correlation outputs
│   ├── correlation_matrix_*.csv
│   ├── correlation_vs_spatial_analysis.png
│   ├── fire_risk_distributions.png
│   ├── morans_i_matrix.csv
│   ├── plot_distributions.py
│   ├── quantile_correlation_matrix.*
│   └── [various correlation matrices]
├── outputs/               # All generated outputs
│   ├── fire_risk_scores_output.*
│   ├── fire_risk_distributions.png
│   ├── ranking_analysis_*.png
│   └── separation_lp_percentile_results.*
├── travel_times_output/   # Travel time calculations and data
│   ├── calculate_travel_times*.py
│   ├── join_travel_times.py
│   ├── merge_chunks.py
│   ├── parcel_travel_times*.*
│   ├── travel_time_map.py
│   └── travel_times_checkpoint*.pkl
└── venv_backups/         # Virtual environments
    ├── venv311/
    └── venv_clean/
```

## Key File Locations

### When working on backend logic:
- **Main app**: `app.py`
- **Configuration**: `config.py`
- **Utilities**: `utils.py`
- **Exceptions**: `exceptions.py`

### When working on frontend:
- **Main HTML**: `templates/index.html`
- **Components**: `templates/components/`
- **JavaScript**: `static/js/`

### When looking for scripts or tools:
- **All scripts**: `extras/scripts/`
- **Database setup**: `extras/scripts/setup_database.py`
- **Data loading**: `extras/scripts/load_data.py`

### When checking documentation:
- **Main README**: `README.md`
- **Refactoring plans**: `extras/documentation/refactor-plan.md`
- **Todo list**: `extras/documentation/todo.md` 
