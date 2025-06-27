# Claude Development Rules for Fire Risk Calculator

## Project Context
This is a fire risk mapping application with:
- Flask backend (app.py) 
- Client-side scoring calculations (JavaScript)
- Mapbox GL JS for visualization
- PostgreSQL database with parcel data
- Redis caching for performance

## Architecture Principles
- **Client-side scoring**: All normalization and score calculations happen in JavaScript
- **Server-side data**: Only data retrieval, optimization, and caching
- **Vector tiles**: Use AttributeCollection format, not full GeoJSON with geometry

## Development Rules

### Code Changes
1. **Always read files before editing** - Use Read tool before any Edit/Write operations
2. **Preserve existing patterns** - Follow established code style and conventions
3. **No server-side scoring** - Keep all score calculations on client side
4. **Test commands**: Always run linting/typechecking after changes if available

### Scoring System
- **Raw Min-Max**: Uses true min/max values, no log transforms
- **Robust Min-Max**: Uses log transforms and 97th percentile caps for structures
- **Quantile**: Uses log transforms with percentile ranking
- **Client handles everything**: Server only provides raw data

### File Management
- **Archive important changes** - Create backups before major deletions
- **Use descriptive commit messages** - Follow existing git message style
- **Don't create unnecessary docs** - Only create .md files when explicitly requested

### Optimization System
- **Absolute optimization only** - Relative optimization was removed
- **File-based storage** - Use temp directories, not in-memory storage
- **Memory efficiency** - Always consider memory usage for large datasets

### Variable Naming
- **Base variables**: qtrmi, hwui, hagri, hvhsz, hfb, slope, neigh1d, hbrn, par_buf_sl, hlfmi_agfb
- **Score suffixes**: _s (always use), _q (quantile), _z (deprecated)
- **Raw variables**: Use RAW_VAR_MAP for mapping base to raw column names

### Common Tasks
- **Plot updates**: Labels go top-right, distinguish Raw vs Robust vs Quantile
- **Distribution endpoint**: Return raw database values, client handles normalization
- **Weight optimization**: Use LP solver, store files on disk, return minimal response

### Error Handling
- **Graceful degradation** - Always provide fallbacks for missing data
- **Detailed logging** - Log important operations and errors
- **User-friendly messages** - Return helpful error messages to client

## Current Status
- Server-side scoring functions removed (archived in /archive/app_withscorecalcs.py)
- All relative optimization functionality deleted
- Client handles all score calculations
- Distribution plots show Raw vs Robust Min-Max vs Quantile clearly

##General Rules
- Dont use emojis
- Dont include 'authored by Claude' in git commits
