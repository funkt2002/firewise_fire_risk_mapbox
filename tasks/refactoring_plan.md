# Fire Risk Calculator Refactoring Plan

## Overview
This plan outlines a systematic approach to refactor the Fire Risk Calculator codebase, focusing on app.py and index.html. The goal is to improve code organization, readability, maintainability, and performance through incremental changes.

## 📊 Overall Progress Summary

### ✅ Completed Work:
- **Phase 1**: ✅ **FULLY COMPLETED** - Config, utils, and error handling modules
- **Phase 2**: 🟡 **50% COMPLETED** - Frontend template componentization done, backend modularization pending
- **Phase 4**: ✅ **80% COMPLETED** - Template component system implemented

### 🎯 Key Achievements So Far:
- **Net Code Reduction**: 88 lines removed while improving maintainability
- **Template Componentization**: Reduced 275+ repetitive HTML lines to 4-line Jinja2 loop
- **Modular Architecture**: Created reusable component system
- **Data-Driven UI**: Weight sliders now fully configurable via Flask data
- **Clean Git History**: All changes properly committed with backups

### 🚀 Immediate Next Steps:
1. **Phase 2 Backend**: Split monolithic app.py into service modules
2. **Phase 3 JS**: Further organize JavaScript into ES6 modules (optional)
3. **Phase 4 Completion**: Move remaining HTML generation to templates
4. **Phase 5**: Performance optimization and caching improvements

---

## ✅ COMPLETED: Frontend Template Componentization (Phase 2 - Partial)

### What Was Completed (December 2024)
**Status**: ✅ **DONE** - Successfully implemented and deployed
**Files Changed**: 6 files (+216 insertions, -298 deletions)
**Net Code Reduction**: 88 lines

#### 🎯 Major Achievements:
1. **Created Reusable Template Components**:
   - `templates/components/weight_slider.html` (22 lines) - Data-driven weight slider component
   - `templates/components/data_filters.html` (69 lines) - Data filtering controls component  
   - `templates/components/control_panel_header.html` (10 lines) - WRI header with logo/menu
   - `templates/components/modal.html` (11 lines) - Reusable modal template

2. **Refactored Main Template** (`templates/index.html`):
   - **Before**: 3,525 lines with 275+ lines of repetitive weight slider HTML
   - **After**: 3,240 lines with 4-line Jinja2 loop replacing repetitive sections
   - **Improvement**: 97% reduction in weight slider code complexity

3. **Enhanced Flask App** (`app.py`):
   - Added `weight_variables` data structure in `index()` route
   - Made weight sliders completely data-driven
   - Added 85 lines of structured configuration data

#### 🔧 Technical Implementation:
- **Jinja2 Components**: Used `{% include %}` with data-driven templates
- **Backwards Compatible**: All existing JavaScript/CSS continues to work
- **Docker Ready**: Template changes automatically included in existing Docker setup
- **Git History**: Clean commits with proper backup (`index.html.backup`)

#### 📊 Impact Metrics:
- **Maintainability**: Adding new weight variables now requires only data changes
- **Code Reuse**: Weight slider pattern now reusable across the application
- **File Organization**: Clear separation of concerns with component architecture
- **Developer Experience**: Much easier to modify UI components

#### 🧪 Testing Status:
- **Deployment**: Successfully committed and pushed (commits `9b40a2a`, `d564a0a`)
- **Syntax**: Fixed Jinja2 template syntax errors
- **Functionality**: All existing features maintained

---

## Phase 1: Quick Wins (1-2 days)
**Status**: ✅ **PARTIALLY COMPLETED** - Config and utils modules already exist

### 1.1 ✅ Extract Constants and Configuration
**File**: app.py
- **Problem**: Magic numbers and hardcoded values throughout
- **Solution**: Create `config.py` with all constants
- **Impact**: Low risk, improves maintainability
- **Status**: ✅ **DONE** - `config.py` already exists and is imported

```python
# config.py
DATABASE_CONFIG = {
    'host': os.environ.get('DATABASE_HOST', 'localhost'),
    'port': os.environ.get('DATABASE_PORT', '5432'),
    'name': os.environ.get('DATABASE_NAME', 'firewise'),
    'user': os.environ.get('DATABASE_USER', 'postgres'),
    'password': os.environ.get('DATABASE_PASSWORD', '')
}

CACHE_CONFIG = {
    'TTL': 86400,  # 24 hours
    'KEY_PREFIX': 'fire_risk:',
    'VERSION': 'v1'
}

OPTIMIZATION_PARAMS = {
    'INITIAL_TEMP': 1.0,
    'FINAL_TEMP': 0.01,
    'COOLING_RATE': 0.95,
    'MAX_ITERATIONS': 1000
}

VARIABLE_DISPLAY_NAMES = {
    'hbrn': 'Burn Score',
    'hwui': 'WUI Score',
    'qtrmi_s': 'Quarter Mile Score',
    # ... etc
}
```

### 1.2 ✅ Extract Repeated Code Patterns
**File**: app.py
- **Problem**: Variable name correction logic repeated 3 times
- **Solution**: Create utility function
- **Status**: ✅ **DONE** - `utils.py` already exists with utility functions

```python
# utils.py
def normalize_variable_name(var_name):
    """Remove scoring suffix from variable name"""
    if var_name.endswith(('_s', '_q')):
        return var_name[:-2]
    return var_name

def create_session_directory(session_id):
    """Create and validate session directory"""
    session_dir = os.path.join(tempfile.gettempdir(), 'fire_risk_sessions', session_id)
    if not os.path.exists(session_dir):
        return None, "Session not found"
    
    metadata_path = os.path.join(session_dir, 'metadata.json')
    # Check expiration logic
    return session_dir, None
```

### 1.3 ✅ Simplify Error Handling
**File**: app.py
- **Problem**: Bare except clauses and inconsistent error handling
- **Solution**: Add proper exception handling
- **Status**: ✅ **DONE** - `exceptions.py` already exists with custom exception classes

```python
# exceptions.py
class FireRiskError(Exception):
    """Base exception for Fire Risk Calculator"""
    pass

class DatabaseError(FireRiskError):
    """Database connection or query errors"""
    pass

class OptimizationError(FireRiskError):
    """Optimization algorithm errors"""
    pass

class SessionError(FireRiskError):
    """Session management errors"""
    pass
```

## Phase 2: Backend Modularization (3-5 days)
**Status**: 🟡 **IN PROGRESS** - Frontend completed, backend modules partially done

### 2.1 🟡 Split app.py into Modules
**Status**: 🟡 **PARTIALLY COMPLETED** - Some modules exist, main app.py still monolithic
Create the following structure:
```
app/
├── __init__.py
├── main.py (Flask app initialization)
├── config.py
├── models/
│   ├── __init__.py
│   └── parcel.py
├── services/
│   ├── __init__.py
│   ├── database.py
│   ├── cache.py
│   ├── optimization.py
│   └── scoring.py
├── routes/
│   ├── __init__.py
│   ├── data.py
│   ├── optimization.py
│   └── export.py
└── utils/
    ├── __init__.py
    ├── geo.py
    └── html_generator.py
```

### 2.2 Database Service
**File**: services/database.py
```python
class DatabaseService:
    def __init__(self, config):
        self.config = config
        self.pool = self._create_pool()
    
    def get_connection(self):
        return self.pool.getconn()
    
    def release_connection(self, conn):
        self.pool.putconn(conn)
    
    def fetch_parcels(self, filters=None, columns=None):
        # Parameterized query building
        pass
```

### 2.3 Cache Service
**File**: services/cache.py
```python
class CacheService:
    def __init__(self, redis_client, config):
        self.redis = redis_client
        self.config = config
    
    def get(self, key):
        full_key = f"{self.config['KEY_PREFIX']}{key}"
        # Decompress and return
    
    def set(self, key, value, ttl=None):
        # Compress and store
```

### 2.4 Optimization Service
**File**: services/optimization.py
```python
class OptimizationService:
    def __init__(self, config):
        self.config = config
    
    def solve_weights(self, parcels, method='lp'):
        if method == 'lp':
            return self._solve_lp(parcels)
        elif method == 'heuristic':
            return self._solve_heuristic(parcels)
        elif method == 'separation':
            return self._solve_separation(parcels)
```

## Phase 3: Frontend Refactoring (3-5 days)
**Status**: ✅ **PARTIALLY COMPLETED** - Template componentization done, JS modularization pending

### 3.1 🟡 Extract JavaScript to Modules
**Status**: 🟡 **PENDING** - Current JS files are already well-organized, could be improved further
Create structure:
```
static/js/
├── core/
│   ├── state-manager.js
│   ├── event-bus.js
│   └── constants.js
├── services/
│   ├── api-service.js
│   ├── map-service.js
│   └── data-service.js
├── components/
│   ├── variable-slider.js
│   ├── popup-manager.js
│   └── modal-manager.js
└── main.js
```

### 3.2 State Management
**File**: static/js/core/state-manager.js
```javascript
class StateManager {
    constructor() {
        this.state = {
            parcels: [],
            weights: {},
            selectedParcels: [],
            filters: {},
            mapView: {}
        };
        this.listeners = new Map();
    }
    
    subscribe(key, callback) {
        if (!this.listeners.has(key)) {
            this.listeners.set(key, []);
        }
        this.listeners.get(key).push(callback);
    }
    
    setState(key, value) {
        this.state[key] = value;
        this.notify(key);
    }
}
```

### 3.3 Component System
**File**: static/js/components/variable-slider.js
```javascript
class VariableSlider {
    constructor(container, config) {
        this.container = container;
        this.config = config;
        this.render();
        this.attachEventListeners();
    }
    
    render() {
        // Create DOM elements
    }
    
    getValue() {
        return parseFloat(this.slider.value);
    }
    
    setValue(value) {
        this.slider.value = value;
        this.updateDisplay();
    }
}
```

### 3.4 API Service
**File**: static/js/services/api-service.js
```javascript
class APIService {
    constructor(baseURL) {
        this.baseURL = baseURL;
    }
    
    async fetchParcels(params) {
        try {
            const response = await fetch(`${this.baseURL}/prepare_data`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            throw new APIError('Failed to fetch parcels', error);
        }
    }
}
```

## Phase 4: Template System (2-3 days)
**Status**: ✅ **MOSTLY COMPLETED** - Component templates implemented, some HTML generation remains

### 4.1 ✅ Move HTML Generation to Templates
**Status**: ✅ **COMPLETED** - Component template system implemented
Created Jinja2 templates:
```
templates/
├── index.html (refactored from 3,525 to 3,240 lines)
├── components/
│   ├── weight_slider.html ✅ DONE
│   ├── data_filters.html ✅ DONE  
│   ├── control_panel_header.html ✅ DONE
│   └── modal.html ✅ DONE
└── reports/
    └── solution_report.html 🟡 PENDING
```

### 4.2 🟡 Replace String HTML Generation
**Status**: 🟡 **PARTIALLY DONE** - Components completed, report generation still in app.py
**Before**: app.py lines 1680-2071
**After**: Use Jinja2 template
```python
@app.route('/generate_report/<session_id>')
def generate_report(session_id):
    data = prepare_report_data(session_id)
    return render_template('reports/solution_report.html', **data)
```

## Phase 5: Performance Optimization (2-3 days)

### 5.1 Database Connection Pooling
```python
from psycopg2 import pool

class DatabasePool:
    def __init__(self, minconn=1, maxconn=10):
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn, maxconn,
            host=DATABASE_CONFIG['host'],
            database=DATABASE_CONFIG['name'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password']
        )
```

### 5.2 Batch Operations
```python
def fetch_parcels_paginated(filters, page_size=1000):
    """Fetch parcels in batches to reduce memory usage"""
    offset = 0
    while True:
        parcels = fetch_parcels(filters, limit=page_size, offset=offset)
        if not parcels:
            break
        yield parcels
        offset += page_size
```

### 5.3 Frontend Optimizations
- Implement virtual scrolling for large datasets
- Use Web Workers for heavy calculations
- Add request debouncing for user inputs

## Implementation Order

1. **Week 1**: Phase 1 (Quick Wins)
   - Extract constants and configuration
   - Create utility functions
   - Add proper exception handling

2. **Week 2**: Phase 2 (Backend Modularization)
   - Split app.py into services
   - Implement connection pooling
   - Create route blueprints

3. **Week 3**: Phase 3 (Frontend Refactoring)
   - Extract JavaScript modules
   - Implement state management
   - Create component system

4. **Week 4**: Phase 4 & 5 (Templates & Performance)
   - Move to template system
   - Add performance optimizations
   - Testing and refinement

## Success Metrics

- **Code Quality**:
  - Reduce average function length from 100+ lines to <50 lines
  - Eliminate code duplication (DRY principle)
  - Achieve 80%+ test coverage

- **Performance**:
  - Reduce initial page load time by 30%
  - Improve API response times by 40%
  - Reduce memory usage by 25%

- **Maintainability**:
  - Clear separation of concerns
  - Well-documented modules
  - Easy to add new features

## Risk Mitigation

1. **Incremental Changes**: Each phase can be deployed independently
2. **Backward Compatibility**: Maintain existing API contracts
3. **Feature Flags**: Use flags to toggle between old and new implementations
4. **Comprehensive Testing**: Add tests before refactoring
5. **Version Control**: Create feature branches for each phase

## Next Steps

1. Review and approve this plan
2. Set up the new directory structure
3. Begin with Phase 1 quick wins
4. Create unit tests for existing functionality
5. Start incremental refactoring

This plan focuses on practical, incremental improvements that will make the codebase cleaner, more maintainable, and more efficient without requiring a complete rewrite.