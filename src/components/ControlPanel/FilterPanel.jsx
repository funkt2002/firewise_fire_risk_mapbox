import React, { useState } from 'react';
import { useAppState } from '../../hooks/useAppState';

/**
 * Filter panel component for data filtering options
 */
function FilterPanel() {
  const { state, actions } = useAppState();
  const { filters } = state.ui;
  
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Handle filter changes
  const handleFilterChange = (filterName, value) => {
    actions.updateFilters({ [filterName]: value });
  };
  
  // Handle year built filter
  const handleYearBuiltToggle = (enabled) => {
    if (enabled) {
      const yearValue = document.getElementById('filter-yearbuilt')?.value || 1996;
      actions.updateFilters({ yearbuilt_max: parseInt(yearValue) });
    } else {
      actions.updateFilters({ yearbuilt_max: null });
    }
  };
  
  return (
    <div className="control-section">
      <div 
        style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', marginBottom: '8px' }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h2 style={{ margin: 0 }}>Data Filters</h2>
        <span style={{ marginLeft: '8px', fontSize: '12px', color: '#888' }}>
          {isExpanded ? '▼' : '▶'}
        </span>
      </div>
      
      {isExpanded && (
        <div style={{ marginLeft: '12px', paddingLeft: '8px', borderLeft: '2px solid rgba(255,255,255,0.1)' }}>
          
          {/* Property Characteristics */}
          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ fontSize: '13px', marginBottom: '10px', fontWeight: 'bold' }}>
              Parcel Characteristics
            </h3>
            
            <div className="input-group" style={{ 
              display: 'flex', 
              alignItems: 'center', 
              marginBottom: '10px', 
              fontSize: '12px', 
              background: 'rgba(255,255,255,0.03)', 
              padding: '8px', 
              borderRadius: '4px' 
            }}>
              <input 
                type="checkbox" 
                id="filter-yearbuilt-enabled"
                checked={filters.yearbuilt_max !== null}
                onChange={(e) => handleYearBuiltToggle(e.target.checked)}
                style={{ marginRight: '8px', verticalAlign: 'middle' }}
              />
              <span style={{ whiteSpace: 'nowrap', verticalAlign: 'middle', fontWeight: 500 }}>
                Built after year:
              </span>
              <input 
                type="number" 
                className="number-input"
                id="filter-yearbuilt"
                min="1800" 
                max="2024" 
                value={filters.yearbuilt_max || 1996}
                onChange={(e) => handleFilterChange('yearbuilt_max', parseInt(e.target.value))}
                disabled={filters.yearbuilt_max === null}
                style={{ marginLeft: '8px', width: '80px', verticalAlign: 'middle' }}
              />
            </div>
            
            <div className="toggle-container" style={{ marginLeft: '24px', marginBottom: '12px' }}>
              <label className="toggle-label" style={{ fontSize: '11px', color: '#ccc' }}>
                <input 
                  type="checkbox"
                  checked={filters.exclude_yearbuilt_unknown}
                  onChange={(e) => handleFilterChange('exclude_yearbuilt_unknown', e.target.checked)}
                />
                Exclude parcels with unknown construction year
              </label>
            </div>
          </div>
          
          {/* Additional filters can be added here following the same pattern */}
          
        </div>
      )}
    </div>
  );
}

export default FilterPanel;