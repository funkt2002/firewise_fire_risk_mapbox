import { ScoringService } from '../services/ScoringService';

describe('ScoringService - Memory Efficiency', () => {
  let service;
  
  beforeEach(() => {
    service = new ScoringService();
  });
  
  describe('Single data storage', () => {
    it('should store data only once in attributeMap', () => {
      const mockData = {
        type: 'AttributeCollection',
        attributes: [
          { parcel_id: '001', qtrmi_cnt: 10, hwui_pct: 50 },
          { parcel_id: '002', qtrmi_cnt: 20, hwui_pct: 80 },
          { parcel_id: '003', qtrmi_cnt: 5, hwui_pct: 20 }
        ]
      };
      
      service.initialize(mockData);
      
      // Check that data is stored only in attributeMap
      expect(service.attributeMap.size).toBe(3);
      expect(service.factorScoresCache.size).toBe(0); // Not calculated yet
      
      // Verify no duplicate storage
      const parcel001 = service.getParcelAttributes('001');
      const parcel001Again = service.getParcelAttributes('001');
      expect(parcel001).toBe(parcel001Again); // Same reference, not duplicated
    });
    
    it('should efficiently retrieve parcels without duplication', () => {
      const mockData = {
        type: 'AttributeCollection',
        attributes: Array(1000).fill(null).map((_, i) => ({
          parcel_id: `P${i.toString().padStart(4, '0')}`,
          qtrmi_cnt: Math.random() * 100,
          hwui_pct: Math.random() * 100
        }))
      };
      
      service.initialize(mockData);
      
      // Get all parcels multiple times
      const parcels1 = service.getAllParcels();
      const parcels2 = service.getAllParcels();
      
      // Should create new array but reference same attribute objects
      expect(parcels1).not.toBe(parcels2); // Different arrays
      expect(parcels1.length).toBe(1000);
      expect(parcels2.length).toBe(1000);
      
      // But attributes should be the same reference
      expect(parcels1[0].attributes).toBe(
        service.getParcelAttributes(parcels1[0].attributes.parcel_id)
      );
    });
  });
  
  describe('Memory-efficient filtering', () => {
    it('should filter without creating duplicate data', () => {
      const mockData = {
        type: 'AttributeCollection',
        attributes: [
          { parcel_id: '001', yearbuilt: 1990 },
          { parcel_id: '002', yearbuilt: 2000 },
          { parcel_id: '003', yearbuilt: 2010 },
          { parcel_id: '004', yearbuilt: 2020 },
          { parcel_id: '005', yearbuilt: null }
        ]
      };
      
      service.initialize(mockData);
      
      const filters = { yearbuilt_max: 2005 };
      const filtered = service.filterParcels(filters);
      
      expect(filtered).toHaveLength(3); // 1990, 2000, and null
      
      // Verify filtered parcels reference same attributes
      filtered.forEach(parcel => {
        const originalAttrs = service.getParcelAttributes(parcel.attributes.parcel_id);
        expect(parcel.attributes).toBe(originalAttrs);
      });
    });
  });
  
  describe('Cache management', () => {
    it('should only cache calculated scores, not raw data', () => {
      const mockData = {
        type: 'AttributeCollection',
        attributes: [
          { parcel_id: '001', qtrmi_cnt: 10 },
          { parcel_id: '002', qtrmi_cnt: 20 },
          { parcel_id: '003', qtrmi_cnt: 5 }
        ]
      };
      
      service.initialize(mockData);
      
      // Before calculation
      expect(service.factorScoresCache.size).toBe(0);
      
      // Simulate calculation results
      const scoredParcels = [
        {
          attributes: {
            parcel_id: '001',
            qtrmi_normalized: 0.5
          },
          compositeScore: 50
        },
        {
          attributes: {
            parcel_id: '002',
            qtrmi_normalized: 1.0
          },
          compositeScore: 100
        }
      ];
      
      service.updateFactorScoresCache(scoredParcels);
      
      // After calculation - only scores cached
      expect(service.factorScoresCache.size).toBe(2);
      expect(service.factorScoresCache.get('001')).toHaveProperty('composite_score', 50);
      expect(service.factorScoresCache.get('001')).toHaveProperty('qtrmi_s', 50);
    });
  });
  
  describe('Legacy compatibility', () => {
    it('should update legacy globals without duplication', () => {
      // Mock window object
      global.window = {
        map: { getSource: () => null, getLayer: () => null },
        fireRiskScoring: {}
      };
      
      const results = {
        scores: new Map([['001', 75], ['002', 85], ['003', 65]]),
        topParcelIds: ['002', '001']
      };
      
      service.updateLegacyGlobals(results);
      
      // Check globals are set
      expect(window.parcelScores).toEqual({
        '001': 75,
        '002': 85,
        '003': 65
      });
      expect(window.top500ParcelIds).toEqual(['002', '001']);
      expect(window.filteredParcelIds).toEqual(['001', '002', '003']);
      
      // Verify no duplication - same data referenced
      expect(Object.keys(window.parcelScores).length).toBe(3);
    });
  });
  
  describe('Performance', () => {
    it('should handle large datasets efficiently', () => {
      const largeData = {
        type: 'AttributeCollection',
        attributes: Array(10000).fill(null).map((_, i) => ({
          parcel_id: `P${i.toString().padStart(5, '0')}`,
          qtrmi_cnt: Math.random() * 100,
          hwui_pct: Math.random() * 100,
          hvhsz_pct: Math.random() * 100,
          par_buf_sl: Math.random() * 50
        }))
      };
      
      const startTime = Date.now();
      service.initialize(largeData);
      const initTime = Date.now() - startTime;
      
      // Initialization should be fast
      expect(initTime).toBeLessThan(100); // Less than 100ms for 10k parcels
      
      // Retrieval should be instant
      const retrieveStart = Date.now();
      const parcel = service.getParcelAttributes('P05000');
      const retrieveTime = Date.now() - retrieveStart;
      
      expect(parcel).toBeDefined();
      expect(retrieveTime).toBeLessThan(1); // Sub-millisecond lookup
    });
  });
});