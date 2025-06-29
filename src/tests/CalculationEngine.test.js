import { CalculationEngine } from '../services/CalculationEngine';

describe('CalculationEngine', () => {
  const mockParcels = [
    {
      attributes: {
        parcel_id: '001',
        qtrmi_cnt: 10,
        hwui_pct: 50,
        hvhsz_pct: 30,
        par_buf_sl: 15,
        hlfmi_agfb: 20,
        hagri_pct: 0,
        hfb_pct: 0,
        hbrn_pct: 0,
        par_slp_pct: 5,
        neigh1d_cnt: 3
      }
    },
    {
      attributes: {
        parcel_id: '002',
        qtrmi_cnt: 20,
        hwui_pct: 80,
        hvhsz_pct: 60,
        par_buf_sl: 25,
        hlfmi_agfb: 10,
        hagri_pct: 0,
        hfb_pct: 0,
        hbrn_pct: 0,
        par_slp_pct: 10,
        neigh1d_cnt: 5
      }
    },
    {
      attributes: {
        parcel_id: '003',
        qtrmi_cnt: 5,
        hwui_pct: 20,
        hvhsz_pct: 10,
        par_buf_sl: 5,
        hlfmi_agfb: 40,
        hagri_pct: 0,
        hfb_pct: 0,
        hbrn_pct: 0,
        par_slp_pct: 2,
        neigh1d_cnt: 1
      }
    }
  ];
  
  const mockWeights = {
    qtrmi_s: 30,
    hwui_s: 34,
    hvhsz_s: 26,
    par_buf_sl_s: 11,
    hlfmi_agfb_s: 0,
    hagri_s: 0,
    hfb_s: 0,
    hbrn_s: 0,
    slope_s: 0,
    neigh1d_s: 0
  };
  
  describe('computeScores', () => {
    it('should return empty results for empty parcels', () => {
      const results = CalculationEngine.computeScores([], mockWeights);
      
      expect(results.scoredParcels).toHaveLength(0);
      expect(results.scores.size).toBe(0);
      expect(results.rankings.size).toBe(0);
      expect(results.topParcelIds).toHaveLength(0);
    });
    
    it('should calculate scores for all parcels', () => {
      const results = CalculationEngine.computeScores(mockParcels, mockWeights);
      
      expect(results.scoredParcels).toHaveLength(3);
      expect(results.scores.size).toBe(3);
      expect(results.rankings.size).toBe(3);
    });
    
    it('should rank parcels correctly', () => {
      const results = CalculationEngine.computeScores(mockParcels, mockWeights);
      
      // Parcel 002 should have highest score (more structures, higher WUI/hazard)
      expect(results.rankings.get('002')).toBe(1);
      expect(results.rankings.get('001')).toBe(2);
      expect(results.rankings.get('003')).toBe(3);
    });
    
    it('should select top N parcels', () => {
      const results = CalculationEngine.computeScores(mockParcels, mockWeights, {
        maxParcels: 2
      });
      
      expect(results.topParcelIds).toHaveLength(2);
      expect(results.topParcelIds).toContain('002');
      expect(results.topParcelIds).toContain('001');
      expect(results.topParcelIds).not.toContain('003');
    });
    
    it('should handle different scoring methods', () => {
      const methods = ['raw_minmax', 'robust_minmax', 'quantile'];
      
      methods.forEach(method => {
        const results = CalculationEngine.computeScores(mockParcels, mockWeights, {
          scoringMethod: method
        });
        
        expect(results.scoredParcels).toHaveLength(3);
        expect(results.scores.size).toBe(3);
        
        // All scores should be between 0 and 100
        results.scores.forEach(score => {
          expect(score).toBeGreaterThanOrEqual(0);
          expect(score).toBeLessThanOrEqual(100);
        });
      });
    });
    
    it('should handle null/undefined values', () => {
      const parcelsWithNulls = [
        {
          attributes: {
            parcel_id: '004',
            qtrmi_cnt: null,
            hwui_pct: undefined,
            hvhsz_pct: 30,
            par_buf_sl: 15,
            hlfmi_agfb: 20
          }
        }
      ];
      
      const results = CalculationEngine.computeScores(parcelsWithNulls, mockWeights);
      
      expect(results.scoredParcels).toHaveLength(1);
      expect(results.scores.get('004')).toBeDefined();
      expect(results.scores.get('004')).toBeGreaterThanOrEqual(0);
    });
  });
  
  describe('normalizeFactors', () => {
    it('should normalize values between 0 and 1', () => {
      const normalized = CalculationEngine.normalizeRawMinMax(mockParcels, ['qtrmi']);
      
      normalized.forEach(parcel => {
        const value = parcel.attributes.qtrmi_normalized;
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThanOrEqual(1);
      });
    });
    
    it('should invert specified factors', () => {
      const parcelsWithInvertible = [
        {
          attributes: {
            parcel_id: '001',
            hagri_pct: 20,  // Should be inverted (agricultural is protective)
            hlfmi_agfb: 30  // Should be inverted (low fuel is protective)
          }
        },
        {
          attributes: {
            parcel_id: '002',
            hagri_pct: 80,  // Higher agricultural should result in lower risk
            hlfmi_agfb: 10
          }
        }
      ];
      
      const normalized = CalculationEngine.normalizeRawMinMax(parcelsWithInvertible, ['hagri', 'hlfmi_agfb']);
      
      // Parcel with higher agricultural coverage should have lower normalized score
      expect(normalized[0].attributes.hagri_normalized).toBeGreaterThan(
        normalized[1].attributes.hagri_normalized
      );
    });
  });
  
  describe('calculateStatistics', () => {
    it('should calculate correct statistics', () => {
      const scores = new Map([
        ['001', 20],
        ['002', 40],
        ['003', 60],
        ['004', 80],
        ['005', 100]
      ]);
      
      const stats = CalculationEngine.calculateStatistics(scores);
      
      expect(stats.min).toBe(20);
      expect(stats.max).toBe(100);
      expect(stats.mean).toBe(60);
      expect(stats.median).toBe(60);
      expect(stats.std).toBeCloseTo(31.62, 1);
    });
    
    it('should handle empty scores', () => {
      const stats = CalculationEngine.calculateStatistics(new Map());
      
      expect(stats.min).toBe(0);
      expect(stats.max).toBe(0);
      expect(stats.mean).toBe(0);
      expect(stats.median).toBe(0);
      expect(stats.std).toBe(0);
    });
  });
});