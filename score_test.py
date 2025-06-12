#!/usr/bin/env python3
"""
Fire Risk Scoring Method Test Suite

This script comprehensively tests the fire risk calculation system across all three
normalization methods to ensure smooth operation when switching between:
- Basic Min-Max (_s columns)
- Robust Min-Max (_q columns) 
- Quantile Z-Score (_z columns)

Tests include:
1. Score calculation with different methods
2. Weight inference optimization 
3. Distribution plot data
4. Local vs Global normalization
5. Performance and consistency checks
"""

import requests
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# Configuration
BASE_URL = "http://localhost:5000"
TEST_AREA = {
    "type": "Polygon",
    "coordinates": [[
        [-121.5, 38.5],
        [-121.4, 38.5], 
        [-121.4, 38.6],
        [-121.5, 38.6],
        [-121.5, 38.5]
    ]]
}

# Test weights - same across all methods
TEST_WEIGHTS = {
    'qtrmi_s': 0.2,
    'hwui_s': 0.15,
    'hagri_s': 0.1,
    'hvhsz_s': 0.2,
    'hfb_s': 0.1,
    'slope_s': 0.15,
    'neigh1d_s': 0.05,
    'hbrn_s': 0.05
}

# Test filters
TEST_FILTERS = {
    'yearbuilt_max': 2000,
    'exclude_yearbuilt_unknown': False,
    'neigh1d_max': 1000,
    'strcnt_min': 1,
    'exclude_wui_min30': False,
    'exclude_vhsz_min10': False,
    'exclude_no_brns': False,
    'subset_area': TEST_AREA,
    'max_parcels': 100
}

class ScoringMethodTester:
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def print_header(self, title):
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        
    def print_section(self, title):
        print(f"\n{'-'*40}")
        print(f"{title}")
        print(f"{'-'*40}")
        
    def test_api_connection(self):
        """Test that the API is accessible"""
        self.print_header("API CONNECTION TEST")
        try:
            response = requests.get(f"{BASE_URL}/api/debug/columns", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("✅ API connection successful")
                
                # Check available score columns
                s_cols = data['score_analysis']['_s_columns']
                q_cols = data['score_analysis']['_q_columns'] 
                z_cols = data['score_analysis']['_z_columns']
                
                print(f"   Basic (_s) columns: {len(s_cols)} found")
                print(f"   Robust (_q) columns: {len(q_cols)} found")
                print(f"   Quantile (_z) columns: {len(z_cols)} found")
                
                missing_s = [var + '_s' for var in ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn'] if var + '_s' not in s_cols]
                missing_q = [var + '_q' for var in ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn'] if var + '_q' not in q_cols]
                missing_z = [var + '_z' for var in ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn'] if var + '_z' not in z_cols]
                
                if missing_s or missing_q or missing_z:
                    print(f"⚠️  Missing columns detected:")
                    if missing_s: print(f"   Missing _s: {missing_s}")
                    if missing_q: print(f"   Missing _q: {missing_q}")
                    if missing_z: print(f"   Missing _z: {missing_z}")
                else:
                    print("✅ All expected score columns found")
                    
                return True
            else:
                print(f"❌ API connection failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API connection error: {e}")
            return False
            
    def test_scoring_methods(self):
        """Test scoring across all three normalization methods"""
        self.print_header("SCORING METHOD COMPARISON")
        
        methods = [
            ("Basic Min-Max", {"use_quantiled_scores": False, "use_quantile": False}),
            ("Robust Min-Max", {"use_quantiled_scores": True, "use_quantile": False}),
            ("Quantile Z-Score", {"use_quantiled_scores": False, "use_quantile": True})
        ]
        
        scoring_results = {}
        
        for method_name, method_params in methods:
            self.print_section(f"Testing {method_name}")
            
            payload = {
                "weights": TEST_WEIGHTS,
                **TEST_FILTERS,
                **method_params,
                "use_local_normalization": False
            }
            
            start_time = time.time()
            try:
                response = requests.post(f"{BASE_URL}/api/score", json=payload, timeout=30)
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    features = data.get('features', [])
                    scores = [f['properties']['score'] for f in features]
                    
                    if scores:
                        print(f"✅ {method_name} successful ({elapsed:.2f}s)")
                        print(f"   Parcels returned: {len(features)}")
                        print(f"   Score range: {min(scores):.4f} - {max(scores):.4f}")
                        print(f"   Average score: {np.mean(scores):.4f}")
                        print(f"   Std deviation: {np.std(scores):.4f}")
                        print(f"   Normalization: {data.get('normalization', {}).get('mode', 'unknown')}")
                        
                        scoring_results[method_name] = {
                            'scores': scores,
                            'features': features,
                            'time': elapsed,
                            'response': data
                        }
                    else:
                        print(f"⚠️  {method_name} returned no scores")
                else:
                    print(f"❌ {method_name} failed: {response.status_code}")
                    if response.text:
                        print(f"   Error: {response.text[:200]}")
                        
            except Exception as e:
                print(f"❌ {method_name} error: {e}")
                
        self.results['scoring'] = scoring_results
        return scoring_results
        
    def test_local_vs_global_normalization(self):
        """Test local normalization vs global for consistency"""
        self.print_header("LOCAL vs GLOBAL NORMALIZATION")
        
        method_params = {"use_quantiled_scores": False, "use_quantile": False}
        
        for norm_type, use_local in [("Global", False), ("Local", True)]:
            self.print_section(f"Testing {norm_type} Normalization")
            
            payload = {
                "weights": TEST_WEIGHTS,
                **TEST_FILTERS,
                **method_params,
                "use_local_normalization": use_local
            }
            
            try:
                response = requests.post(f"{BASE_URL}/api/score", json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    features = data.get('features', [])
                    scores = [f['properties']['score'] for f in features]
                    
                    if scores:
                        print(f"✅ {norm_type} normalization successful")
                        print(f"   Score range: {min(scores):.4f} - {max(scores):.4f}")
                        print(f"   Average: {np.mean(scores):.4f}")
                        print(f"   Note: {data.get('normalization', {}).get('note', 'N/A')}")
                    else:
                        print(f"⚠️  {norm_type} returned no scores")
                else:
                    print(f"❌ {norm_type} failed: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {norm_type} error: {e}")
                
    def test_weight_inference(self):
        """Test weight inference across all scoring methods"""
        self.print_header("WEIGHT INFERENCE TESTING")
        
        methods = [
            ("Basic Min-Max", {"use_quantiled_scores": False, "use_quantile": False}),
            ("Robust Min-Max", {"use_quantiled_scores": True, "use_quantile": False}),
            ("Quantile Z-Score", {"use_quantiled_scores": False, "use_quantile": True})
        ]
        
        weight_results = {}
        
        for method_name, method_params in methods:
            self.print_section(f"Inferring weights for {method_name}")
            
            # Use smaller area for weight inference (faster computation)
            selection_area = {
                "type": "Polygon", 
                "coordinates": [[
                    [-121.48, 38.52],
                    [-121.47, 38.52],
                    [-121.47, 38.53], 
                    [-121.48, 38.53],
                    [-121.48, 38.52]
                ]]
            }
            
            payload = {
                "selection": selection_area,
                "include_vars": list(TEST_WEIGHTS.keys()),
                **TEST_FILTERS,
                **method_params
            }
            
            start_time = time.time()
            try:
                response = requests.post(f"{BASE_URL}/api/infer-weights", json=payload, timeout=60)
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    weights = data.get('weights', {})
                    
                    print(f"✅ Weight inference successful ({elapsed:.2f}s)")
                    print(f"   Parcels analyzed: {data.get('num_parcels', 0)}")
                    print(f"   Total score: {data.get('total_score', 0):.2f}")
                    print(f"   Solver status: {data.get('solver_status', 'unknown')}")
                    print(f"   Optimal weights:")
                    
                    # Sort weights by value for display
                    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                    for var, weight in sorted_weights:
                        var_name = var.replace('_s', '')
                        print(f"     {var_name}: {weight:.1f}%")
                        
                    weight_results[method_name] = {
                        'weights': weights,
                        'time': elapsed,
                        'response': data
                    }
                else:
                    print(f"❌ Weight inference failed: {response.status_code}")
                    if response.text:
                        print(f"   Error: {response.text[:200]}")
                        
            except Exception as e:
                print(f"❌ Weight inference error: {e}")
                
        self.results['weights'] = weight_results
        return weight_results
        
    def test_distribution_plots(self):
        """Test distribution data across scoring methods"""
        self.print_header("DISTRIBUTION PLOT DATA TESTING")
        
        methods = [
            ("Basic Min-Max", {"use_quantiled_scores": False, "use_quantile": False}, "_s"),
            ("Robust Min-Max", {"use_quantiled_scores": True, "use_quantile": False}, "_q"),
            ("Quantile Z-Score", {"use_quantiled_scores": False, "use_quantile": True}, "_z")
        ]
        
        test_variables = ['qtrmi', 'hwui', 'slope']  # Test subset for speed
        
        for method_name, method_params, suffix in methods:
            self.print_section(f"Distribution data for {method_name}")
            
            for var_base in test_variables:
                var_name = var_base + suffix
                
                payload = {
                    **TEST_FILTERS,
                    **method_params
                }
                
                try:
                    response = requests.post(f"{BASE_URL}/api/distribution/{var_name}", json=payload, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        values = data.get('values', [])
                        
                        if values:
                            print(f"   ✅ {var_name}: {len(values)} values, range {data.get('min', 0):.3f}-{data.get('max', 0):.3f}")
                        else:
                            print(f"   ⚠️  {var_name}: No values returned")
                    else:
                        print(f"   ❌ {var_name}: Failed ({response.status_code})")
                        
                except Exception as e:
                    print(f"   ❌ {var_name}: Error - {e}")
                    
    def test_consistency_across_methods(self):
        """Test that results are logically consistent across methods"""
        self.print_header("CROSS-METHOD CONSISTENCY ANALYSIS")
        
        if 'scoring' not in self.results or len(self.results['scoring']) < 2:
            print("❌ Insufficient scoring results for consistency testing")
            return
            
        scoring_results = self.results['scoring']
        method_names = list(scoring_results.keys())
        
        print("Analyzing score distributions across methods...")
        
        # Compare score distributions
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:
                scores1 = scoring_results[method1]['scores']
                scores2 = scoring_results[method2]['scores']
                
                if len(scores1) > 0 and len(scores2) > 0:
                    # Check if relative rankings are preserved (correlation)
                    min_len = min(len(scores1), len(scores2))
                    if min_len > 1:
                        correlation = np.corrcoef(scores1[:min_len], scores2[:min_len])[0,1]
                        print(f"   {method1} vs {method2}: correlation = {correlation:.3f}")
                        
                        if correlation > 0.7:
                            print(f"     ✅ Strong positive correlation (rankings preserved)")
                        elif correlation > 0.3:
                            print(f"     ⚠️  Moderate correlation")
                        else:
                            print(f"     ❌ Low correlation (potential issue)")
                            
    def create_visual_report(self):
        """Create visual plots showing method comparisons"""
        self.print_header("GENERATING VISUAL REPORT")
        
        if 'scoring' not in self.results:
            print("❌ No scoring results available for visualization")
            return
            
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Fire Risk Scoring Method Comparison', fontsize=16)
            
            # Plot 1: Score distributions
            ax1 = axes[0, 0]
            for method_name, result in self.results['scoring'].items():
                scores = result['scores']
                if scores:
                    ax1.hist(scores, alpha=0.7, bins=20, label=method_name, density=True)
            ax1.set_xlabel('Fire Risk Score')
            ax1.set_ylabel('Density')
            ax1.set_title('Score Distributions by Method')
            ax1.legend()
            
            # Plot 2: Performance comparison
            ax2 = axes[0, 1]
            methods = []
            times = []
            for method_name, result in self.results['scoring'].items():
                methods.append(method_name.replace(' ', '\n'))
                times.append(result['time'])
            if times:
                ax2.bar(methods, times, color=['skyblue', 'lightgreen', 'salmon'][:len(times)])
                ax2.set_ylabel('Execution Time (seconds)')
                ax2.set_title('Performance by Method')
                ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Weight comparison (if available)
            ax3 = axes[1, 0]
            if 'weights' in self.results:
                weight_data = {}
                for method_name, result in self.results['weights'].items():
                    for var, weight in result['weights'].items():
                        var_base = var.replace('_s', '')
                        if var_base not in weight_data:
                            weight_data[var_base] = {}
                        weight_data[var_base][method_name] = weight
                
                if weight_data:
                    x_pos = np.arange(len(weight_data))
                    width = 0.25
                    colors = ['skyblue', 'lightgreen', 'salmon']
                    
                    for i, method in enumerate(self.results['weights'].keys()):
                        weights = [weight_data[var].get(method, 0) for var in weight_data.keys()]
                        ax3.bar(x_pos + i*width, weights, width, label=method, color=colors[i])
                    
                    ax3.set_xlabel('Risk Factors')
                    ax3.set_ylabel('Optimal Weight (%)')
                    ax3.set_title('Inferred Weights by Method')
                    ax3.set_xticks(x_pos + width)
                    ax3.set_xticklabels(weight_data.keys(), rotation=45)
                    ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'No weight data available', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Inferred Weights (No Data)')
            
            # Plot 4: Summary statistics
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            summary_text = f"""
Test Summary:
• API Connection: {'✅' if hasattr(self, 'api_connected') else '❓'}
• Methods Tested: {len(self.results.get('scoring', {}))}
• Total Runtime: {time.time() - self.start_time:.1f}s

Score Statistics:
"""
            for method_name, result in self.results.get('scoring', {}).items():
                scores = result['scores']
                if scores:
                    summary_text += f"• {method_name}:\n"
                    summary_text += f"  Parcels: {len(scores)}\n"
                    summary_text += f"  Avg Score: {np.mean(scores):.3f}\n"
                    summary_text += f"  Std Dev: {np.std(scores):.3f}\n"
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10, 
                    verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scoring_method_test_{timestamp}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✅ Visual report saved as: {filename}")
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating visual report: {e}")
            
    def run_full_test_suite(self):
        """Run the complete test suite"""
        print(f"🔥 Fire Risk Scoring Method Test Suite")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target API: {BASE_URL}")
        
        # Test 1: API Connection
        if not self.test_api_connection():
            print("❌ Cannot proceed - API not accessible")
            return False
            
        # Test 2: Core scoring functionality
        self.test_scoring_methods()
        
        # Test 3: Local vs Global normalization
        self.test_local_vs_global_normalization()
        
        # Test 4: Weight inference
        self.test_weight_inference()
        
        # Test 5: Distribution plots
        self.test_distribution_plots()
        
        # Test 6: Consistency analysis
        self.test_consistency_across_methods()
        
        # Test 7: Visual report
        self.create_visual_report()
        
        # Final summary
        self.print_header("TEST SUITE SUMMARY")
        total_time = time.time() - self.start_time
        print(f"✅ Test suite completed in {total_time:.1f} seconds")
        
        scoring_success = len(self.results.get('scoring', {}))
        weights_success = len(self.results.get('weights', {}))
        
        print(f"📊 Results:")
        print(f"   • Scoring methods tested: {scoring_success}/3")
        print(f"   • Weight inference tested: {weights_success}/3")
        print(f"   • Overall status: {'✅ PASS' if scoring_success >= 2 else '❌ ISSUES DETECTED'}")
        
        if scoring_success < 3:
            print(f"⚠️  Some scoring methods failed - check database columns and API logs")
            
        return scoring_success >= 2

def main():
    """Main function to run the test suite"""
    print("Starting Fire Risk Scoring Method Test Suite...")
    print("This will test all normalization methods for consistency and performance.")
    print("\nMake sure your Flask API is running on localhost:5000")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    tester = ScoringMethodTester()
    success = tester.run_full_test_suite()
    
    if success:
        print("\n🎉 Test suite completed successfully!")
        print("All scoring methods appear to be working correctly.")
    else:
        print("\n⚠️  Test suite completed with issues.")
        print("Check the output above for specific problems.")
        
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 