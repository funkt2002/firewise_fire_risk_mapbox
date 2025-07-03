"""
Test Phase 1 refactoring: config, exceptions, and utils
"""
import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch, mock_open

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, DevelopmentConfig, ProductionConfig, get_config
from exceptions import (
    FireRiskError, DatabaseError, ValidationError, SessionError,
    validate_required_fields, validate_field_type, validate_numeric_range
)
from utils import (
    normalize_variable_name, correct_variable_name, get_base_variable_names,
    validate_variable_names, get_session_directory, validate_session_id,
    format_number, safe_float, validate_parcel_ids, validate_weights
)


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def test_config_basic_functionality(self):
        """Test basic config functionality."""
        # Test variable validation
        self.assertTrue(Config.is_valid_variable('hbrn'))
        self.assertFalse(Config.is_valid_variable('invalid_var'))
        
        # Test display names
        self.assertEqual(Config.get_display_name('hbrn'), 'Burn Score')
        self.assertEqual(Config.get_display_name('unknown'), 'unknown')
        
        # Test invert check
        self.assertTrue(Config.should_invert_variable('hagri'))
        self.assertFalse(Config.should_invert_variable('hbrn'))
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        key = Config.get_cache_key('parcels', 'test123')
        self.assertIn('fire_risk:', key)
        self.assertIn('v1:', key)
        self.assertIn('parcels', key)
        self.assertIn('test123', key)
    
    def test_variable_name_correction(self):
        """Test variable name correction."""
        self.assertEqual(Config.correct_variable_name('par_bufl'), 'par_buf_sl')
        self.assertEqual(Config.correct_variable_name('normal_var'), 'normal_var')
    
    def test_config_environments(self):
        """Test different config environments."""
        dev_config = get_config('development')
        self.assertTrue(dev_config.DEBUG)
        
        # Production config requires env vars, so we'll test the class directly
        self.assertFalse(ProductionConfig.DEBUG)


class TestExceptions(unittest.TestCase):
    """Test custom exceptions."""
    
    def test_fire_risk_error(self):
        """Test base FireRiskError."""
        error = FireRiskError("Test error", {"key": "value"}, 400)
        self.assertEqual(error.message, "Test error")
        self.assertEqual(error.details, {"key": "value"})
        self.assertEqual(error.status_code, 400)
    
    def test_database_error(self):
        """Test DatabaseError with original error."""
        original = Exception("Connection failed")
        error = DatabaseError("DB error", original_error=original)
        self.assertEqual(error.message, "DB error")
        self.assertIn("original_error", error.details)
    
    def test_validation_helpers(self):
        """Test validation helper functions."""
        # Test required fields
        with self.assertRaises(ValidationError):
            validate_required_fields({}, ['required_field'])
        
        # Should not raise
        validate_required_fields({'required_field': 'value'}, ['required_field'])
        
        # Test field type validation
        with self.assertRaises(ValidationError):
            validate_field_type({'field': 'string'}, 'field', int)
        
        # Should not raise
        validate_field_type({'field': 123}, 'field', int)
        
        # Test numeric range
        with self.assertRaises(ValidationError):
            validate_numeric_range(5, min_val=10)
        
        with self.assertRaises(ValidationError):
            validate_numeric_range(15, max_val=10)
        
        # Should not raise
        self.assertEqual(validate_numeric_range(7, min_val=5, max_val=10), 7)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_variable_name_utilities(self):
        """Test variable name utility functions."""
        # Test normalize
        self.assertEqual(normalize_variable_name('hbrn_s'), 'hbrn')
        self.assertEqual(normalize_variable_name('hbrn_q'), 'hbrn')
        self.assertEqual(normalize_variable_name('hbrn'), 'hbrn')
        
        # Test correct
        self.assertEqual(correct_variable_name('par_bufl'), 'par_buf_sl')
        
        # Test get base names
        vars_with_suffixes = ['hbrn_s', 'hwui_q', 'par_bufl_s']
        base_names = get_base_variable_names(vars_with_suffixes)
        expected = ['hbrn', 'hwui', 'par_buf_sl']
        self.assertEqual(base_names, expected)
        
        # Test validation
        valid, invalid = validate_variable_names(['hbrn', 'invalid_var', 'hwui_s'])
        self.assertIn('hbrn', valid)
        self.assertIn('hwui_s', valid)
        self.assertIn('invalid_var', invalid)
    
    def test_session_utilities(self):
        """Test session management utilities."""
        # Test session ID validation
        self.assertEqual(validate_session_id('test-123'), 'test-123')
        
        with self.assertRaises(ValidationError):
            validate_session_id('')
        
        with self.assertRaises(ValidationError):
            validate_session_id('invalid@session')
        
        # Test session directory
        session_dir = get_session_directory('test123')
        self.assertIn('fire_risk_sessions', session_dir)
        self.assertIn('test123', session_dir)
    
    def test_formatting_utilities(self):
        """Test data formatting utilities."""
        # Test number formatting
        self.assertEqual(format_number(1234.567, decimals=2), '1,234.57')
        self.assertEqual(format_number(1234, decimals=0), '1,234')
        self.assertEqual(format_number(None), 'N/A')
        
        # Test safe conversions
        self.assertEqual(safe_float('123.45'), 123.45)
        self.assertEqual(safe_float('invalid', 0.0), 0.0)
        self.assertEqual(safe_float(None, 5.0), 5.0)
    
    def test_validation_utilities(self):
        """Test validation utilities."""
        # Test parcel ID validation
        valid_ids = validate_parcel_ids(['parcel-1', 'parcel_2', 'PARCEL123'])
        self.assertEqual(len(valid_ids), 3)
        
        with self.assertRaises(ValidationError):
            validate_parcel_ids([])  # Empty list
        
        # Test weight validation
        weights = {'hbrn': 0.5, 'hwui': 0.3}
        validated = validate_weights(weights)
        self.assertEqual(validated['hbrn'], 0.5)
        self.assertEqual(validated['hwui'], 0.3)
        
        with self.assertRaises(ValidationError):
            validate_weights({'hbrn': 1.5})  # Out of range
        
        with self.assertRaises(ValidationError):
            validate_weights({'invalid_var': 0.5})  # Invalid variable


class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def test_config_utils_integration(self):
        """Test config and utils working together."""
        # Test that utils can use config correctly
        from utils import get_variable_display_info
        display_info = get_variable_display_info('hbrn_s')
        
        self.assertEqual(display_info['base'], 'hbrn')
        self.assertEqual(display_info['display_name'], 'Burn Score')
        self.assertFalse(display_info['should_invert'])
    
    def test_exception_validation_integration(self):
        """Test exceptions and validation working together."""
        # Test that validation functions raise proper exceptions
        try:
            validate_parcel_ids([])
        except ValidationError as e:
            self.assertIsInstance(e, FireRiskError)
            self.assertEqual(e.status_code, 400)


if __name__ == '__main__':
    # Run specific test groups
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='store_true', help='Test config only')
    parser.add_argument('--exceptions', action='store_true', help='Test exceptions only')
    parser.add_argument('--utils', action='store_true', help='Test utils only')
    args = parser.parse_args()
    
    if args.config:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestConfig)
    elif args.exceptions:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestExceptions)
    elif args.utils:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestUtils)
    else:
        # Run all tests
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)