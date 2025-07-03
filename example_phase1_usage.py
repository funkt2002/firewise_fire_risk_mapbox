"""
Example usage of Phase 1 refactoring components
"""
from config import Config, get_config
from exceptions import ValidationError, handle_api_errors
from utils import (
    correct_variable_names, validate_weights, validate_parcel_ids,
    format_number, get_session_directory, safe_float
)


def example_config_usage():
    """Example of using the config system."""
    print("=== Config Usage Example ===")
    
    # Get configuration
    config = get_config('development')
    print(f"Debug mode: {config.DEBUG}")
    print(f"Cache TTL: {config.REDIS['cache_ttl']}")
    
    # Use config methods
    print(f"Display name for 'hbrn': {Config.get_display_name('hbrn')}")
    print(f"Should invert 'hagri': {Config.should_invert_variable('hagri')}")
    
    # Generate cache key
    cache_key = Config.get_cache_key('parcels', 'user123')
    print(f"Cache key: {cache_key}")


def example_validation():
    """Example of validation with proper error handling."""
    print("\n=== Validation Example ===")
    
    try:
        # Variable name correction
        variables = ['hbrn_s', 'par_bufl', 'hwui_q']
        corrected = correct_variable_names(variables)
        print(f"Corrected variables: {variables} -> {corrected}")
        
        # Weight validation
        weights = {'hbrn': 0.5, 'hwui': 0.3, 'hagri': 0.2}
        validated_weights = validate_weights(weights)
        print(f"Validated weights: {validated_weights}")
        
        # Parcel ID validation
        parcel_ids = ['parcel-1', 'parcel_2', 'INVALID@ID', 'valid_123']
        valid_ids = validate_parcel_ids(parcel_ids)
        print(f"Valid parcel IDs: {valid_ids}")
        
    except ValidationError as e:
        print(f"Validation error: {e.message}")
        print(f"Details: {e.details}")


def example_utilities():
    """Example of utility functions."""
    print("\n=== Utilities Example ===")
    
    # Number formatting
    numbers = [1234567.89, 42, None, 'invalid']
    for num in numbers:
        formatted = format_number(num, decimals=2)
        print(f"{num} -> {formatted}")
    
    # Safe conversion
    values = ['123.45', 'invalid', None, 67.89]
    for val in values:
        safe_val = safe_float(val, default=0.0)
        print(f"safe_float({val}) -> {safe_val}")
    
    # Session directory
    session_dir = get_session_directory('test-session-123')
    print(f"Session directory: {session_dir}")


@handle_api_errors
def example_api_function(data):
    """Example API function with error handling decorator."""
    print("\n=== API Error Handling Example ===")
    
    # This will raise ValidationError which will be caught by decorator
    if not data.get('required_field'):
        raise ValidationError("Required field missing", {'field': 'required_field'})
    
    return {'status': 'success', 'data': data}


def main():
    """Run all examples."""
    example_config_usage()
    example_validation()
    example_utilities()
    
    # Test API error handling
    try:
        result = example_api_function({})
        print(f"API result: {result}")
    except Exception as e:
        print(f"API error handled: {e}")
    
    # Test with valid data
    try:
        result = example_api_function({'required_field': 'test'})
        print(f"API result: {result}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == '__main__':
    main()