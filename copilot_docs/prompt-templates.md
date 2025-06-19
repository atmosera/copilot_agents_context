# Prompt Templates Library

## Table of Contents
- [Code Generation Templates](#code-generation-templates)
- [Documentation Templates](#documentation-templates)
- [Testing Templates](#testing-templates)
- [Refactoring Templates](#refactoring-templates)
- [Architecture Templates](#architecture-templates)
- [Debugging Templates](#debugging-templates)
- [Security Templates](#security-templates)
- [Performance Templates](#performance-templates)

## Code Generation Templates

### Function Generation Template
```
# Template: Function with specific requirements
# Create a {function_type} function that {primary_purpose}
# Requirements:
# - {requirement_1}
# - {requirement_2}
# - {requirement_3}
# Parameters: {parameter_description}
# Returns: {return_description}
# Error handling: {error_handling_approach}

def {function_name}({parameters}) -> {return_type}:
    """
    {brief_description}
    
    Args:
        {parameter_documentation}
    
    Returns:
        {return_documentation}
    
    Raises:
        {exception_documentation}
    """
    # Implementation will be generated here
```

### Class Generation Template
```
# Template: Class with specific behavior
/**
 * Create a {class_name} class that implements {interface_or_pattern}
 * Requirements:
 * - {requirement_1}
 * - {requirement_2}
 * - {requirement_3}
 * Design patterns: {patterns_to_use}
 * Thread safety: {thread_safety_requirements}
 * Performance considerations: {performance_notes}
 */
public class {class_name} implements {interface} {
    // Class implementation will be generated here
}
```

### API Endpoint Template
```
# Template: REST API endpoint
# Create a {http_method} endpoint for {resource_name}
# Path: {endpoint_path}
# Purpose: {endpoint_purpose}
# Request body: {request_schema}
# Response format: {response_schema}
# Authentication: {auth_requirements}
# Validation: {validation_rules}
# Error responses: {error_handling}

@app.route('{endpoint_path}', methods=['{http_method}'])
def {function_name}():
    """
    {endpoint_description}
    """
    # Implementation will be generated here
```

### Database Model Template
```
# Template: Database model/entity
# Create a {model_name} model for {business_entity}
# Database: {database_type}
# ORM: {orm_framework}
# Fields:
# - {field_1}: {type_1} - {description_1}
# - {field_2}: {type_2} - {description_2}
# Relationships: {relationship_description}
# Constraints: {constraint_description}
# Indexes: {index_requirements}

class {model_name}(db.Model):
    """
    {model_description}
    """
    # Model definition will be generated here
```

## Documentation Templates

### Function Documentation Template
```
# Template: Generate comprehensive documentation
# Generate detailed docstring for this function following {documentation_standard}
# Include: purpose, parameters, returns, exceptions, examples
# Style: {style_guide} (Google/Sphinx/NumPy)
# Audience: {target_audience}

def {existing_function}():
    # Existing implementation
    pass
```

### API Documentation Template
```
# Template: API documentation generation
# Generate OpenAPI/Swagger documentation for this endpoint
# Include: parameters, request/response schemas, status codes, examples
# Format: {documentation_format}
# Security: {security_requirements}

@app.route('/api/{resource}')
def {endpoint_function}():
    # Existing implementation
    pass
```

### Class Documentation Template
```
# Template: Class documentation
# Generate comprehensive class documentation
# Include: purpose, usage examples, method overview, inheritance hierarchy
# Format: {documentation_format}
# Include diagrams: {diagram_requirements}

class {existing_class}:
    # Existing implementation
    pass
```

### README Template
```
# Template: Project README generation
# Generate a comprehensive README.md for this {project_type} project
# Include: description, installation, usage, API reference, contributing guidelines
# Audience: {target_audience}
# Sections: {required_sections}
# Style: {documentation_style}

# Project structure context:
# {project_structure_description}
```

## Testing Templates

### Unit Test Template
```
# Template: Comprehensive unit tests
# Generate pytest unit tests for the {class_or_function_name}
# Test scenarios:
# - Happy path with valid inputs
# - Edge cases: {edge_case_description}
# - Error conditions: {error_conditions}
# - Boundary values: {boundary_conditions}
# Mocking: {external_dependencies_to_mock}
# Fixtures: {required_fixtures}

class Test{ClassName}:
    # Test methods will be generated here
```

### Integration Test Template
```
# Template: Integration tests
# Generate integration tests for {component_name}
# Test scope: {integration_scope}
# External systems: {external_systems}
# Test data setup: {test_data_requirements}
# Cleanup requirements: {cleanup_description}
# Performance criteria: {performance_requirements}

class Test{ComponentName}Integration:
    # Integration test methods will be generated here
```

### Performance Test Template
```
# Template: Performance testing
# Generate performance tests for {functionality}
# Load requirements: {load_description}
# Performance criteria: {performance_metrics}
# Test scenarios: {performance_scenarios}
# Monitoring: {metrics_to_track}

def test_{functionality}_performance():
    # Performance test implementation will be generated here
```

### Mock Generation Template
```
# Template: Mock object creation
# Generate mock objects for {external_dependency}
# Mock type: {mock_type}
# Behavior to simulate: {mock_behavior}
# Test scenarios: {test_scenarios}
# Return values: {expected_returns}

@pytest.fixture
def mock_{dependency_name}():
    # Mock implementation will be generated here
```

## Refactoring Templates

### Extract Method Template
```
# Template: Method extraction refactoring
# Extract method from this code block
# New method name: {method_name}
# Purpose: {method_purpose}
# Parameters: {parameter_description}
# Maintain existing behavior: {behavior_requirements}

# Code to refactor:
{existing_code_block}
```

### Design Pattern Implementation Template
```
# Template: Pattern implementation
# Refactor this code to implement {design_pattern} pattern
# Key components: {pattern_components}
# Benefits: {expected_benefits}
# Maintain compatibility: {compatibility_requirements}
# Existing functionality: {existing_behavior}

{existing_code}
```

### Code Optimization Template
```
# Template: Performance optimization
# Optimize this code for {optimization_goal}
# Current performance: {current_metrics}
# Target performance: {target_metrics}
# Constraints: {optimization_constraints}
# Maintain functionality: {functional_requirements}

{code_to_optimize}
```

### Legacy Code Modernization Template
```
# Template: Legacy code update
# Modernize this {language} code to use {modern_features}
# Target version: {target_version}
# New features to adopt: {features_to_use}
# Backward compatibility: {compatibility_requirements}
# Testing strategy: {testing_approach}

{legacy_code}
```

## Architecture Templates

### Microservice Design Template
```
# Template: Microservice implementation
# Design a microservice for {business_domain}
# Responsibilities: {service_responsibilities}
# Communication: {communication_patterns}
# Data storage: {data_requirements}
# Scalability: {scaling_requirements}
# Monitoring: {observability_needs}
# Security: {security_requirements}

class {ServiceName}Service:
    # Service implementation will be generated here
```

### Database Schema Template
```
# Template: Database schema design
# Design database schema for {business_domain}
# Entities: {entity_list}
# Relationships: {relationship_description}
# Performance requirements: {performance_needs}
# Scalability: {scaling_considerations}
# Migration strategy: {migration_approach}

-- Schema definition will be generated here
```

### API Gateway Template
```
# Template: API Gateway configuration
# Configure API gateway for {system_name}
# Routes: {route_definitions}
# Authentication: {auth_strategy}
# Rate limiting: {rate_limiting_rules}
# Load balancing: {load_balancing_strategy}
# Monitoring: {monitoring_requirements}

{api_gateway_config}
```

### Event-Driven Architecture Template
```
# Template: Event system design
# Design event-driven system for {use_case}
# Events: {event_types}
# Publishers: {publisher_services}
# Subscribers: {subscriber_services}
# Message broker: {broker_technology}
# Error handling: {error_strategy}
# Ordering guarantees: {ordering_requirements}

class {EventSystem}:
    # Event system implementation will be generated here
```

## Debugging Templates

### Error Investigation Template
```
# Template: Error analysis and debugging
# Analyze this error and provide debugging steps
# Error message: {error_message}
# Context: {error_context}
# Expected behavior: {expected_outcome}
# Environment: {environment_details}
# Recent changes: {recent_modifications}

{problematic_code}
```

### Performance Debugging Template
```
# Template: Performance issue analysis
# Analyze performance issues in this code
# Symptoms: {performance_symptoms}
# Metrics: {current_metrics}
# Expected performance: {target_metrics}
# Profiling data: {profiling_information}
# Environment: {runtime_environment}

{slow_code}
```

### Memory Leak Investigation Template
```
# Template: Memory leak analysis
# Investigate potential memory leaks in this code
# Symptoms: {memory_symptoms}
# Growth pattern: {memory_growth_pattern}
# Suspected areas: {suspected_code_areas}
# Monitoring data: {memory_metrics}

{suspicious_code}
```

### Concurrency Bug Template
```
# Template: Concurrency issue debugging
# Debug concurrency issues in this multi-threaded code
# Symptoms: {concurrency_symptoms}
# Thread configuration: {thread_setup}
# Shared resources: {shared_state}
# Synchronization: {current_synchronization}
# Race conditions: {potential_races}

{concurrent_code}
```

## Security Templates

### Security Audit Template
```
# Template: Security vulnerability assessment
# Perform security audit of this code
# Security concerns: {security_focus_areas}
# Compliance requirements: {compliance_standards}
# Threat model: {threat_considerations}
# Input validation: {validation_requirements}
# Authentication: {auth_mechanisms}

{code_to_audit}
```

### Secure Implementation Template
```
# Template: Secure coding implementation
# Implement secure {functionality_type}
# Security requirements: {security_standards}
# Input validation: {validation_rules}
# Output encoding: {encoding_requirements}
# Authentication: {auth_requirements}
# Authorization: {authz_requirements}
# Logging: {security_logging}

def secure_{function_name}({parameters}):
    # Secure implementation will be generated here
```

### Cryptography Template
```
# Template: Cryptographic implementation
# Implement cryptographic {crypto_operation}
# Algorithm: {crypto_algorithm}
# Key management: {key_requirements}
# Security level: {security_strength}
# Compliance: {crypto_standards}
# Error handling: {crypto_error_handling}

def {crypto_function}({parameters}):
    # Cryptographic implementation will be generated here
```

### Authentication Template
```
# Template: Authentication system
# Implement authentication system using {auth_method}
# User storage: {user_store}
# Session management: {session_strategy}
# Password policy: {password_requirements}
# Multi-factor: {mfa_requirements}
# Rate limiting: {rate_limiting}

class AuthenticationService:
    # Authentication implementation will be generated here
```

## Performance Templates

### Optimization Analysis Template
```
# Template: Performance optimization analysis
# Analyze and optimize this code for {performance_goal}
# Current bottlenecks: {known_bottlenecks}
# Performance metrics: {current_metrics}
# Target metrics: {performance_targets}
# Constraints: {optimization_constraints}
# Profiling results: {profiling_data}

{code_to_optimize}
```

### Caching Strategy Template
```
# Template: Caching implementation
# Implement caching strategy for {data_type}
# Cache type: {cache_technology}
# TTL strategy: {expiration_policy}
# Invalidation: {invalidation_strategy}
# Consistency: {consistency_requirements}
# Performance goals: {caching_goals}

class CacheManager:
    # Caching implementation will be generated here
```

### Database Query Optimization Template
```
# Template: Query optimization
# Optimize database queries for {query_purpose}
# Current performance: {current_query_metrics}
# Target performance: {target_metrics}
# Index strategy: {indexing_approach}
# Query patterns: {query_optimization_techniques}
# Connection pooling: {connection_strategy}

-- Optimized queries will be generated here
```

### Async Processing Template
```
# Template: Asynchronous processing
# Implement async processing for {processing_task}
# Concurrency model: {async_model}
# Task queue: {queue_technology}
# Error handling: {async_error_handling}
# Progress tracking: {progress_monitoring}
# Scaling strategy: {horizontal_scaling}

async def {async_function}({parameters}):
    # Async implementation will be generated here
```

## Usage Instructions

### How to Use These Templates

1. **Choose the Appropriate Template**
   - Select based on your specific task
   - Consider the programming language and framework
   - Match the complexity level to your needs

2. **Customize the Template**
   - Replace placeholders with specific requirements
   - Add context relevant to your project
   - Include any constraints or preferences

3. **Provide Context**
   - Include relevant existing code
   - Mention the broader system architecture
   - Specify the target environment

4. **Iterate and Refine**
   - Review generated code carefully
   - Provide feedback for improvements
   - Build upon initial suggestions

### Template Customization Examples

#### Basic Customization
```
# Original template
# Create a {function_type} function that {primary_purpose}

# Customized template
# Create a data validation function that validates user registration data
# Requirements:
# - Email format validation
# - Password strength checking
# - Age verification (18+ required)
# - Username uniqueness check
```

#### Advanced Customization
```
# Original template with full context
# Create a caching strategy for user profile data
# Cache type: Redis with clustering
# TTL strategy: 1 hour for active users, 24 hours for inactive
# Invalidation: Real-time on profile updates, batch cleanup daily
# Consistency: Eventually consistent with 5-second max lag
# Performance goals: Sub-100ms cache hits, 99.9% availability
```

### Best Practices for Template Usage

1. **Be Specific**: The more specific your requirements, the better the generated code
2. **Include Context**: Provide relevant background information
3. **Specify Constraints**: Mention any limitations or requirements
4. **Review Output**: Always review and test generated code
5. **Iterate**: Use feedback to improve subsequent generations

### Template Maintenance

- Keep templates updated with current best practices
- Add new templates as patterns emerge
- Remove or update deprecated approaches
- Share successful customizations with the team
