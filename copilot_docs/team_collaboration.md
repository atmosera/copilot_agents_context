## 🏗️ Framework for Team AI Standards

### 1. Shared Context Template
Create a centralized knowledge base that all team members can reference in their AI interactions:

```markdown
# Team AI Development Standards
# consulting_skills_tracker Team Guidelines

## Project Overview
- Application: Flask-based consulting skills tracking system
- Architecture: Service-oriented with Azure integration
- Primary Languages: Python 3.12, TypeScript, SQL
- Infrastructure: Terraform, Azure App Service, Azure Storage

## Universal Context for All AI Interactions

### Technology Stack
- Backend: Flask 2.3+ with Blueprint organization
- Authentication: Azure Entra ID with managed identity
- Data Storage: Azure Blob Storage with JSON documents
- Configuration: Environment variables with .env file support
- Testing: pytest with Azure service mocking
- Deployment: GitHub Actions with Terraform

### Architecture Patterns
- Service Layer: Business logic with comprehensive error handling
- Controller Layer: Request/response handling with Flask Blueprints
- Data Layer: Azure Storage integration with local fallback
- Authentication: Managed identity with credential chaining

### Code Quality Standards
- Type Hints: Required for all function parameters and return values
- Documentation: Comprehensive docstrings with Args/Returns/Examples
- Error Handling: Custom exception classes with structured logging
- Testing: 90%+ coverage with unit and integration tests
- Security: Input validation, parameterized queries, secure headers
```

### 2. Standard Prompt Templates

#### Template: New Feature Development

```markdown
# Team Feature Development Template
Copy and customize this template for all new feature requests:
---

## Context: Consulting Skills Tracker Team Standards

### Project Architecture:
- Flask service-oriented architecture with Azure integration
- Authentication via Azure Entra ID with managed identity
- JSON blob storage in Azure Storage with local fallback
- Service → Controller → Data layer separation

### Code Standards:
- Python 3.12 with comprehensive type hints
- Custom exception classes: ValidationError, BusinessRuleError, SystemError
- Structured logging with correlation IDs
- 90%+ test coverage with pytest and Azure service mocking

### Integration Requirements**:
- Use existing authentication middleware from skills_web_app.authentication
- Follow data patterns from skills_web_app.data.azure_storage
- Implement same error handling as skills_web_app.controller.routes
- Support both Azure cloud and local development environments

## Feature Request: [Describe your specific feature]

### Requirements:
1. Service Layer: Implement business logic following patterns in [existing service file]
2. Controller: Add routes using Flask Blueprint structure
3. Data Integration: Use established Azure Storage patterns with retry logic
4. Testing: Include unit tests with mocked Azure services
5. Documentation: Update relevant API documentation and README

**Quality Checklist**:
- Type hints for all function signatures
- Comprehensive docstrings with examples
- Custom exception handling with correlation IDs
- Structured logging with contextual information
- Input validation using established patterns
- Unit tests with >90% coverage
- Integration tests for complete workflows
- Support for both cloud and local environments

---

Please generate code that follows these established patterns and integrates 
seamlessly with our existing consulting skills tracker application.

{feature_description}
```

#### Template: Code Review and Refactoring

```markdown
# Team Code Review Template

## Context: Consulting Skills Tracker Standards

**Current Code Location**: [File path and line numbers]
**Review Type**: [Bug fix / Performance improvement / Refactoring / Security enhancement]

**Team Standards to Maintain**:
- Custom exception hierarchy (ValidationError, BusinessRuleError, SystemError)
- Correlation ID logging pattern: logger.info(f"Message", extra={"correlation_id": id})
- Type hints and comprehensive docstrings
- Azure service integration with retry logic and fallback

**Review Requirements**:
- Standards Compliance: Ensure code follows team conventions
- Integration Check: Verify compatibility with existing codebase
- Security Review: Check for potential vulnerabilities
- Performance Analysis: Identify optimization opportunities
- Testing Requirements: Suggest test cases if missing

**Specific Areas of Concern**:
- [List any specific issues or areas you want the AI to focus on]

**Please provide**:
- Issues Found**: List any deviations from team standards
- Recommended Changes: Specific code improvements
- Integration Notes: How changes affect other components
- Testing Suggestions: Additional test cases needed

**Code to Review**:
{Code to Review}

```
---

### AI Interaction Guidelines

**Always Include Team Context**:

**Use Established Patterns**:
- Reference existing files when asking for similar functionality
- Follow the same error handling approach across all services
- Use the same authentication and authorization patterns
- Maintain consistent logging and monitoring approaches

#### Create Code Review Process for AI-Generated Code
**Author Responsibilities**:
- Run quality gate validation before creating PR
- Include AI prompt used to generate the code (in PR description)
- Verify integration with existing codebase
- Add comprehensive tests following team patterns

**Reviewer Responsibilities**:
Check team standards compliance using validation checklist
Verify AI-generated code integrates properly
Ensure test coverage meets team requirements (90%+)
Validate documentation and error handling patterns

---

### Example AI Prompts for This Project

**Adding New Service Method**:
```markdown
Following our consulting skills tracker patterns from skills_web_app/data/azure_storage.py:
- Use the same retry logic and error handling
- Implement correlation ID logging
- Support both Azure and local environments
- Include comprehensive type hints and docstrings
- Add appropriate input validation

Create a new method to [describe functionality]
```

**Creating New API Endpoint**:
```markdown
Following our Flask Blueprint patterns from skills_web_app/controller/routes.py:
- Use the same authentication middleware
- Implement consistent error response format  
- Add request validation and sanitization
- Include correlation ID in all log messages
- Follow the same JSON response structure

Create a new endpoint for [describe functionality]
```
