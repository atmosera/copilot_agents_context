### 1. Coding Style Preferences

#### Python Style Example
**Generic AI Output**:
```python
def get_data(id):
    data = database.query("SELECT * FROM table WHERE id = " + str(id))
    return data
```
#### Training Prompt for Style Preferences

```markdown
## My Python Coding Style Preferences

Please always follow these patterns when generating Python code:

**Function Structure**:
- Use descriptive function names with underscores (get_consultant_data_by_id)
- Include comprehensive type hints for all parameters and return values
- Add detailed docstrings with Args, Returns, Raises, and Example sections

**Error Handling**:
- Use custom exception classes (ValidationError, DatabaseError, BusinessRuleError)
- Include correlation IDs for tracing requests across services
- Implement structured logging with contextual information
- Provide specific error messages with actionable information

**Logging Standards**:
- Use correlation IDs in all log messages
- Include relevant context in extra fields
- Log at appropriate levels (info for success, error for failures)
- Follow format: logger.info(f"Description", extra={"correlation_id": id})

**Validation Patterns**:
- Validate all input parameters at function entry
- Use regex patterns for format validation
- Raise ValidationError with descriptive messages
- Check for None values and empty strings

**Database Interactions**:
- Always use parameterized queries to prevent SQL injection
- Handle DatabaseError exceptions specifically
- Include timeout and retry logic for production systems
- Return domain objects rather than raw database results

**Code Organization**:
- Separate concerns into distinct functions
- Use early returns to reduce nesting
- Group related functionality into logical blocks
- Include comprehensive comments for business logic
```
**After Fine-Tuning to Your Style**:
```python
def get_consultant_data_by_id(consultant_id: str) -> Optional[ConsultantData]:
    """
    Retrieve consultant data with comprehensive error handling and logging.
    
    Args:
        consultant_id: Unique identifier for the consultant
        
    Returns:
        ConsultantData object if found, None if not found
        
    Raises:
        ValidationError: If consultant_id format is invalid
        DatabaseError: If database operation fails
        
    Example:
        >>> consultant = get_consultant_data_by_id("consultant_123")
        >>> print(consultant.name if consultant else "Not found")
    """
    correlation_id = generate_correlation_id()
    logger.info(
        f"Retrieving consultant data for ID: {consultant_id}",
        extra={"correlation_id": correlation_id, "consultant_id": consultant_id}
    )
    
    # Validate input parameters
    if not consultant_id or not isinstance(consultant_id, str):
        raise ValidationError("Consultant ID must be a non-empty string")
    
    if not re.match(r'^consultant_[a-zA-Z0-9]+$', consultant_id):
        raise ValidationError(f"Invalid consultant ID format: {consultant_id}")
    
    try:
        # Use parameterized query to prevent SQL injection
        result = self.database.execute_query(
            "SELECT * FROM consultants WHERE consultant_id = %s AND active = true",
            (consultant_id,)
        )
        
        if result:
            consultant_data = ConsultantData.from_database_row(result[0])
            logger.info(
                f"Successfully retrieved consultant data",
                extra={
                    "correlation_id": correlation_id,
                    "consultant_id": consultant_id,
                    "data_fields": list(consultant_data.__dict__.keys())
                }
            )
            return consultant_data
        else:
            logger.info(
                f"No consultant found with ID: {consultant_id}",
                extra={"correlation_id": correlation_id}
            )
            return None
            
    except DatabaseError as e:
        logger.error(
            f"Database error retrieving consultant: {str(e)}",
            extra={"correlation_id": correlation_id}
        )
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error retrieving consultant: {str(e)}",
            extra={"correlation_id": correlation_id}
        )
        raise DatabaseError(f"Failed to retrieve consultant data: {str(e)}")
```

### 2. Architecture Pattern Alignment

#### Project-Specific Context

```markdown
## Consulting Skills Tracker - Architecture Context

**Project Structure**:
```
consulting_skills_tracker/
├── python_skills_app/
│   ├── skills_web_app/
│   │   ├── authentication/     # Entra ID integration
│   │   ├── controller/         # Flask routes and request handling
│   │   ├── data/              # Azure Storage and data persistence
│   │   └── state_management/   # Application configuration and state
│   └── app.py                 # Application entry point
├── iac/                       # Terraform infrastructure
└── .github/workflows/         # CI/CD pipelines
```

**Established Patterns**:

1. **Authentication Flow**:
   - Use managed identity for Azure services
   - Implement credential chaining: ManagedIdentity → AzureCLI → DefaultAzure
   - Environment-aware authentication (cloud vs local development)
   - Session management via Flask-Session

2. **Data Layer**:
   - JSON blob storage in Azure Storage
   - Local file fallback for development
   - Comprehensive error handling with retry logic
   - Schema validation using jsonschema library

3. **Controller Layer**:
   - Flask Blueprint organization
   - Consistent error response formatting
   - Request validation middleware
   - Correlation ID propagation

4. **Configuration Management**:
   - Environment variables with .env file support
   - Different configs for cloud vs local environments
   - Secure secret management through Azure Key Vault

**When generating code for this project, always**:
- Follow the established directory structure
- Use the existing authentication patterns
- Implement the same error handling and logging approach
- Support both Azure and local development environments
- Include comprehensive type hints and documentation
```

### 3. Language-Specific Fine-Tuning
**C++ Style Guidelines**:
```markdown
## My C++ Coding Preferences

**Memory Management**:
- Use smart pointers (unique_ptr, shared_ptr) instead of raw pointers
- Follow RAII principles for resource management
- Prefer move semantics over copy operations
- Delete copy constructor/assignment when appropriate

**Error Handling**:
- Use exceptions for error conditions
- Create custom exception classes inheriting from std::exception
- Provide descriptive error messages
- Use std::optional for functions that might not return a value

**Thread Safety**:
- Use std::mutex for thread synchronization
- Employ RAII lock guards (std::lock_guard, std::unique_lock)
- Mark thread-safe methods as const when possible
- Document thread safety guarantees

**Modern C++ Features**:
- Use auto for type deduction when it improves readability
- Prefer range-based for loops
- Use constexpr for compile-time constants
- Employ structured bindings for multiple return values

**Code Organization**:
- Separate interface (.h) from implementation (.cpp)
- Use namespace organization
- Include comprehensive Doxygen comments
- Follow consistent naming conventions (snake_case for variables, PascalCase for classes)
```
**After Fine-Tuning Style**
```cpp
// Training Example: Memory Management and RAII
class ConsultantSkillsManager {
private:
    std::unique_ptr<DatabaseConnection> db_connection_;
    std::shared_ptr<Logger> logger_;
    mutable std::mutex skills_mutex_;
    
public:
    explicit ConsultantSkillsManager(
        std::unique_ptr<DatabaseConnection> db_connection,
        std::shared_ptr<Logger> logger = nullptr
    ) : db_connection_(std::move(db_connection)),
        logger_(logger ? logger : std::make_shared<ConsoleLogger>()) {
        
        if (!db_connection_) {
            throw std::invalid_argument("Database connection cannot be null");
        }
        
        logger_->info("ConsultantSkillsManager initialized successfully");
    }
    
    // No copy constructor/assignment - use move semantics
    ConsultantSkillsManager(const ConsultantSkillsManager&) = delete;
    ConsultantSkillsManager& operator=(const ConsultantSkillsManager&) = delete;
    
    // Move constructor and assignment
    ConsultantSkillsManager(ConsultantSkillsManager&&) = default;
    ConsultantSkillsManager& operator=(ConsultantSkillsManager&&) = default;
    
    std::optional<ConsultantSkills> GetConsultantSkills(
        const std::string& consultant_id
    ) const {
        std::lock_guard<std::mutex> lock(skills_mutex_);
        
        if (consultant_id.empty()) {
            logger_->error("Empty consultant ID provided");
            return std::nullopt;
        }
        
        try {
            auto query_result = db_connection_->ExecuteQuery(
                "SELECT * FROM consultant_skills WHERE consultant_id = ?",
                {consultant_id}
            );
            
            if (query_result.empty()) {
                logger_->info("No skills found for consultant: " + consultant_id);
                return std::nullopt;
            }
            
            return ConsultantSkills::FromDatabaseRow(query_result[0]);
            
        } catch (const DatabaseException& e) {
            logger_->error("Database error: " + std::string(e.what()));
            throw;
        } catch (const std::exception& e) {
            logger_->error("Unexpected error: " + std::string(e.what()));
            throw ConsultantSkillsException("Failed to retrieve skills");
        }
    }
};
```

**Java Spring Boot Style Guidelines**:
```markdown
## My Java Spring Boot Preferences

**Dependency Injection**:
- Use constructor injection instead of field injection
- Make all injected fields final
- Use @RequiredArgsConstructor from Lombok when appropriate
- Validate dependencies in constructor or with @PostConstruct

**Transaction Management**:
- Use @Transactional at service layer methods
- Specify appropriate propagation and isolation levels
- Handle exceptions that cause rollback appropriately
- Keep transactions as short as possible

**Logging and Monitoring**:
- Use SLF4J with Logback
- Include correlation IDs using MDC
- Log at appropriate levels with structured messages
- Integrate metrics collection with Micrometer
- Use @Timed for method-level performance monitoring

**Error Handling**:
- Create custom exception classes for different error types
- Use @ControllerAdvice for global exception handling
- Include correlation IDs in error responses
- Log exceptions with appropriate context

**Validation**:
- Use Bean Validation annotations (@Valid, @NotNull, etc.)
- Create custom validators for business rules
- Validate at service boundaries
- Return structured validation results

**Testing**:
- Use @SpringBootTest for integration tests
- Mock external dependencies with @MockBean
- Use TestContainers for database integration tests
- Follow AAA pattern (Arrange, Act, Assert)
```

**After Fine-Tuning Style**

```java
// Training Example: Service Layer with Proper Dependency Injection
@Service
@Transactional
@Slf4j
public class ConsultantSkillsService {
    
    private final ConsultantRepository consultantRepository;
    private final SkillsRepository skillsRepository;
    private final ValidationService validationService;
    private final AuditService auditService;
    private final MetricsService metricsService;
    
    // Constructor injection (preferred over field injection)
    public ConsultantSkillsService(
            ConsultantRepository consultantRepository,
            SkillsRepository skillsRepository,
            ValidationService validationService,
            AuditService auditService,
            MetricsService metricsService) {
        this.consultantRepository = consultantRepository;
        this.skillsRepository = skillsRepository;
        this.validationService = validationService;
        this.auditService = auditService;
        this.metricsService = metricsService;
    }
    
    @Retryable(
        value = {DataAccessException.class},
        maxAttempts = 3,
        backoff = @Backoff(delay = 1000, multiplier = 2)
    )
    public SkillsUpdateResult updateConsultantSkills(
            @Valid @NotNull String consultantId,
            @Valid @NotNull SkillsUpdateRequest request) {
        
        // Create correlation ID for request tracing
        String correlationId = UUID.randomUUID().toString();
        MDC.put("correlationId", correlationId);
        MDC.put("consultantId", consultantId);
        
        log.info("Starting skills update for consultant: {}", consultantId);
        Timer.Sample sample = Timer.start(metricsService.getMeterRegistry());
        
        try {
            // Validate consultant exists and is active
            Consultant consultant = consultantRepository.findById(consultantId)
                .orElseThrow(() -> new ConsultantNotFoundException(
                    String.format("Consultant not found: %s", consultantId)));
            
            if (!consultant.isActive()) {
                throw new BusinessRuleException(
                    String.format("Cannot update skills for inactive consultant: %s", 
                                consultantId));
            }
            
            // Validate request data
            ValidationResult validationResult = validationService
                .validateSkillsUpdate(request);
            
            if (!validationResult.isValid()) {
                log.warn("Skills validation failed for consultant: {}, errors: {}", 
                        consultantId, validationResult.getErrorMessages());
                throw new ValidationException(
                    "Skills validation failed", 
                    validationResult.getErrors());
            }
            
            // Perform the update
            List<ConsultantSkill> updatedSkills = performSkillsUpdate(
                consultant, request);
            
            // Save changes and audit
            skillsRepository.saveAll(updatedSkills);
            auditService.recordSkillsUpdate(
                consultantId, 
                request.getUpdatedBy(), 
                updatedSkills);
            
            // Record success metrics
            metricsService.incrementCounter("skills.update.success");
            sample.stop(Timer.builder("skills.update.duration")
                .tag("result", "success")
                .register(metricsService.getMeterRegistry()));
            
            log.info("Successfully updated {} skills for consultant: {}", 
                    updatedSkills.size(), consultantId);
            
            return SkillsUpdateResult.builder()
                .consultantId(consultantId)
                .updatedSkills(updatedSkills.stream()
                    .map(SkillDto::fromEntity)
                    .collect(Collectors.toList()))
                .validationWarnings(validationResult.getWarnings())
                .updateTimestamp(Instant.now())
                .correlationId(correlationId)
                .build();
                
        } catch (ConsultantNotFoundException | BusinessRuleException e) {
            metricsService.incrementCounter("skills.update.error", "type", "business");
            log.error("Business rule violation during skills update: {}", e.getMessage());
            throw e;
        } catch (ValidationException e) {
            metricsService.incrementCounter("skills.update.error", "type", "validation");
            log.error("Validation error during skills update: {}", e.getMessage());
            throw e;
        } catch (Exception e) {
            metricsService.incrementCounter("skills.update.error", "type", "system");
            log.error("Unexpected error during skills update", e);
            throw new SkillsUpdateException(
                "Failed to update consultant skills", e);
        } finally {
            MDC.clear();
        }
    }
    
    private List<ConsultantSkill> performSkillsUpdate(
            Consultant consultant, 
            SkillsUpdateRequest request) {
        
        return request.getSkillUpdates().entrySet().stream()
            .map(entry -> updateOrCreateSkill(consultant, entry.getKey(), entry.getValue()))
            .collect(Collectors.toList());
    }
    
    private ConsultantSkill updateOrCreateSkill(
            Consultant consultant, 
            String skillName, 
            Integer skillLevel) {
        
        return consultant.getSkills().stream()
            .filter(skill -> skill.getSkillName().equals(skillName))
            .findFirst()
            .map(skill -> {
                skill.setSkillLevel(skillLevel);
                skill.setLastUpdated(Instant.now());
                return skill;
            })
            .orElse(ConsultantSkill.builder()
                .consultant(consultant)
                .skillName(skillName)
                .skillLevel(skillLevel)
                .createdDate(Instant.now())
                .lastUpdated(Instant.now())
                .build());
    }
}
```



---

## 🎯 Advanced Fine-Tuning Techniques

### 1. Context Stacking

Build context incrementally across conversations:

**Session 1: Foundation**
```markdown
I'm working on a Python Flask application for tracking consultant skills. 
The app uses Azure Blob Storage for data persistence and Entra ID for authentication.
I prefer comprehensive error handling, structured logging, and type hints.
```

**Session 3: Pattern Reinforcement**
```markdown
Building on our previous work with the consultant skills tracker:
- Follow the same authentication patterns we established
- Use the ValidationError and BusinessRuleError exceptions we defined
- Implement the same correlation ID logging pattern
- Support both Azure and local development environments

Now I need to add a new feature for...
```

**Session 10: Advanced Integration**
```markdown
Continuing with our consulting skills tracker patterns:
- Use established service layer architecture from skills_service.py
- Follow the same JSON schema validation approach
- Integrate with the audit logging we implemented
- Maintain the same API response format

For this new endpoint, I need to...
```
---
### 2. Pattern Documentation

Create a personal style guide that you reference in AI conversations:

```markdown
## My Development Style Guide

### Python Patterns
**File Organization**:
- Services: Business logic with comprehensive error handling
- Controllers: Request/response handling with validation
- Models: Data structures with type hints and validation
- Utils: Shared functionality and helper methods

**Error Handling**:
- Custom exception hierarchy (ValidationError, BusinessRuleError, SystemError)
- Correlation IDs for request tracing
- Structured logging with contextual information
- Graceful degradation for external service failures

**Testing Patterns**:
- Pytest with fixtures for common test data
- Mock external dependencies (Azure services, databases)
- Parameterized tests for input validation
- Integration tests for complete workflows

### Code Quality Standards
- 90%+ test coverage for new code
- Type hints for all function signatures
- Comprehensive docstrings with examples
- Consistent naming conventions
- Performance considerations for database queries

### Architecture Decisions
- Service-oriented architecture with clear boundaries
- Environment-aware configuration (cloud vs local)
- Stateless service design for scalability
- Event-driven updates for data synchronization
```

### 3. Feedback Loop Optimization

Structure feedback to improve future suggestions:

**Effective Feedback Example**:
```markdown
The code you generated is close, but please adjust:

✅ Good: The error handling structure and logging format
✅ Good: The type hints and docstring style
❌ Needs change: Use our custom ValidationError instead of ValueError
❌ Needs change: Add correlation ID to all log messages
❌ Needs change: Include retry logic for Azure Storage operations
❌ Needs change: Support both cloud and local environments

Pattern to follow: Look at skills_service.py lines 45-80 for the exact 
error handling and logging pattern I prefer.
```

**Ineffective Feedback Example**:
```markdown
This doesn't look right. Make it better and follow our standards.
```


