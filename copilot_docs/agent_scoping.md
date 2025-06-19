## 🎯 Scoping GitHub Copilot Agents

### 1. Architecture Agent
**Primary Use**: High-level design and structural guidance
**Example Interaction**:
```markdown
You: "I need to add a new skills assessment feature to our consulting tracker. 
How should I structure this to integrate with our existing Flask architecture?"

Agent: "Based on your current structure, I recommend:

1. Service Layer: Create `skills_assessment_service.py` following the pattern 
   from your existing services
2. Controller: Add routes to `controller/routes.py` with Blueprint organization
3. Data Layer: Extend `data/data_handler.py` with assessment-specific methods
4. Templates: Create assessment templates in the same structure as existing forms

This maintains your separation of concerns and integrates with your current 
authentication and state management patterns."
```

### 2. Security Agent
**Primary Use**: Security-focused code generation and review
**Security Pattern Example**:
```python
# Security Agent generates secure input validation
def validate_consultant_skills_input(skills_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize consultant skills input with comprehensive security checks.
    
    Security measures implemented:
    - Input sanitization against XSS attacks
    - SQL injection prevention through parameterized queries
    - Data type validation and bounds checking
    - Rate limiting for API endpoints
    """
    # Sanitize all string inputs
    sanitized_data = {}
    for key, value in skills_data.items():
        if isinstance(value, str):
            # Remove potential XSS vectors
            sanitized_value = bleach.clean(
                value, 
                tags=[], 
                attributes={}, 
                strip=True
            )
            # Validate length constraints
            if len(sanitized_value) > MAX_FIELD_LENGTH:
                raise ValidationError(f"Field {key} exceeds maximum length")
            sanitized_data[key] = sanitized_value
        elif isinstance(value, (int, float)):
            # Validate numeric ranges
            if not MIN_SKILL_LEVEL <= value <= MAX_SKILL_LEVEL:
                raise ValidationError(f"Skill level {value} out of valid range")
            sanitized_data[key] = value
        else:
            raise ValidationError(f"Invalid data type for field {key}")
    
    return sanitized_data
```

### 3. Testing Agent
**Primary Use**: Test generation and quality assurance
**Test Generation Example**:
```python
# Generate comprehensive test suite
class TestConsultantSkillsService:
    """Comprehensive test suite for ConsultantSkillsService."""
    
    @pytest.fixture
    def skills_service(self):
        """Create a properly configured skills service for testing."""
        mock_data_handler = Mock(spec=DataHandler)
        mock_auth_service = Mock(spec=AuthenticationService)
        return ConsultantSkillsService(
            data_handler=mock_data_handler,
            auth_service=mock_auth_service
        )
    
    @pytest.fixture
    def sample_skills_data(self):
        """Sample skills data for testing."""
        return {
            "consultant_id": "test-123",
            "technical_skills": {
                "python": 5,
                "java": 3,
                "azure": 4
            },
            "soft_skills": {
                "communication": 5,
                "leadership": 4
            }
        }
    
    def test_calculate_billable_hours_success(self, skills_service, sample_skills_data):
        """Test successful billable hours calculation."""
        # Arrange
        expected_result = BillingCalculationResult(
            total_hours=40.0,
            total_amount=Decimal("4000.00"),
            validation_status="valid"
        )
        skills_service.data_handler.get_consultant.return_value = {
            "id": "test-123",
            "status": "active"
        }
        
        # Act
        result = skills_service.calculate_consultant_billable_hours(
            consultant_id="test-123",
            time_entries=[],
            billing_rate=Decimal("100.00")
        )
        
        # Assert
        assert result.total_hours == expected_result.total_hours
        assert result.total_amount == expected_result.total_amount
        skills_service.data_handler.get_consultant.assert_called_once_with("test-123")
    
    def test_calculate_billable_hours_invalid_consultant(self, skills_service):
        """Test error handling for invalid consultant."""
        # Arrange
        skills_service.data_handler.get_consultant.side_effect = ValidationError(
            "Consultant not found"
        )
        
        # Act & Assert
        with pytest.raises(ValidationError, match="Consultant not found"):
            skills_service.calculate_consultant_billable_hours(
                consultant_id="invalid-123",
                time_entries=[],
                billing_rate=Decimal("100.00")
            )
    
    @pytest.mark.parametrize("billing_rate,expected_error", [
        (Decimal("-10.00"), "Billing rate must be positive"),
        (Decimal("0.00"), "Billing rate must be positive"),
        (Decimal("10000.00"), "Billing rate exceeds maximum allowed")
    ])
    def test_calculate_billable_hours_invalid_rates(
        self, 
        skills_service, 
        billing_rate, 
        expected_error
    ):
        """Test validation of billing rates."""
        with pytest.raises(ValidationError, match=expected_error):
            skills_service.calculate_consultant_billable_hours(
                consultant_id="test-123",
                time_entries=[],
                billing_rate=billing_rate
            )
```
---

## 🛠️ IDE-Specific Agent Integration

### VS Code Integration
**Features**:
- Seamless inline suggestions
- Chat sidebar for complex queries
- Workspace-aware context
- Extension ecosystem integration

**Optimization Tips**:
```json
// .vscode/settings.json
{
    "github.copilot.enable": {
        "*": true,
        "yaml": true,
        "plaintext": false,
        "markdown": true
    },
    "github.copilot.inlineSuggest.enable": true,
    "github.copilot.chat.localeOverride": "en"
}
```

### IntelliJ IDEA/PyCharm Integration
**Features**:
- IDE-native code completion
- Refactoring assistance
- Project structure awareness
- Plugin ecosystem compatibility

**Configuration**:
```xml
<!-- .idea/workspace.xml -->
<component name="PropertiesComponent">
    <property name="copilot.enabled" value="true" />
    <property name="copilot.inline.suggestions" value="true" />
    <property name="copilot.chat.enabled" value="true" />
</component>
```

### Eclipse Integration
**Features**:
- Project-aware suggestions
- Build path integration
- Team workspace sharing
- Plugin coordination

**Setup Considerations**:
- Ensure Copilot plugin is installed for all team members
- Configure shared project settings for consistency
- Integrate with existing Eclipse plugins and workflows

---
## 🎯 Language-Specific Agent Scoping

### Python Development
**Example: Flask Route Generation**:
```python
# Create a Flask route for updating consultant skills with validation
@app.route('/consultant/<consultant_id>/skills', methods=['PUT'])
@login_required
def update_consultant_skills(consultant_id: str) -> Response:
    """
    Update consultant skills with comprehensive validation and error handling.
    
    Args:
        consultant_id: Unique identifier for the consultant
        
    Returns:
        JSON response with update status and any validation errors
    """
    try:
        # Validate user permissions
        if not current_user.can_modify_consultant(consultant_id):
            return jsonify({
                "error": "Insufficient permissions",
                "code": "PERMISSION_DENIED"
            }), 403
        
        # Validate and parse request data
        skills_data = request.get_json()
        if not skills_data:
            return jsonify({
                "error": "Request body is required",
                "code": "MISSING_DATA"
            }), 400
        
        # Update skills using service layer
        result = skills_service.update_consultant_skills(
            consultant_id=consultant_id,
            skills_data=skills_data,
            updated_by=current_user.id
        )
        
        return jsonify({
            "success": True,
            "consultant_id": consultant_id,
            "updated_skills": result.updated_skills,
            "validation_warnings": result.warnings
        }), 200
    # Handle specific errors    
    except ValidationError as e:
        logger.warning(f"Skills validation failed: {e}")
        return jsonify({
            "error": str(e),
            "code": "VALIDATION_ERROR"
        }), 400
        
    except Exception as e:
        logger.error(f"Unexpected error updating skills: {e}")
        return jsonify({
            "error": "Internal server error",
            "code": "INTERNAL_ERROR"
        }), 500
```

### C/C++ Development
**Example: Memory-Safe Implementation**:
```cpp
// Create a safe string manipulation class following RAII principles
class SafeString {
private:
    char* data_;
    size_t size_;
    size_t capacity_;
    
    void ensure_capacity(size_t required_size) {
        if (required_size > capacity_) {
            size_t new_capacity = std::max(required_size, capacity_ * 2);
            char* new_data = new char[new_capacity];
            
            if (data_) {
                std::memcpy(new_data, data_, size_);
                delete[] data_;
            }
            
            data_ = new_data;
            capacity_ = new_capacity;
        }
    }
    
public:
    // Constructor
    explicit SafeString(const char* initial = "") 
        : data_(nullptr), size_(0), capacity_(0) {
        if (initial) {
            size_t len = std::strlen(initial);
            ensure_capacity(len + 1);
            std::strcpy(data_, initial);
            size_ = len;
        }
    }
    
    // Copy constructor
    SafeString(const SafeString& other) 
        : data_(nullptr), size_(0), capacity_(0) {
        if (other.data_) {
            ensure_capacity(other.size_ + 1);
            std::strcpy(data_, other.data_);
            size_ = other.size_;
        }
    }
    
    // Move constructor
    SafeString(SafeString&& other) noexcept 
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    
    // Destructor
    ~SafeString() {
        delete[] data_;
    }
    
    // Assignment operators
    SafeString& operator=(const SafeString& other) {
        if (this != &other) {
            SafeString temp(other);
            swap(temp);
        }
        return *this;
    }
    
    SafeString& operator=(SafeString&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }
    
    // Safe string operations
    void append(const char* str) {
        if (!str) return;
        
        size_t len = std::strlen(str);
        ensure_capacity(size_ + len + 1);
        std::strcat(data_, str);
        size_ += len;
    }
    
    const char* c_str() const {
        return data_ ? data_ : "";
    }
    
    size_t length() const {
        return size_;
    }
    
private:
    void swap(SafeString& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }
};
```

### Java Development
**Example: Spring Service Implementation**:
```java
// Create a Spring service for consultant skills management with proper dependency injection
@Service
@Transactional
public class ConsultantSkillsService {
    
    private final ConsultantRepository consultantRepository;
    private final SkillsRepository skillsRepository;
    private final ValidationService validationService;
    private final AuditService auditService;
    
    private static final Logger logger = LoggerFactory.getLogger(ConsultantSkillsService.class);
    
    public ConsultantSkillsService(
            ConsultantRepository consultantRepository,
            SkillsRepository skillsRepository,
            ValidationService validationService,
            AuditService auditService) {
        this.consultantRepository = consultantRepository;
        this.skillsRepository = skillsRepository;
        this.validationService = validationService;
        this.auditService = auditService;
    }
    
    public SkillsUpdateResult updateConsultantSkills(
            String consultantId, 
            SkillsUpdateRequest request) {
        
        logger.info("Starting skills update for consultant: {}", consultantId);
        
        try {
            // Validate consultant exists and is active
            Consultant consultant = consultantRepository.findById(consultantId)
                .orElseThrow(() -> new ConsultantNotFoundException(
                    "Consultant not found: " + consultantId));
            
            if (!consultant.isActive()) {
                throw new BusinessRuleException(
                    "Cannot update skills for inactive consultant: " + consultantId);
            }
            
            // Validate skills data
            ValidationResult validationResult = validationService
                .validateSkillsUpdate(request);
            
            if (!validationResult.isValid()) {
                throw new ValidationException(
                    "Skills validation failed", validationResult.getErrors());
            }
            
            // Update skills
            List<ConsultantSkill> updatedSkills = updateSkillsInternal(
                consultant, request.getSkills());
            
            // Save changes
            skillsRepository.saveAll(updatedSkills);
            
            // Audit the change
            auditService.recordSkillsUpdate(
                consultantId, 
                request.getUpdatedBy(), 
                updatedSkills);
            
            logger.info("Successfully updated {} skills for consultant: {}", 
                       updatedSkills.size(), consultantId);
            
            return SkillsUpdateResult.builder()
                .consultantId(consultantId)
                .updatedSkills(updatedSkills)
                .validationWarnings(validationResult.getWarnings())
                .updateTimestamp(Instant.now())
                .build();
                
        } catch (Exception e) {
            logger.error("Error updating skills for consultant: {}", consultantId, e);
            throw new SkillsUpdateException(
                "Failed to update consultant skills", e);
        }
    }
    
    private List<ConsultantSkill> updateSkillsInternal(
            Consultant consultant, 
            Map<String, Integer> skillUpdates) {
        
        List<ConsultantSkill> updatedSkills = new ArrayList<>();
        
        for (Map.Entry<String, Integer> entry : skillUpdates.entrySet()) {
            String skillName = entry.getKey();
            Integer skillLevel = entry.getValue();
            
            // Find existing skill or create new one
            ConsultantSkill skill = consultant.getSkills().stream()
                .filter(s -> s.getSkillName().equals(skillName))
                .findFirst()
                .orElse(ConsultantSkill.builder()
                    .consultant(consultant)
                    .skillName(skillName)
                    .build());
            
            skill.setSkillLevel(skillLevel);
            skill.setLastUpdated(Instant.now());
            
            updatedSkills.add(skill);
        }
        
        return updatedSkills;
    }
    
    @Retryable(value = {DataAccessException.class}, maxAttempts = 3)
    public Optional<ConsultantSkills> getConsultantSkills(String consultantId) {
        logger.debug("Retrieving skills for consultant: {}", consultantId);
        
        return consultantRepository.findById(consultantId)
            .map(consultant -> ConsultantSkills.builder()
                .consultantId(consultant.getId())
                .skills(consultant.getSkills())
                .lastUpdated(consultant.getSkillsLastUpdated())
                .build());
    }
}
```