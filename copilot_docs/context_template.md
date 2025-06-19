## 1. Example: Building Session Context

**Initial Request**:
```markdown
I'm working on the consulting skills tracker Flask app. I need to add a new feature 
for tracking consultant certifications.
```

**After 30 minutes of development**:
```markdown
Building on the certification tracking feature we've been developing:
- We established the CertificationService class with Azure Storage integration
- We defined the CertificationValidator with our custom exception patterns
- We created the certification_controller.py with Flask Blueprint structure

Now I need to add the frontend template that integrates with our existing 
authentication flow and follows the same styling as consultant_skills.html.
```


## 2. Progressive Context Building

Build context incrementally, layering new information on established foundations:

**Session 1: Foundation**
```markdown
Context: Consulting Skills Tracker Flask App
- Service-oriented architecture with Azure integration
- Python 3.12 with comprehensive type hints and error handling
- Azure Blob Storage with local development fallback
```

**Session 5: Expanded Context**
```markdown
Context: Consulting Skills Tracker (Building on established patterns)
- CertificationService follows SkillsService architecture
- Uses custom exception hierarchy (ValidationError, BusinessRuleError)
- Implements correlation ID logging like AuthenticationService
- Integrates with existing user management and audit systems
- Supports both Azure and local environments per configuration patterns
```

**Session 10: Rich Context**
```markdown
Context: Consulting Skills Tracker Advanced Features
- Building on certification management foundation we established
- Extends our audit trail system with certification-specific events
- Uses the notification service patterns from skills assessment
- Integrates with external certification APIs using our retry logic
- Follows the same security patterns as sensitive data handling
- Implements manager approval workflow like skills level 4-5 approvals
```

## 3. Context Recovery Strategies

Efficiently rebuild context when returning to previous work:

```markdown
# Context Recovery Template

## Returning to: [Feature/Task Name]

### Quick Project Recap:
**Application**: Flask-based consulting skills tracker with Azure integration
**Current Feature**: [Brief description of what you're working on]
**Last Session Date**: [When you last worked on this]

### Previous Session Summary:
**Completed Work**:
- [✅] Service layer implementation with business logic
- [✅] Data model definition with validation schemas
- [✅] Unit tests with Azure service mocking
- [⏳] Controller integration (partially complete)

**Technical Decisions Made**:
- **Data Structure**: [Brief description of data model decisions]
- **Integration Approach**: [How it connects to existing systems]
- **Error Handling**: [Specific exception and logging patterns chosen]
- **Authentication**: [Security integration decisions]

**Code Files Modified**:
- `skills_web_app/services/certification_service.py` - Core business logic
- `skills_web_app/data/certification_data.py` - Data models and validation
- `tests/test_certification_service.py` - Unit tests with mocking

### Current State Assessment:
**What's Working**:
- Service layer handles all business logic correctly
- Data validation catches all edge cases tested
- Azure Storage integration works in both cloud and local modes

**Outstanding Issues**:
- [❌] Controller routes need authentication integration
- [❌] Frontend templates need styling consistency
- [❌] Manager approval workflow not yet implemented

### Today's Context and Goals:
Building on the solid foundation we established, today I need to:
1. **Complete controller integration** following our Flask Blueprint patterns
2. **Add frontend templates** consistent with existing consultant profile UI
3. **Implement authentication** using our established middleware
4. **Test complete workflow** from frontend to storage

Please help me continue this work using the same patterns and standards 
we established in previous sessions.
```

## 4. Database Schema Integration

Make AI aware of your data structures for better code generation:
## Consultant Profile Schema
```json
{
  "consultant_id": "string (UUID format, required)",
  "personal_info": {
    "name": "string (required, max 100 chars)",
    "email": "string (email format, required)",
    "phone": "string (optional, format: +1-xxx-xxx-xxxx)",
    "location": "string (optional, city/state format)"
  },
  "employment_info": {
    "hire_date": "string (ISO date format)",
    "department": "string (enum: Technology, Business, Data, Strategy)",
    "level": "string (enum: Associate, Senior, Principal, Director)",
    "manager_id": "string (references consultant_id)"
  },
  "skills": {
    "technical_skills": {
      "[skill_name]": {
        "level": "integer (1-5, required)",
        "last_assessed": "string (ISO datetime)",
        "assessor_id": "string (references consultant_id)",
        "evidence": ["array of strings (optional)"]
      }
    },
    "soft_skills": {
      "[skill_name]": {
        "level": "integer (1-5, required)",
        "last_assessed": "string (ISO datetime)",
        "assessor_id": "string (references consultant_id)"
      }
    }
  },
  "certifications": {
    "[certification_id]": {
      "name": "string (required)",
      "issuing_organization": "string (required)",
      "issue_date": "string (ISO date)",
      "expiry_date": "string (ISO date, optional)",
      "credential_id": "string (optional)",
      "validation_status": "string (enum: verified, pending, expired)"
    }
  },
  "project_history": [
    {
      "project_id": "string (UUID)",
      "project_name": "string",
      "role": "string",
      "start_date": "string (ISO date)",
      "end_date": "string (ISO date, optional)",
      "skills_used": ["array of skill names"],
      "client_industry": "string (optional)"
    }
  ],
  "metadata": {
    "created_date": "string (ISO datetime)",
    "last_updated": "string (ISO datetime)",
    "version": "integer (for optimistic locking)"
  }
}
```

## 5. API Integration Context

Provide AI with external API specifications:

```markdown
# Microsoft Graph API Integration Context

## Authentication
- Method: Managed Identity with Graph API permissions
- Scopes: User.Read, User.Read.All, Directory.Read.All
- Token Caching: 1-hour cache with automatic refresh
- Error Handling: GraphAPIError with retry logic

## Common Endpoints
# User Profile Endpoint
GET https://graph.microsoft.com/v1.0/users/{user-id}
Response: {
    "id": "string",
    "displayName": "string", 
    "mail": "string",
    "jobTitle": "string",
    "department": "string",
    "manager": {"@odata.id": "string"}
}

# Manager Information
GET https://graph.microsoft.com/v1.0/users/{user-id}/manager
Response: {
    "id": "string",
    "displayName": "string",
    "mail": "string"
}

# Direct Reports
GET https://graph.microsoft.com/v1.0/users/{user-id}/directReports
Response: {
    "value": [
        {
            "id": "string",
            "displayName": "string",
            "mail": "string"
        }
    ]
}
```
