# Security Analysis

## Current Security Posture

This project is a **portfolio/demonstration project**. It showcases MLOps practices but is **NOT production-ready** from a security standpoint.

## Known Security Limitations

### 🔴 CRITICAL (Must Address for Production)

#### 1. No Authentication/Authorization
**Issue**: API endpoints are completely open
- Anyone can make predictions
- Anyone can access drift detection data
- No user tracking or access control

**Risk**: Unauthorized access, abuse, cost implications

**Solution for Production**:
```python
# Add API key authentication
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/predict")
def predict(api_key: str = Depends(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(403, "Invalid API key")
    # ... rest of code
```

#### 2. No Rate Limiting
**Issue**: Single user can overwhelm the API
**Risk**: DDoS, resource exhaustion, service degradation

**Solution for Production**:
```bash
pip install slowapi
```
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("100/minute")
def predict(...):
    pass
```

#### 3. CORS Allows All Origins
**Issue**: `allow_origins=["*"]` permits any website to call the API
**Risk**: Cross-site request forgery, unauthorized access

**Current Code** (src/api/main.py:33-39):
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ TOO PERMISSIVE
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Solution for Production**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains only
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Authorization"],
)
```

#### 4. Sensitive Data in Logs/Database
**Issue**: Prediction logs store all input features (could include PII)
**Risk**: Data breach, privacy violations, GDPR non-compliance

**Mitigation**:
- Encrypt database at rest
- Hash/anonymize PII before logging
- Implement data retention policies
- Add GDPR-compliant data deletion

### 🟡 MEDIUM (Important for Production)

#### 5. No HTTPS/TLS
**Issue**: Currently uses HTTP (unencrypted)
**Risk**: Man-in-the-middle attacks, data interception

**Solution**:
- Deploy behind reverse proxy (Nginx) with SSL
- Use Let's Encrypt for certificates
- Enforce HTTPS redirects

#### 6. No Input Sanitization Beyond Validation
**Issue**: Pydantic validates types but doesn't sanitize
**Risk**: Potential injection attacks if data is used in queries/logs

**Current Protection**: SQLAlchemy ORM (parameterized queries) ✅

#### 7. Database File Permissions
**Issue**: SQLite file (`predictions.db`) has default permissions
**Risk**: Local file access if server is compromised

**Solution**:
```bash
chmod 600 predictions.db  # Owner read/write only
```

#### 8. No Secrets Management
**Issue**: No environment variables or secrets vault
**Risk**: If API keys added later, might be hardcoded

**Solution**:
```python
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("API_KEY")
```

### 🟢 LOW (Nice to Have)

#### 9. No Audit Logging
**Issue**: No logging of who accessed what and when
**Solution**: Add structured logging with user IDs, timestamps, actions

#### 10. No Request Validation Size Limits
**Issue**: Could send massive batch predictions
**Current Protection**: `max_length=100` in schema ✅

#### 11. Model Poisoning Risk
**Issue**: No protection against adversarial inputs
**Mitigation**: Monitor drift detection for anomalous patterns

## Data Privacy Considerations

### Personally Identifiable Information (PII)

The model uses these potentially sensitive features:
- ❌ **Income** - Financial PII
- ❌ **Property Value** - Financial PII
- ❌ **Credit Score** - Highly sensitive PII
- ⚠️ **Gender** - Demographic PII
- ⚠️ **Age** - Demographic PII
- ⚠️ **Region** - Location PII

### Recommendations

1. **Anonymization**: Hash user identifiers before storage
2. **Minimization**: Only log essential features for drift detection
3. **Encryption**: Encrypt database and backups
4. **Access Control**: Restrict who can query prediction logs
5. **Retention**: Auto-delete logs after N days
6. **Consent**: Document data usage in terms of service

## Files Excluded from Git (Security)

✅ Already protected in `.gitignore`:
- `*.db` - Database files with predictions
- `*.env` - Environment variables
- `*.log` - Log files may contain sensitive data
- `data/raw/*.csv` - Original dataset

⚠️ **Verify before pushing**:
```bash
# Check what will be committed
git status

# Look for sensitive files
git ls-files | grep -E "\.db$|\.env$|\.log$|password|secret"
```

## Docker Security

### Current Dockerfile Issues

1. **Running as Root**: Container runs as root user
   ```dockerfile
   # Add before CMD
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```

2. **No Image Scanning**: Should scan for vulnerabilities
   ```bash
   docker scan loan-prediction-api
   ```

3. **Secrets in Build**: Don't add secrets to image layers
   ```dockerfile
   # Use build args for non-sensitive config only
   ARG MODEL_VERSION
   # Use secrets mounts for sensitive data
   ```

## Production Security Checklist

Before deploying to production:

- [ ] Add API authentication (API keys, OAuth, JWT)
- [ ] Implement rate limiting per user/IP
- [ ] Restrict CORS to specific domains
- [ ] Enable HTTPS/TLS
- [ ] Encrypt database at rest
- [ ] Set up secrets management (AWS Secrets Manager, HashiCorp Vault)
- [ ] Add comprehensive audit logging
- [ ] Implement data retention and deletion policies
- [ ] Run security scans (SAST, DAST, dependency scanning)
- [ ] Set up Web Application Firewall (WAF)
- [ ] Implement monitoring and alerting
- [ ] Create incident response plan
- [ ] Conduct penetration testing
- [ ] Review and comply with GDPR/CCPA if applicable
- [ ] Run container as non-root user
- [ ] Scan Docker images for vulnerabilities
- [ ] Set resource limits (memory, CPU)
- [ ] Disable unnecessary endpoints
- [ ] Implement IP whitelisting for admin endpoints

## Responsible Disclosure

This is a portfolio project and not deployed to production. However, if you find security vulnerabilities:

1. **DO NOT** open a public GitHub issue
3. Provide: Description, steps to reproduce, potential impact
4. Allow: 90 days for response before public disclosure

## Compliance Considerations

### GDPR (if processing EU citizen data)
- Right to access
- Right to deletion
- Right to data portability
- Consent management
- Data processing agreements

### Financial Regulations (loan data)
- Fair lending laws
- Anti-discrimination requirements
- Data retention requirements
- Audit trail requirements

## For Portfolio/Demo Use

This project is **safe for portfolio use** because:

✅ No real user data
✅ No production deployment
✅ Clearly marked as demonstration
✅ Sensitive files excluded from git
✅ No hardcoded secrets or API keys

## Summary

| Security Aspect | Current State | Production Ready? |
|----------------|---------------|-------------------|
| Authentication | None | ❌ No |
| Authorization | None | ❌ No |
| Rate Limiting | None | ❌ No |
| CORS Policy | Too permissive | ❌ No |
| HTTPS/TLS | HTTP only | ❌ No |
| Input Validation | Pydantic ✓ | ✅ Yes |
| SQL Injection | ORM ✓ | ✅ Yes |
| Secrets Management | None | ⚠️ Partial |
| Audit Logging | Basic | ⚠️ Partial |
| Data Encryption | None | ❌ No |

**Overall Assessment**: Good for portfolio/demo, **requires significant hardening for production**.
