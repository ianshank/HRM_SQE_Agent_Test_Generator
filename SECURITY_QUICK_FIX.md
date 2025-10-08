# Security Quick Fix Guide

This guide provides immediate actions to address the security vulnerabilities identified by Snyk.

## Immediate Actions (15 minutes)

### 1. Upgrade Dependencies

Run these commands to fix the high-severity issues:

```bash
cd hrm_eval

# Upgrade sentence-transformers (fixes HIGH severity arbitrary code execution)
pip install --upgrade sentence-transformers==3.1.0

# Upgrade starlette (fixes HIGH + MEDIUM severity resource issues)
pip install --upgrade starlette==0.47.2

# Update requirements.txt
pip freeze > requirements-new.txt
```

### 2. Fix Insecure Hash (convert_sqe_data.py)

**File:** `hrm_eval/convert_sqe_data.py`  
**Line:** 93

**Change from:**
```python
hashlib.md5(data.encode()).hexdigest()
```

**Change to:**
```python
hashlib.sha256(data.encode()).hexdigest()
```

### 3. Fix Path Traversal Vulnerabilities (deploy.py)

**File:** `hrm_eval/deploy.py`

Add this function at the top of the file (after imports):

```python
from pathlib import Path

def validate_safe_path(user_path: str, base_dir: str) -> Path:
    """
    Validate path is within base directory to prevent traversal attacks.
    
    Args:
        user_path: User-provided path string
        base_dir: Base directory to constrain paths within
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path traversal attempt detected
    """
    base = Path(base_dir).resolve()
    target = (base / user_path).resolve()
    
    # Ensure target is within base
    try:
        target.relative_to(base)
    except ValueError:
        raise ValueError(
            f"Path traversal detected: {user_path} "
            f"attempts to escape {base_dir}"
        )
    
    return target
```

**Then update lines 159 and 223:**

```python
# Line 159 - Replace:
path = Path(args.path)
# With:
path = validate_safe_path(args.path, base_directory)

# Line 223 - Replace:
path = Path(args.path)
# With:
path = validate_safe_path(args.path, base_directory)
```

Note: You'll need to define `base_directory` appropriately based on your deployment context.

## Verification

After making these changes, verify the fixes:

```bash
# Re-run Snyk scans
cd /Users/iancruickshank/Downloads/hrm_train_us_central1

# Check dependencies
snyk test --file=hrm_eval/requirements.txt --skip-unresolved

# Check code
snyk code test

# Run your test suite
cd hrm_eval
pytest tests/ --ignore=tests/test_integration.py
```

## Expected Results

After fixes:
- **Dependency vulnerabilities:** Reduced from 13 to 10 (PyTorch issues remain, no fix available)
- **Code vulnerabilities:** Reduced from 3 to 0
- **Overall risk:** Significantly reduced from HIGH to MEDIUM

## PyTorch Vulnerabilities (No immediate fix)

The remaining 10 PyTorch vulnerabilities have no patches available yet. Mitigation strategies:

1. **Monitor updates:** Check https://github.com/pytorch/pytorch/releases weekly
2. **Input validation:** Ensure all tensor inputs are validated
3. **Sandboxing:** Consider running PyTorch operations in isolated containers
4. **Risk acceptance:** Document the accepted risk until patches are available

## Full Details

See `SECURITY_ANALYSIS_REPORT.md` for comprehensive details, long-term recommendations, and CI/CD integration.

---

**Estimated time:** 15 minutes for code changes, 30 minutes including testing  
**Priority:** HIGH - Fixes 3 immediately exploitable vulnerabilities  
**Impact:** Eliminates all HIGH severity code issues
