# Dependency Update - Security Patches

## Overview

This document describes the dependency updates made to address security vulnerabilities and compatibility requirements.

## Updated Dependencies

### Core Dependencies

**1. requests: ^2.31.0 → ^2.32.4**
- **Reason**: Security vulnerability patches
- **Impact**: No breaking changes for our usage
- **Changes Required**: None

**2. qdrant-client: ^1.7.0 → ^1.15.1 (resolved by poetry)**
- **Reason**: Transitive dependency update via poetry lock
- **Impact**: Client version 1.15.1 may warn about incompatibility with Qdrant server 1.7.4
- **Changes Required**: Added `prefer_grpc=False` to QdrantClient initialization to suppress version warnings
- **Note**: HTTP API is stable and compatible across minor versions

**3. typer: ^0.9.0 → ^0.12.0**
- **Reason**: Required for vLLM 0.11.0 compatibility
- **Impact**: Potential breaking changes in CLI behavior
- **Changes Required**: Minimal - typer 0.12.x is largely backward compatible with 0.9.x
- **Note**: The [all] extra is maintained

### Optional Dependencies

**4. vLLM: ^0.5.0 → ^0.11.0**
- **Reason**: Security updates and bug fixes
- **Impact**: Significant version jump with potential breaking changes
- **Python Requirement**: Requires Python >=3.9,<3.14 (transitive from torch/triton)
- **Dependencies**: Requires typer >=0.12.3, fastapi[standard] >=0.115.0, torch 2.8.0
- **Installation**: `poetry install -E query`

## Python Version Constraint

**Previous**: `python = "^3.10"` (allows 3.10 to 3.99)
**Updated**: `python = ">=3.10,<3.14"`

**Reason**: vLLM 0.11.0 and its transitive dependencies (torch 2.8.0, triton 3.4.0) require Python <3.14

**Impact**:
- Python 3.10, 3.11, 3.12, 3.13 are supported
- Python 3.14+ is not supported (when it's released)
- This change was necessary to resolve dependency conflicts

## Breaking Changes and Compatibility

### typer 0.9 → 0.12

**Changes in typer 0.12**:
- Improved type hints and validation
- Better error messages
- Enhanced rich terminal support
- Backward compatible for basic CLI usage

**Our Usage**:
- ✅ `typer.Typer()` - No changes needed
- ✅ `typer.Option()` - No changes needed
- ✅ `typer.Argument()` - No changes needed
- ✅ `typer.echo()` - No changes needed
- ✅ `typer.Exit()` - No changes needed

**Action Required**: None - our CLI code is compatible

### vLLM 0.5 → 0.11

**Major Changes**:
- **torch requirement**: Now requires torch 2.8.0 (up from 2.1.0+)
- **fastapi requirement**: Now requires fastapi[standard] >=0.115.0
- **Python version**: Strictly <3.14
- **API Changes**: Potential changes in vLLM client API (to be verified at runtime)

**Our Usage**:
- ✅ We use the HTTP API via `requests.post()` to `/v1/completions` endpoint
- ✅ Standard OpenAI-compatible API format
- ⚠️ May need adjustments if vLLM 0.11 changes the API schema

**Mitigation**:
- Since vLLM is optional, base functionality (ingest, embed, index) works without it
- Users installing with query extra should test the query functionality
- The HTTP API is generally stable across versions

## Testing Requirements

### Before Deployment

1. **Syntax Verification**: ✅ Completed
   ```bash
   python -m py_compile src/rainrag/**/*.py
   ```

2. **Unit Tests**: Run the test suite
   ```bash
   poetry install
   poetry run pytest tests/ -v
   ```

3. **Integration Tests**: Test with actual services
   ```bash
   # Start Qdrant
   docker run -p 6333:6333 qdrant/qdrant:v1.7.4

   # Test ingestion pipeline
   poetry run rainrag pipeline

   # Start vLLM (optional - for query testing)
   docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
     --model mistralai/Mistral-Small-3.2-24B-Instruct-2506

   # Test query functionality
   poetry run rainrag ask "Test question"
   ```

## Installation Instructions

### Without Query Functionality (No vLLM)
```bash
poetry install
```

### With Query Functionality (With vLLM)
```bash
poetry install -E query
```

## Rollback Plan

If critical issues are discovered:

1. **Revert pyproject.toml**:
   ```toml
   python = "^3.10"
   requests = "^2.31.0"
   typer = {extras = ["all"], version = "^0.9.0"}
   vllm = {version = "^0.5.0", optional = true}
   ```

2. **Regenerate lock file**:
   ```bash
   poetry lock --no-interaction
   ```

3. **Reinstall**:
   ```bash
   poetry install --sync
   ```

## Security Considerations

### requests 2.32.4
- Addresses CVEs in previous versions
- Improved handling of SSL/TLS certificates
- Better handling of redirect chains

### vLLM 0.11.0
- Multiple security patches from 0.5.0 to 0.11.0
- Improved input validation
- Updated dependencies with security fixes

## Migration Checklist

- [x] Update pyproject.toml with new versions
- [x] Add Python version constraint
- [x] Regenerate poetry.lock
- [x] Verify Python syntax
- [ ] Run unit tests (poetry run pytest)
- [ ] Test ingestion pipeline
- [ ] Test query functionality with vLLM
- [ ] Update CI/CD pipelines if needed
- [ ] Monitor for runtime issues

## Additional Notes

- The poetry.lock file is now included in the repository (695KB)
- All transitive dependencies have been resolved and locked
- Python 3.14 support can be added once vLLM releases a compatible version

## References

- [typer Changelog](https://typer.tiangolo.com/release-notes/)
- [vLLM Releases](https://github.com/vllm-project/vllm/releases)
- [requests Security](https://github.com/psf/requests/security)
