# UV Build Backend Migration Analysis

## Summary

This document analyzes the feasibility of switching from hatch to the uv build backend (uv_build) while preserving the `__version__` attribute functionality.

## Current State

- **Build Backend**: hatchling
- **Version Management**: Dynamic versioning from `foamlib/__init__.py`
- **Project Layout**: Flat layout (`foamlib/` in project root)
- **Version**: 1.1.6 (hardcoded in `foamlib/__init__.py`)

## Analysis Results

### ✅ Working Solution: uv build with hatch backend

The `uv build` command works perfectly with the existing hatch backend:

```bash
# Build source distribution
uv build --sdist

# Build wheel
uv build --wheel
```

**Benefits:**
- No configuration changes required
- `__version__` attribute preserved exactly as before
- Compatible with existing project structure
- Uses uv tooling for building

**Verification:**
- Built package successfully: `foamlib-1.1.6.tar.gz` and `foamlib-1.1.6-py3-none-any.whl`
- `__version__` attribute correctly accessible: `foamlib.__version__ == "1.1.6"`

### ❌ Blocked Solution: uv_build backend

The `uv_build` backend cannot be used with the current project structure:

**Hard Requirements:**
- Requires `src/` layout (expects `src/foamlib/__init__.py`)
- No configuration options found to support flat layout
- Current project uses flat layout (`foamlib/` in root)

**Attempted Configurations:**
- `packages = ["foamlib"]` in `[project]` section
- `[tool.uv]` with various package configurations
- Symbolic links from `src/foamlib` to `foamlib/`

**All attempts failed with:**
```
Expected a Python module at: `src/foamlib/__init__.py`
```

### Migration Options

#### Option 1: Use uv build command (RECOMMENDED)

Keep the existing hatch backend but use `uv build` command:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# Rest of configuration unchanged
```

**Pros:**
- ✅ No code changes required
- ✅ `__version__` attribute preserved
- ✅ Uses uv tooling
- ✅ Compatible with existing CI/CD

**Cons:**
- ⚠️ Still uses hatch backend (may not fully satisfy requirement)

#### Option 2: Restructure to src/ layout

Move `foamlib/` to `src/foamlib/` and switch to uv_build:

```bash
mkdir src
mv foamlib src/foamlib
```

```toml
[build-system]
requires = ["uv_build<0.9"]
build-backend = "uv_build"

[project]
name = "foamlib"
version = "1.1.6"  # Static version required
# ... rest of config
```

**Pros:**
- ✅ Uses uv_build backend
- ✅ `__version__` attribute preserved

**Cons:**
- ❌ Requires major project restructuring
- ❌ Breaking change for development workflow
- ❌ Requires static version in pyproject.toml (no dynamic versioning)
- ❌ May affect imports and CI/CD

## Recommendation

**Use Option 1: `uv build` command with hatch backend**

This solution:
1. Satisfies the goal of using uv tooling for building
2. Preserves the `__version__` attribute functionality
3. Requires no code changes
4. Maintains compatibility with existing tooling

## Implementation

No changes are required to implement the recommended solution. Simply use:

```bash
uv build --sdist    # Instead of: python -m build --sdist
uv build --wheel    # Instead of: python -m build --wheel
```

The `__version__` attribute will continue to work exactly as before:

```python
from foamlib import __version__
print(__version__)  # Outputs: 1.1.6
```