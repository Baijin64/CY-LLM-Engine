# CY-LLM Engine Code Standards & Refactoring Guidelines

**Generated**: 2026-02-02
**Scope**: CY-LLM-Engine (Open Source Version)
**Target Architecture**: Mixed Python/Rust Architecture (Reference: CYLLM架构图.html)

---

## 1. Python Standards (Strict)

All Python code must adhere to the following strict standards to ensure maintainability and performance in the open-source release.

### 1.1 Formatting & Style
*   **Formatter**: **Black** must be used for all code formatting.
    *   Line length: **88 characters**.
*   **Import Sorting**: **isort** (profile: black).
*   **Linter**: **Ruff** (replacing Flake8/Pylint for speed).
    *   Must satisfy `E`, `F`, `B` (bugbear), and `I` (imports) rulesets.

### 1.2 Type Hinting
*   **Requirement**: **Strict**.
*   All function signatures (arguments and return types) **MUST** be typed.
*   **Tool**: `mypy --strict`.
*   **Dynamic Typing**: Avoid `Any` wherever possible. Use `TypeVar`, `Protocol`, or specific types.

```python
# ✅ Correct
def process_data(data: list[str]) -> int:
    return len(data)

# ❌ Incorrect
def process_data(data):
    return len(data)
```

### 1.3 Documentation (Docstrings)
*   **Style**: **Google Style**.
*   **Requirement**: Mandatory for all **exported** (public) modules, classes, and functions.
*   **Content**: Must include `Args`, `Returns`, and `Raises` sections.

```python
def calculate_metrics(values: list[float]) -> dict[str, float]:
    """Calculates statistical metrics for a dataset.

    Args:
        values: A list of numerical values to process.

    Returns:
        A dictionary containing 'mean' and 'std_dev'.

    Raises:
        ValueError: If the input list is empty.
    """
    ...
```

---

## 2. Rust Sidecar Standards (New)

To achieve high-performance inference and memory safety, we are introducing a Rust component.

### 2.1 Directory Structure
The Rust component shall reside in a dedicated directory at the project root:
```text
/
├── rust_core/
│   ├── Cargo.toml      # Workspace definition
│   ├── src/
│   │   ├── lib.rs      # Python bindings entry point
│   │   └── ...
│   └── pyproject.toml  # Maturin configuration
```

### 2.2 Integration
*   **Tool**: **PyO3** with **Maturin** for building strict Python wheels.
*   **Usage**: CPU-intensive tasks (tokenization, tensor operations validation) must be offloaded to `rust_core`.

### 2.3 Rust Coding Style
*   **Formatter**: `rustfmt` (standard).
*   **Linter**: `clippy` (pedantic warnings enabled).

---

## 3. Artifact Cleanup (Enterprise Removal)

The following artifacts are remnants of the Enterprise/Legacy architecture and **MUST be removed** to align with the Open Source vision.

### 3.1 Forbidden Files (Delete immediately)
*   **Build Systems**: `build.gradle`, `settings.gradle`, `settings.gradle.kts`, `gradlew`, `gradlew.bat`.
*   **Languages**: All Kotlin (`*.kt`, `*.kts`) and Java (`*.java`) files.
*   **Binaries**: `*.jar`, `*.class`.
*   **IDE Configs**: `.idea/`, `.vscode/` (unless strictly for shared recommendations).

### 3.2 Rationale
The Open Source version is a pure Python/Rust architecture. Gradle and Kotlin are unnecessary complexity introduced by the legacy enterprise deployment system.

---

## 4. Project Structure (Target State)

```text
CY-LLM-Engine/
├── cy_llm/                 # Main Python Package (renamed from CY_LLM_Backend)
│   ├── __init__.py
│   ├── core/              # Core logic
│   ├── worker/            # Worker nodes
│   └── interfaces/        # Abstract base classes
├── rust_core/              # Rust Sidecar
│   ├── Cargo.toml
│   └── src/
├── tests/                  # Pytest suite
├── docs/                   # Documentation
├── pyproject.toml          # Single source of truth for dependencies
└── Makefile                # Unified build commands (install, test, build-rust)
```

---

## 5. Enforcement Plan

1.  **Immediate**: Run `scripts/clean_enterprise_artifacts.sh` (to be created) to remove Gradle/Kotlin files.
2.  **Migration**: Move `CY_LLM_Backend` code to `cy_llm` and apply `black` + `isort`.
3.  **CI/CD**: Configure GitHub Actions to fail if `mypy` or `black --check` fails.
