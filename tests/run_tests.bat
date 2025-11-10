@echo off
REM run_tests.bat - Windows test runner for AI Interviewer Bot

echo ================================================
echo AI Interviewer Bot - Test Suite Runner (Windows)
echo ================================================
echo.

REM Check if pytest is installed
python -m pytest --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pytest not found!
    echo Please install with: pip install -r requirements-test-minimal.txt
    pause
    exit /b 1
)

REM Parse command line arguments
set TEST_TYPE=all
set COVERAGE=true
set MARKERS=

:parse_args
if "%1"=="" goto run_tests
if "%1"=="--unit" (
    set TEST_TYPE=unit
    set MARKERS=-m unit
    shift
    goto parse_args
)
if "%1"=="--fast" (
    set MARKERS=-m "not slow"
    shift
    goto parse_args
)
if "%1"=="--no-cov" (
    set COVERAGE=false
    shift
    goto parse_args
)
shift
goto parse_args

:run_tests
echo Test Type: %TEST_TYPE%
echo Coverage: %COVERAGE%
echo.

REM Build pytest command
set PYTEST_CMD=python -m pytest tests/

if not "%MARKERS%"=="" (
    set PYTEST_CMD=%PYTEST_CMD% %MARKERS%
)

if "%COVERAGE%"=="true" (
    set PYTEST_CMD=%PYTEST_CMD% --cov=app --cov-report=html --cov-report=term-missing
)

echo Running: %PYTEST_CMD%
echo ================================================
echo.

REM Run tests
%PYTEST_CMD%

if errorlevel 1 (
    echo.
    echo ================================================
    echo [FAILED] Tests failed!
    echo ================================================
    pause
    exit /b 1
) else (
    echo.
    echo ================================================
    echo [SUCCESS] All tests passed!
    if "%COVERAGE%"=="true" (
        echo.
        echo Coverage report: htmlcov\index.html
        echo Open with: start htmlcov\index.html
    )
    echo ================================================
    pause
    exit /b 0
)