@echo off
REM AI_Biomed Windows Orchestrator

IF "%1"=="" GOTO help
IF "%1"=="test" GOTO test
IF "%1"=="benchmark" GOTO benchmark
IF "%1"=="nephro" GOTO nephro
IF "%1"=="figures" GOTO figures
IF "%1"=="app" GOTO app
IF "%1"=="all" GOTO all

:help
echo AI_Biomed Research Framework
echo Usage: run.bat [test^|benchmark^|nephro^|figures^|app^|all]
GOTO end

:test
python -m pytest tests/ -v
GOTO end

:benchmark
python main.py --mode benchmark
GOTO end

:nephro
python main.py --mode nephro
GOTO end

:figures
python main.py --mode figures
GOTO end

:app
streamlit run app.py
GOTO end

:all
python main.py --mode all
GOTO end

:end

