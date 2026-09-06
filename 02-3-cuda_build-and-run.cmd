@echo off
setlocal
cd /d "%~dp0"
where nvcc >nul 2>nul
if errorlevel 1 (
  echo nvcc was not found in PATH. Open the NVIDIA CUDA Command Prompt, or add CUDA\v13.3\bin to PATH.
  pause
  exit /b 1
)
nvcc cuda_ollama_demo.cu -o cuda_ollama_demo.exe -Xcompiler /EHsc -diag-suppress=68
if errorlevel 1 (
  echo Build failed. Install Visual Studio Build Tools with the Desktop development with C++ workload.
  pause
  exit /b 1
)
cuda_ollama_demo.exe "Explain CUDA and local LLM inference in one short sentence."
pause
