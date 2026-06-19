@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM cascade-router — Windows build + HTTP proxy server
REM Run from: cpp_core\build_and_run.bat

set "ORT_VERSION=1.17.1"
set "ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v%ORT_VERSION%/onnxruntime-win-x64-%ORT_VERSION%.zip"
set "ORT_ROOT=C:\libs\onnxruntime"
set "ORT_ZIP=%TEMP%\onnxruntime-win-x64-%ORT_VERSION%.zip"
set "MODEL_PATH=..\models\all-MiniLM-L6-v2-int8.onnx"
set "MODEL_FP32=..\models\all-MiniLM-L6-v2.onnx"
set "MODEL_FALLBACK=..\models\all-MiniLM-L6-v2-onnx\model.onnx"

echo.
echo ============================================================
echo  cascade-router — build and start HTTP proxy
echo ============================================================
echo.

cd /d "%~dp0"

REM ------------------------------------------------------------------
REM 0. Locate CMake (VS-bundled or PATH)
REM ------------------------------------------------------------------
set "CMAKE_EXE=cmake"
where cmake >nul 2>&1
if errorlevel 1 (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq delims=" %%i in (`"!VSWHERE!" -latest -requires Microsoft.Component.MSBuild -find Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`) do (
            set "CMAKE_EXE=%%i"
        )
    )
)
if /i not "!CMAKE_EXE!"=="cmake" (
    echo [OK] Using CMake: !CMAKE_EXE!
) else (
    where cmake >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] CMake not found. Install Visual Studio 2022 with C++ workload or CMake.
        exit /b 1
    )
)

REM ------------------------------------------------------------------
REM 1. Download / extract ONNX Runtime (skip if already present)
REM ------------------------------------------------------------------
if exist "%ORT_ROOT%\include\onnxruntime_cxx_api.h" (
    echo [OK] ONNX Runtime already present at %ORT_ROOT%
) else (
    echo [1/5] Downloading ONNX Runtime v%ORT_VERSION% ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "Invoke-WebRequest -Uri '%ORT_URL%' -OutFile '%ORT_ZIP%' -UseBasicParsing"
    if errorlevel 1 (
        echo [ERROR] Failed to download ONNX Runtime.
        exit /b 1
    )

    echo [2/5] Extracting to %ORT_ROOT% ...
    if not exist "C:\libs" mkdir "C:\libs"
    if exist "%ORT_ROOT%" rmdir /s /q "%ORT_ROOT%"

    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "Expand-Archive -Path '%ORT_ZIP%' -DestinationPath 'C:\libs' -Force"
    if errorlevel 1 (
        echo [ERROR] Failed to extract ONNX Runtime archive.
        exit /b 1
    )

    if exist "C:\libs\onnxruntime-win-x64-%ORT_VERSION%" (
        move /y "C:\libs\onnxruntime-win-x64-%ORT_VERSION%" "%ORT_ROOT%" >nul
    )

    if not exist "%ORT_ROOT%\include\onnxruntime_cxx_api.h" (
        echo [ERROR] ONNX Runtime headers not found after extract.
        exit /b 1
    )
    echo [OK] ONNX Runtime installed at %ORT_ROOT%
)

REM ------------------------------------------------------------------
REM 2. Verify ONNX model exists
REM ------------------------------------------------------------------
if not exist "%MODEL_FP32%" (
    if exist "%MODEL_FALLBACK%" (
        echo [INFO] Copying %MODEL_FALLBACK% to %MODEL_FP32%
        copy /y "%MODEL_FALLBACK%" "%MODEL_FP32%" >nul
    )
)
if not exist "%MODEL_PATH%" (
    echo [ERROR] Quantized model not found: %MODEL_PATH%
    echo         Run: .venv\Scripts\python.exe -m src.quantize_onnx
    exit /b 1
)
echo [OK] Model found: %MODEL_PATH%

if not exist "..\models\vocab.txt" (
    echo [INFO] vocab.txt missing — downloading via Python ...
    pushd ..
    .venv\Scripts\python.exe -m src.download_vocab
    if errorlevel 1 (
        echo [ERROR] Failed to download vocab.txt
        popd
        exit /b 1
    )
    popd
)
if not exist "..\models\vocab.txt" (
    echo [ERROR] vocab.txt not found at ..\models\vocab.txt
    exit /b 1
)
echo [OK] Vocab found: ..\models\vocab.txt

if not exist "..\models\router_weights.json" (
    echo [INFO] router_weights.json missing — exporting via Python ...
    pushd ..
    .venv\Scripts\python.exe -m src.export_weights
    if errorlevel 1 (
        echo [ERROR] Failed to export router weights
        popd
        exit /b 1
    )
    popd
)
if not exist "..\models\router_weights.json" (
    echo [ERROR] router_weights.json not found at ..\models\router_weights.json
    exit /b 1
)
echo [OK] Router weights found: ..\models\router_weights.json

REM ------------------------------------------------------------------
REM 3. CMake configure
REM ------------------------------------------------------------------
echo.
echo [3/5] Configuring CMake (httplib + simdjson via FetchContent) ...
"!CMAKE_EXE!" -S . -B build -DONNXRUNTIME_ROOT=%ORT_ROOT% -G "Visual Studio 17 2022" -A x64
if errorlevel 1 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

REM ------------------------------------------------------------------
REM 4. Build Release
REM ------------------------------------------------------------------
echo.
echo [4/5] Building Release (benchmark_onnx + proxy_server) ...
"!CMAKE_EXE!" --build build --config Release
if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b 1
)

REM ------------------------------------------------------------------
REM 5. Copy DLL + start proxy server
REM ------------------------------------------------------------------
if not exist "build\Release" mkdir "build\Release"
if exist "%ORT_ROOT%\lib\onnxruntime.dll" (
    copy /y "%ORT_ROOT%\lib\onnxruntime.dll" "build\Release\" >nul
    echo [OK] Copied onnxruntime.dll to build\Release\
) else (
    echo [WARN] onnxruntime.dll not found in %ORT_ROOT%\lib
)

echo.
echo [5/5] Starting HTTP proxy on http://localhost:8000
echo       Press Ctrl+C to stop.
echo ------------------------------------------------------------
build\Release\proxy_server.exe "..\models\all-MiniLM-L6-v2-int8.onnx"
