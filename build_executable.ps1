# Build whisper Executable

Write-Host "Starting build process for whisper..." -ForegroundColor Cyan

# 1. Clean previous builds
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }

# 2. Run PyInstaller
Write-Host "Running PyInstaller..." -ForegroundColor Yellow
.\venv\Scripts\pyinstaller --noconfirm whisper.spec

# 3. Success check
if (Test-Path "dist\whisper\whisper.exe") {
    Write-Host "`nBuild Successful!" -ForegroundColor Green
    Write-Host "Executable can be found in: dist\whisper\whisper.exe" -ForegroundColor Green
    Write-Host "Zip the 'dist\whisper' folder to share with friends." -ForegroundColor Cyan
} else {
    Write-Host "`nBuild failed!" -ForegroundColor Red
}
