$CurrentDir = Get-Location
$BuildDir = Join-Path $CurrentDir.Path "build"
if (!(Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir
}

$ShortcutPath = Join-Path $BuildDir "whisper.lnk"
$WScriptShell = New-Object -ComObject WScript.Shell

$PythonExe = Join-Path $CurrentDir "venv\Scripts\pythonw.exe"
$ScriptPath = Join-Path $CurrentDir "dictate.py"

$Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $PythonExe
$Shortcut.Arguments = "`"$ScriptPath`""
$Shortcut.WorkingDirectory = $CurrentDir.Path
$Shortcut.IconLocation = "$CurrentDir\icon.ico"
$Shortcut.Description = "whisper Voice-to-Text Dictation"
$Shortcut.Save()

Write-Host "Shortcut created in build folder: $ShortcutPath"
