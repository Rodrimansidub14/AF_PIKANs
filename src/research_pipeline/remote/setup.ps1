# Installs only repository-local infrastructure; never authenticates provider accounts.
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$stateDir = Join-Path $repoRoot 'var/agent_runs'
$envDir = Join-Path $stateDir 'remote-env'
$tempDir = Join-Path $stateDir 'setup-tmp'
$cacheDir = Join-Path $stateDir 'uv-cache'
New-Item -ItemType Directory -Force $tempDir, $cacheDir | Out-Null
$saved = @{}
foreach ($name in @('TEMP', 'TMP', 'UV_CACHE_DIR', 'UV_PYTHON_DOWNLOADS')) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
try {
    $env:TEMP = $tempDir
    $env:TMP = $tempDir
    $env:UV_CACHE_DIR = $cacheDir
    $env:UV_PYTHON_DOWNLOADS = 'never'
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'uv must be installed before running this setup.'
    }
    $envPython = Join-Path $envDir 'Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $envPython)) {
        $basePython = & py -3.13 -c 'import sys; print(sys.executable)'
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 is required by Colab MCP.' }
        & uv venv $envDir --python $basePython --no-managed-python
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the local environment.' }
    }
    & uv pip sync (Join-Path $PSScriptRoot 'requirements.lock') --python $envPython
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    Push-Location $repoRoot
    try {
        & $envPython -m src.research_pipeline.remote doctor
        if ($LASTEXITCODE -ne 0) { throw 'Local diagnosis failed.' }
    } finally { Pop-Location }
} finally {
    foreach ($name in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
    }
}
