[CmdletBinding()]
param(
    [switch]$PlanOnly,
    [switch]$ValidateOnly,
    [switch]$Execute
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$SelectedModeCount = [int]$PlanOnly.IsPresent + [int]$ValidateOnly.IsPresent + [int]$Execute.IsPresent
if ($SelectedModeCount -gt 1) {
    throw 'Choose only one of -PlanOnly, -ValidateOnly, or -Execute.'
}
if ($SelectedModeCount -eq 0) {
    $PlanOnly = $true
}

$Orchestrator = Join-Path $PSScriptRoot 'finals_part3_orchestrator.py'
if (-not (Test-Path -LiteralPath $Orchestrator -PathType Leaf)) {
    throw "Finals Part 3 orchestrator is missing: $Orchestrator"
}

$BundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$PythonCandidates = @(
    [pscustomobject]@{ Command = $BundledPython; Prefix = @() },
    [pscustomobject]@{ Command = 'py.exe'; Prefix = @('-3') },
    [pscustomobject]@{ Command = 'python.exe'; Prefix = @() }
)

$PythonCommand = $null
$PythonPrefix = @()
foreach ($Candidate in $PythonCandidates) {
    $CommandPath = $Candidate.Command
    if ([System.IO.Path]::IsPathRooted($CommandPath) -and -not (Test-Path -LiteralPath $CommandPath -PathType Leaf)) {
        continue
    }
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'SilentlyContinue'
        & $CommandPath @($Candidate.Prefix) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) {
            $PythonCommand = $CommandPath
            $PythonPrefix = @($Candidate.Prefix)
            break
        }
    }
    catch {
        continue
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
}
if ($null -eq $PythonCommand) {
    throw 'A working Python 3.10+ runtime was not found.'
}

$ModeArgument = if ($Execute) {
    '--execute'
}
elseif ($ValidateOnly) {
    '--validate-only'
}
else {
    '--plan-only'
}

& $PythonCommand @PythonPrefix $Orchestrator $ModeArgument
exit $LASTEXITCODE
