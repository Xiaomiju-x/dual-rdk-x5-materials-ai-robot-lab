[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,

    [Parameter(Mandatory = $true)]
    [string]$DestinationRoot,

    [switch]$Apply
)

$ErrorActionPreference = "Stop"

$source = (Resolve-Path -LiteralPath $SourceRoot).Path
$destination = (Resolve-Path -LiteralPath $DestinationRoot).Path

if (-not (Test-Path -LiteralPath (Join-Path $destination ".git"))) {
    throw "DestinationRoot must be an isolated Git working tree."
}

$allowedExtensions = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
@(
    ".c", ".cc", ".cpp", ".h", ".hpp",
    ".py", ".ps1", ".sh",
    ".js", ".mjs", ".ts", ".tsx", ".vue", ".css", ".html",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".md", ".rst", ".txt", ".csv", ".sha256",
    ".xml", ".xacro", ".urdf", ".launch", ".rviz", ".service",
    ".uvprojx", ".uvoptx", ".ld", ".cmake", ".msg", ".srv", ".action",
    ".npz"
) | ForEach-Object { [void]$allowedExtensions.Add($_) }

$specialNames = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
@("CMakeLists.txt", "package.xml", "requirements.txt", "Makefile") |
    ForEach-Object { [void]$specialNames.Add($_) }

$excludedPathPattern = [regex]::new(
    "(^|[\\/])(" +
    "\.git|\.venv[^\\/]*|node_modules|__pycache__|\.pytest_cache|\.ruff_cache|" +
    "backups?|archive|artifacts?|models?|releases?|handoffs?|outputs?|runs?|tmp|work|" +
    "dataset|datasets|data|evidence|papers?|embeddings?|raw|processed|staging|" +
    "xrd_knowledge|crystal_data|bpu|bpu_export|" +
    "Objects|Listings|DebugConfig|Device|RTE|firmware_images|" +
    "system_recovery|offline_wheels|vendor|toolchain_cache" +
    ")([\\/]|$)",
    [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
)

$rules = @(
    @{ Source = "predict_engine"; Destination = "ai_brain/predict_engine" },
    @{ Source = "xrd_numerical"; Destination = "ai_brain/xrd_numerical" },
    @{ Source = "xrd_vision"; Destination = "ai_brain/xrd_vision" },
    @{ Source = "spectrum_numerical"; Destination = "ai_brain/pl_numerical" },
    @{ Source = "spectrum_vision"; Destination = "ai_brain/pl_vision" },
    @{ Source = "icmat_foundry"; Destination = "ai_brain/icmat_foundry" },
    @{ Source = "rb_voe"; Destination = "safety/rb_voe" },
    @{ Source = "rb_voe_passive"; Destination = "safety/rb_voe_passive" },
    @{ Source = "embodied_brain/finals_successor"; Destination = "embodied_brain/finals_successor" },
    @{ Source = "embodied_brain/finals_vnext"; Destination = "embodied_brain/finals_vnext" },
    @{ Source = "embodied_brain/finals_cortex"; Destination = "embodied_brain/finals_cortex" },
    @{ Source = "embodied_brain/ros2_ws/src"; Destination = "embodied_brain/ros2_ws/src" },
    @{ Source = "embodied_brain/tools"; Destination = "embodied_brain/tools" },
    @{ Source = "embodied_brain/systemd"; Destination = "embodied_brain/systemd" },
    @{ Source = "embodied_brain/stm32_f407/App"; Destination = "firmware/stm32f407/App" },
    @{ Source = "workstation/dual_arm"; Destination = "workstation/dual_arm" },
    @{ Source = "workstation/dual_arm_successor"; Destination = "workstation/dual_arm_successor" },
    @{ Source = "vps/cmdcenter/cmdcenter"; Destination = "web/command_center/cmdcenter" },
    @{ Source = "vps/cmdcenter/static"; Destination = "web/command_center/static" },
    @{ Source = "vps/cmdcenter/tools"; Destination = "web/command_center/tools" },
    @{ Source = "vps/cmdcenter/systemd"; Destination = "web/command_center/systemd" }
)

$explicitFiles = @(
    @{ Source = "dashboard.py"; Destination = "ai_brain/dashboard/dashboard.py" },
    @{ Source = "voice_backend.py"; Destination = "ai_brain/dashboard/voice_backend.py" },
    @{ Source = "shared_locks.py"; Destination = "ai_brain/dashboard/shared_locks.py" },
    @{ Source = "requirements.txt"; Destination = "requirements/runtime.txt" },
    @{ Source = "requirements-dev.txt"; Destination = "requirements/development.txt" },
    @{ Source = "embodied_brain/README.md"; Destination = "embodied_brain/README.md" },
    @{ Source = "embodied_brain/stm32_f407/LIFT_STAGE_README.md"; Destination = "firmware/stm32f407/README_original.md" },
    @{ Source = "embodied_brain/stm32_f407/a.uvprojx"; Destination = "firmware/stm32f407/xrd_f407.uvprojx" },
    @{ Source = "embodied_brain/stm32_f407/a.uvoptx"; Destination = "firmware/stm32f407/xrd_f407.uvoptx" },
    @{ Source = "vps/cmdcenter/app.py"; Destination = "web/command_center/app.py" },
    @{ Source = "vps/cmdcenter/requirements-production.txt"; Destination = "web/command_center/requirements.txt" },
    @{ Source = "vps/cmdcenter/FINALS_PART4_HANDOFF_20260720.md"; Destination = "web/command_center/FINALS_HANDOFF.md" },
    @{ Source = "icmat_foundry/finals_50model/evidence/final_acceptance/final_acceptance.v1.json"; Destination = "evidence/ai_brain/final_acceptance.v1.json" },
    @{ Source = "icmat_foundry/finals_50model/evidence/final_acceptance/final_acceptance.v1.json.sha256"; Destination = "evidence/ai_brain/final_acceptance.v1.json.sha256" },
    @{ Source = "icmat_foundry/finals_50model/evidence/x5_board_20260804/final_acceptance_v1/X5_BOARD_PHASE_FINAL_RECEIPT_20260804.md"; Destination = "evidence/ai_brain/X5_BOARD_PHASE_FINAL_RECEIPT_20260804.md" },
    @{ Source = "icmat_foundry/finals_50model/evidence/x5_board_20260804/final_acceptance_v1/x5_board_phase_acceptance.v1.json"; Destination = "evidence/ai_brain/x5_board_phase_acceptance.v1.json" },
    @{ Source = "embodied_brain/finals_successor/docs/X5_BOARD_ACCEPTANCE_20260804.md"; Destination = "evidence/embodied_brain/X5_BOARD_ACCEPTANCE_20260804.md" },
    @{ Source = "workstation/dual_arm/FINALS_PART3_HANDOFF_20260720.md"; Destination = "evidence/workstation/FINALS_PART3_HANDOFF_20260720.md" },
    @{ Source = "workstation/dual_arm_successor/docs/X5_BOARD_DEPLOYMENT_HANDOFF_20260804.md"; Destination = "evidence/workstation/X5_BOARD_DEPLOYMENT_HANDOFF_20260804.md" },
    @{ Source = "docs/finals_board_closeout_20260804/DUAL_X5_CANDIDATE_BOARD_CLOSEOUT_20260804.md"; Destination = "evidence/system/DUAL_X5_CANDIDATE_BOARD_CLOSEOUT_20260804.md" }
)

$copyPlan = [System.Collections.Generic.List[object]]::new()

function Get-PublicationFiles {
    param([Parameter(Mandatory = $true)][string]$Root)

    $pending = [System.Collections.Generic.Stack[System.IO.DirectoryInfo]]::new()
    $pending.Push((Get-Item -LiteralPath $Root))

    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        foreach ($entry in Get-ChildItem -LiteralPath $directory.FullName -Force -ErrorAction SilentlyContinue) {
            $relativeEntry = $entry.FullName.Substring($Root.Length).TrimStart([char]92, [char]47)
            if ($entry.PSIsContainer) {
                if (-not $excludedPathPattern.IsMatch($relativeEntry)) {
                    $pending.Push($entry)
                }
                continue
            }
            $entry
        }
    }
}

foreach ($rule in $rules) {
    $ruleSource = Join-Path $source $rule.Source
    if (-not (Test-Path -LiteralPath $ruleSource)) {
        Write-Warning "Source path does not exist: $($rule.Source)"
        continue
    }

    Get-PublicationFiles -Root $ruleSource | ForEach-Object {
        $relative = $_.FullName.Substring($ruleSource.Length).TrimStart([char]92, [char]47)
        if ($excludedPathPattern.IsMatch($relative)) {
            return
        }

        if (-not ($allowedExtensions.Contains($_.Extension) -or $specialNames.Contains($_.Name))) {
            return
        }

        $copyPlan.Add([pscustomobject]@{
            Source = $_.FullName
            Destination = Join-Path $destination (Join-Path $rule.Destination $relative)
        })
    }
}

foreach ($file in $explicitFiles) {
    $fileSource = Join-Path $source $file.Source
    if (-not (Test-Path -LiteralPath $fileSource)) {
        Write-Warning "Source file does not exist: $($file.Source)"
        continue
    }
    $copyPlan.Add([pscustomobject]@{
        Source = $fileSource
        Destination = Join-Path $destination $file.Destination
    })
}

$copyPlan = $copyPlan | Sort-Object Destination -Unique
$totalBytes = ($copyPlan | ForEach-Object { (Get-Item -LiteralPath $_.Source).Length } |
    Measure-Object -Sum).Sum

Write-Host ("Source snapshot plan: {0} files, {1:N2} MiB" -f $copyPlan.Count, ($totalBytes / 1MB))

if (-not $Apply) {
    Write-Host "Dry run only. Re-run with -Apply to copy the reviewed allowlist."
    return
}

foreach ($item in $copyPlan) {
    $parent = Split-Path -Parent $item.Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Copy-Item -LiteralPath $item.Source -Destination $item.Destination -Force
}

Write-Host "Source snapshot copied. Run the publication audit before staging any file."
