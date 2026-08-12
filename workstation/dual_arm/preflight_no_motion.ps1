param(
  [switch]$ExecuteReadOnly,
  [switch]$OperatorAuthorizedReadOnlyProbe,
  [string]$EvidenceOut = ""
)

$ErrorActionPreference = "Stop"
$ConfigPath = Join-Path $PSScriptRoot "station_config.json"
$Config = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$ArmPairKnownHosts = Join-Path $PSScriptRoot "evidence/arm_pair_10.185.110_known_hosts"

$RemotePython = @'
import glob
import importlib.util
import json
import os
import pathlib
import subprocess

def text(path):
    try:
        return pathlib.Path(path).read_text(errors="replace").replace("\x00", "").strip()
    except Exception:
        return None

def run(args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=3)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as exc:
        return {"exit_code": 127, "stdout": "", "stderr": str(exc)}

videos = []
for device in sorted(glob.glob("/dev/video*")):
    name = pathlib.Path(device).name
    videos.append({"device": device, "name": text(f"/sys/class/video4linux/{name}/name")})

services = {}
for service in ("xrd-workcockpit.service", "xrd-workcockpit-arm02.service", "xrd-overhead-camera.service"):
    services[service] = run(["systemctl", "is-active", service])["stdout"] or "unknown"

serial_owner = run(["fuser", "/dev/ttyAMA0"])
payload = {
    "hostname": run(["hostname"])["stdout"],
    "addresses": run(["hostname", "-I"])["stdout"].split(),
    "model": text("/proc/device-tree/model"),
    "cpu_serial": next((line.split(":", 1)[1].strip() for line in text("/proc/cpuinfo").splitlines() if line.startswith("Serial")), None),
    "wlan0_mac": text("/sys/class/net/wlan0/address"),
    "robot_serial_exists": pathlib.Path("/dev/ttyAMA0").exists(),
    "robot_serial_owner_pids": serial_owner["stdout"].split(),
    "video_devices": videos,
    "services": services,
    "modules": {name: bool(importlib.util.find_spec(name)) for name in ("pymycobot", "cv2", "flask")},
    "overhead_camera_script_present": pathlib.Path("/home/rdk/dual_arm/overhead_camera_service.py").is_file(),
    "overhead_camera_unit_present": pathlib.Path("/etc/systemd/system/xrd-overhead-camera.service").is_file(),
    "safety": {
        "serial_opened_by_probe": False,
        "camera_opened_by_probe": False,
        "robot_sdk_imported_by_probe": False,
        "motion_command_sent": False
    }
}
print(json.dumps(payload, separators=(",", ":")))
'@

function Write-Evidence {
  param([object]$Value)
  $json = $Value | ConvertTo-Json -Depth 12
  if (-not [string]::IsNullOrWhiteSpace($EvidenceOut)) {
    $parent = Split-Path -Parent $EvidenceOut
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
      New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    Set-Content -LiteralPath $EvidenceOut -Value $json -Encoding UTF8
  }
  Write-Output $json
}

function Get-TargetSpec {
  param([string]$ArmId)
  if ($ArmId -eq "arm01") {
    return [pscustomobject]@{
      id = "arm01"
      target = "er@198.51.100.136"
      args = @(
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=12",
        "-o", "ConnectionAttempts=2",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "UpdateHostKeys=no",
        "-o", "HostKeyAlias=198.51.100.136",
        "-o", "UserKnownHostsFile=$ArmPairKnownHosts"
      )
    }
  }
  return [pscustomobject]@{
    id = "arm02"
    target = "er@198.51.100.145"
    args = @(
      "-o", "BatchMode=yes",
      "-o", "ConnectTimeout=12",
      "-o", "ConnectionAttempts=2",
      "-o", "StrictHostKeyChecking=yes",
      "-o", "UpdateHostKeys=no",
      "-o", "HostKeyAlias=198.51.100.145",
      "-o", "UserKnownHostsFile=$ArmPairKnownHosts"
    )
  }
}

$CanExecute = [bool]($ExecuteReadOnly -and $OperatorAuthorizedReadOnlyProbe)
$Evidence = [ordered]@{
  schema_version = "xrd-dual-arm-no-motion-preflight-v1"
  created_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
  mode = if ($CanExecute) { "execute_read_only" } else { "dry_run" }
  config = $ConfigPath
  state_changed = $false
  pc_network_changed = $false
  serial_opened = $false
  camera_opened = $false
  robot_sdk_imported = $false
  motion_command_sent = $false
  targets = @()
  identity_ok = $false
  environment_ready_for_camera_setup = $false
  motion_ready = $false
}

if (-not $CanExecute) {
  if ($ExecuteReadOnly -xor $OperatorAuthorizedReadOnlyProbe) {
    $Evidence.status = "blocked_incomplete_execute_gate"
    Write-Evidence $Evidence
    exit 2
  }
  $Evidence.status = "planned_only"
  foreach ($armId in @("arm01", "arm02")) {
    $spec = Get-TargetSpec $armId
    $Evidence.targets += [ordered]@{
      arm_id = $armId
      target = $spec.target
      action = "read sysfs, procfs, service state, dependency presence, and serial owner only"
    }
  }
  Write-Evidence $Evidence
  exit 0
}

$allIdentityOk = $true
$allDependenciesOk = $true
$results = @()

foreach ($armId in @("arm01", "arm02")) {
  $spec = Get-TargetSpec $armId
  $savedErrorPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $raw = @($RemotePython | & ssh @($spec.args) $spec.target "python3 -" 2>&1)
  $exitCode = $LASTEXITCODE
  $ErrorActionPreference = $savedErrorPreference
  $joined = ($raw | ForEach-Object { [string]$_ }) -join "`n"
  $probe = $null
  if ($exitCode -eq 0) {
    try { $probe = $joined | ConvertFrom-Json } catch { $exitCode = 3 }
  }
  $expected = $Config.arms.$armId
  $identityReasons = @()
  if ($null -eq $probe) {
    $identityReasons += "probe_unavailable"
  } else {
    if ($probe.hostname -ne $expected.hostname) { $identityReasons += "hostname_mismatch" }
    if ($probe.cpu_serial -ne $expected.cpu_serial) { $identityReasons += "cpu_serial_mismatch" }
    if ($probe.wlan0_mac -ne $expected.wlan0_mac) { $identityReasons += "wlan0_mac_mismatch" }
    if ($probe.addresses -notcontains $expected.overlay_entry) { $identityReasons += "overlay_address_missing" }
  }
  $identityOk = $exitCode -eq 0 -and $identityReasons.Count -eq 0
  $dependenciesOk = $null -ne $probe -and $probe.modules.pymycobot -and $probe.modules.cv2
  $allIdentityOk = $allIdentityOk -and $identityOk
  $allDependenciesOk = $allDependenciesOk -and $dependenciesOk
  $results += [ordered]@{
    arm_id = $armId
    target = $spec.target
    ssh_exit_code = $exitCode
    identity_ok = $identityOk
    identity_reasons = $identityReasons
    dependencies_ok = $dependenciesOk
    probe = $probe
    transport_error = if ($exitCode -eq 0) { $null } else { $joined }
  }
}

$Evidence.targets = $results
$Evidence.identity_ok = $allIdentityOk
$Evidence.environment_ready_for_camera_setup = $allIdentityOk -and $allDependenciesOk
$Evidence.status = if ($Evidence.environment_ready_for_camera_setup) { "pass" } else { "failed_or_unavailable" }
$Evidence.completed_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
Write-Evidence $Evidence
exit $(if ($Evidence.environment_ready_for_camera_setup) { 0 } else { 2 })
