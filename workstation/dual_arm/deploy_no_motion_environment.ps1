param(
  [switch]$Execute,
  [switch]$OperatorAuthorizedEdgeConfig,
  [string]$EvidenceOut = ""
)

$ErrorActionPreference = "Stop"
$Arm02Target = "er@198.51.100.145"
$ArmPairKnownHosts = Join-Path $PSScriptRoot "evidence/arm_pair_10.185.110_known_hosts"
$Arm02Ssh = @(
  "-o", "BatchMode=yes",
  "-o", "ConnectTimeout=12",
  "-o", "StrictHostKeyChecking=yes",
  "-o", "UpdateHostKeys=no",
  "-o", "HostKeyAlias=198.51.100.145",
  "-o", "UserKnownHostsFile=$ArmPairKnownHosts"
)

$Files = [ordered]@{
  arm02_camera = Join-Path $PSScriptRoot "overhead_camera_service.py"
  arm02_unit = Join-Path $PSScriptRoot "xrd-overhead-camera.service"
  station_config = Join-Path $PSScriptRoot "station_config.json"
}

function File-Sha256([string]$Path) {
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-Evidence([object]$Value) {
  $json = $Value | ConvertTo-Json -Depth 10
  if ($EvidenceOut) {
    $parent = Split-Path -Parent $EvidenceOut
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
      New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    Set-Content -LiteralPath $EvidenceOut -Value $json -Encoding UTF8
  }
  Write-Output $json
}

$CanExecute = [bool]($Execute -and $OperatorAuthorizedEdgeConfig)
$Evidence = [ordered]@{
  schema_version = "xrd-dual-arm-no-motion-deploy-v1"
  created_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
  mode = if ($CanExecute) { "execute" } else { "dry_run" }
  scope = "camera-only service on arm02; X5 is not required or contacted"
  pc_network_changed = $false
  robot_serial_opened = $false
  camera_opened = $false
  service_started = $false
  service_enabled = $false
  motion_command_sent = $false
  files = @{}
  planned_steps = @(
    "back up arm02 crontab and remove only the automatic-ager reboot entry",
    "terminate the automatic-ager process that owns /dev/ttyAMA0",
    "create /home/rdk/dual_arm",
    "backup same-name edge files if present",
    "upload camera-only service and station config to arm02",
    "install xrd-overhead-camera.service but leave it disabled and stopped",
    "verify remote file SHA-256"
  )
}
foreach ($entry in $Files.GetEnumerator()) {
  $Evidence.files[$entry.Key] = [ordered]@{path = $entry.Value; sha256 = File-Sha256 $entry.Value}
}

if (-not $CanExecute) {
  if ($Execute -xor $OperatorAuthorizedEdgeConfig) {
    $Evidence.status = "blocked_incomplete_execute_gate"
    Write-Evidence $Evidence
    exit 2
  }
  $Evidence.status = "planned_only"
  Write-Evidence $Evidence
  exit 0
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$saved = $ErrorActionPreference
$ErrorActionPreference = "Continue"

& ssh @Arm02Ssh $Arm02Target "mkdir -p /home/rdk/dual_arm"
if ($LASTEXITCODE -ne 0) { throw "arm02 SSH preparation failed" }
& scp @Arm02Ssh $Files.arm02_camera "${Arm02Target}:/tmp/overhead_camera_service.py"
if ($LASTEXITCODE -ne 0) { throw "arm02 camera upload failed" }
& scp @Arm02Ssh $Files.arm02_unit "${Arm02Target}:/tmp/xrd-overhead-camera.service"
if ($LASTEXITCODE -ne 0) { throw "arm02 unit upload failed" }
& scp @Arm02Ssh $Files.station_config "${Arm02Target}:/tmp/station_config.json"
if ($LASTEXITCODE -ne 0) { throw "arm02 config upload failed" }

$arm02Install = @"
set -eu
mkdir -p /home/rdk/dual_arm/backups
crontab -l > /home/rdk/dual_arm/backups/crontab-before-dual-arm-$stamp.txt 2>/dev/null || true
crontab -l 2>/dev/null | grep -v '/home/rdk/automatic-ager/runner.sh' > /tmp/xrd-dual-arm-crontab || true
crontab /tmp/xrd-dual-arm-crontab
serial_owners="`$(fuser /dev/ttyAMA0 2>/dev/null || true)"
for pid in `$serial_owners; do
  cmdline="`$(tr '\000' ' ' < "/proc/`$pid/cmdline" 2>/dev/null || true)"
  case "`$cmdline" in
    *'/home/rdk/automatic-ager/aging.py'*) kill -TERM "`$pid" ;;
    *) echo "refusing to stop unexpected arm02 serial owner pid=`$pid cmd=`$cmdline" >&2; exit 3 ;;
  esac
done
sleep 1
if fuser /dev/ttyAMA0 >/tmp/xrd-serial-owner 2>/dev/null; then
  echo "arm02 serial still owned after automatic-ager stop: `$(cat /tmp/xrd-serial-owner)" >&2
  exit 4
fi
for f in /home/rdk/dual_arm/overhead_camera_service.py /home/rdk/dual_arm/station_config.json /etc/systemd/system/xrd-overhead-camera.service; do
  if [ -e "`$f" ]; then sudo -n cp -a "`$f" "`$f.xrd-backup-$stamp"; fi
done
install -m 0755 /tmp/overhead_camera_service.py /home/rdk/dual_arm/overhead_camera_service.py
install -m 0644 /tmp/station_config.json /home/rdk/dual_arm/station_config.json
sudo -n install -m 0644 /tmp/xrd-overhead-camera.service /etc/systemd/system/xrd-overhead-camera.service
sudo -n systemctl daemon-reload
sudo -n systemctl disable xrd-overhead-camera.service >/dev/null 2>&1 || true
sudo -n systemctl stop xrd-overhead-camera.service >/dev/null 2>&1 || true
test "`$(systemctl is-active xrd-overhead-camera.service 2>/dev/null || true)" != active
test "`$(systemctl is-enabled xrd-overhead-camera.service 2>/dev/null || true)" != enabled
sha256sum /home/rdk/dual_arm/overhead_camera_service.py /home/rdk/dual_arm/station_config.json /etc/systemd/system/xrd-overhead-camera.service
"@
$arm02Result = @(& ssh @Arm02Ssh $Arm02Target $arm02Install 2>&1)
if ($LASTEXITCODE -ne 0) { throw "arm02 install failed: $($arm02Result -join ' ')" }

$ErrorActionPreference = $saved
$Evidence.arm02_verification = @($arm02Result | ForEach-Object { [string]$_ })
$Evidence.arm02_automatic_ager_disabled = $true
$Evidence.arm02_robot_serial_owner_after_deploy = $null
$Evidence.completed_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
$Evidence.status = "pass_files_installed_services_stopped"
Write-Evidence $Evidence
