param(
  [switch]$Execute,
  [switch]$OperatorAuthorizedTeachPreparation,
  [string]$EvidenceOut = ""
)

$ErrorActionPreference = "Stop"
$KnownHosts = Join-Path $PSScriptRoot "evidence/arm_pair_10.185.110_known_hosts"
$SshOptions = @(
  "-o", "BatchMode=yes",
  "-o", "ConnectTimeout=12",
  "-o", "StrictHostKeyChecking=yes",
  "-o", "UpdateHostKeys=no",
  "-o", "UserKnownHostsFile=$KnownHosts"
)

function Write-Evidence([object]$Value) {
  $json = $Value | ConvertTo-Json -Depth 8
  if ($EvidenceOut) {
    $parent = Split-Path -Parent $EvidenceOut
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
      New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    Set-Content -LiteralPath $EvidenceOut -Value $json -Encoding UTF8
  }
  Write-Output $json
}

$CanExecute = [bool]($Execute -and $OperatorAuthorizedTeachPreparation)
$Evidence = [ordered]@{
  schema_version = "xrd-dual-arm-teach-serial-preparation-v1"
  created_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
  mode = if ($CanExecute) { "execute" } else { "dry_run" }
  pc_network_changed = $false
  robot_serial_opened = $false
  robot_sdk_imported = $false
  servo_released = $false
  motion_command_sent = $false
  planned_steps = @(
    "verify arm01 and arm02 immutable identities",
    "stop xrd-workcockpit.service on arm01",
    "verify automatic-ager remains absent on arm02",
    "verify /dev/ttyAMA0 has no owner on either Pi"
  )
  motion_ready = $false
  teach_ready = $false
}

if (-not $CanExecute) {
  if ($Execute -xor $OperatorAuthorizedTeachPreparation) {
    $Evidence.status = "blocked_incomplete_execute_gate"
    Write-Evidence $Evidence
    exit 2
  }
  $Evidence.status = "planned_only"
  Write-Evidence $Evidence
  exit 0
}

$arm01Command = @'
set -eu
test "$(hostname)" = mycobot-arm-01
grep -q '^Serial[[:space:]]*: 1000000092fb92d3$' /proc/cpuinfo
test "$(cat /sys/class/net/wlan0/address)" = e4:5f:01:bf:de:a7
hostname -I | tr ' ' '\n' | grep -qx 192.0.2.64
sudo -n systemctl stop xrd-workcockpit.service
test "$(systemctl is-active xrd-workcockpit.service 2>/dev/null || true)" != active
if fuser /dev/ttyAMA0 >/tmp/xrd-arm01-serial-owner 2>/dev/null; then
  echo "arm01 serial remains owned: $(cat /tmp/xrd-arm01-serial-owner)" >&2
  exit 4
fi
echo arm01_identity=pass
echo arm01_workcockpit=stopped
echo arm01_serial_owner=none
'@

$arm02Command = @'
set -eu
test "$(hostname)" = er
grep -q '^Serial[[:space:]]*: 10000000f08c41fc$' /proc/cpuinfo
test "$(cat /sys/class/net/wlan0/address)" = 98:fe:54:0c:94:07
hostname -I | tr ' ' '\n' | grep -qx 192.0.2.136
if crontab -l 2>/dev/null | grep -q '/home/rdk/automatic-ager/runner.sh'; then
  echo 'arm02 automatic-ager reboot entry returned' >&2
  exit 5
fi
if fuser /dev/ttyAMA0 >/tmp/xrd-arm02-serial-owner 2>/dev/null; then
  echo "arm02 serial is owned: $(cat /tmp/xrd-arm02-serial-owner)" >&2
  exit 4
fi
echo arm02_identity=pass
echo arm02_automatic_ager=absent
echo arm02_serial_owner=none
'@

$arm01 = @(& ssh @SshOptions er@198.51.100.136 $arm01Command 2>&1)
if ($LASTEXITCODE -ne 0) { throw "arm01 teach preparation failed: $($arm01 -join ' ')" }
$arm02 = @(& ssh @SshOptions er@198.51.100.145 $arm02Command 2>&1)
if ($LASTEXITCODE -ne 0) { throw "arm02 teach preparation failed: $($arm02 -join ' ')" }

$Evidence.arm01 = @($arm01 | ForEach-Object { [string]$_ })
$Evidence.arm02 = @($arm02 | ForEach-Object { [string]$_ })
$Evidence.teach_serials_available = $true
$Evidence.status = "pass_serials_available_no_motion"
$Evidence.completed_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
Write-Evidence $Evidence
