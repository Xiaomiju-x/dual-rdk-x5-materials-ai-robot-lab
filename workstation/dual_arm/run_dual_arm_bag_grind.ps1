[CmdletBinding()]
param(
    [ValidateRange(1, 20)]
    [int]$GrindCycles = 4,
    [switch]$ValidateOnly,
    [switch]$Execute,
    [switch]$PlanOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$SelectedModeCount = [int]$ValidateOnly.IsPresent + [int]$Execute.IsPresent + [int]$PlanOnly.IsPresent
if ($SelectedModeCount -gt 1) {
    throw 'Choose only one of -ValidateOnly, -Execute, or -PlanOnly.'
}
if ($SelectedModeCount -eq 0) {
    $PlanOnly = $true
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$KnownHosts = Join-Path $RepoRoot 'rb_voe\live_known_hosts'
$ExpectedKnownHostsSha256 = '79fc15d37314f1abeae2b07952695f666c993272453fc582b6e571e42dd4212f'
$LeftFlowHash = 'a52062242654db16a4061ef4d376b737dc010e5084ce5be68ca2934c0d141b8f'
$RightFlowHash = 'c070db7c87455723dd43b3d4727f7968343fa0200483c68b68cd9e4ccb518619'

$Targets = [ordered]@{
    arm01 = [pscustomobject]@{
        Id = 'arm01'
        User = 'er'
        Address = '192.0.2.64'
        UseAiJump = $true
        HostName = 'mycobot-arm-01'
        WlanMac = 'e4:5f:01:bf:de:a7'
        CpuSerial = '1000000092fb92d3'
        HostKey = 'ssh-ed25519 REPLACE_WITH_VERIFIED_HOST_KEY'
    }
    arm02 = [pscustomobject]@{
        Id = 'arm02'
        User = 'er'
        Address = '192.0.2.136'
        UseAiJump = $true
        HostName = 'er'
        WlanMac = '98:fe:54:0c:94:07'
        CpuSerial = '10000000f08c41fc'
        HostKey = 'ssh-ed25519 REPLACE_WITH_VERIFIED_HOST_KEY'
    }
    ai_x5 = [pscustomobject]@{
        Id = 'ai_x5'
        User = 'sunrise'
        Address = '192.0.2.103'
        UseAiJump = $false
        HostName = 'xrd-ai'
        WlanMac = 'b4:2f:03:31:97:b9'
        CpuSerial = ''
        HostKey = 'ssh-ed25519 REPLACE_WITH_VERIFIED_HOST_KEY'
    }
}

function New-AiJumpProxyCommand {
    if ($KnownHosts -match '\s') {
        throw 'The frozen known_hosts path contains whitespace and cannot be embedded safely in ProxyCommand.'
    }

    $jump = $Targets.ai_x5
    return (@(
        'ssh.exe',
        '-F', 'NUL',
        '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=8',
        '-o', 'ConnectionAttempts=1',
        '-o', 'ServerAliveInterval=3',
        '-o', 'ServerAliveCountMax=2',
        '-o', 'StrictHostKeyChecking=yes',
        '-o', "UserKnownHostsFile=$KnownHosts",
        '-o', 'GlobalKnownHostsFile=NUL',
        '-o', 'KnownHostsCommand=none',
        '-o', "HostKeyAlias=$($jump.Address)",
        '-o', 'HostKeyAlgorithms=ssh-ed25519',
        '-o', 'UpdateHostKeys=no',
        '-o', 'CheckHostIP=yes',
        '-o', 'PasswordAuthentication=no',
        '-o', 'KbdInteractiveAuthentication=no',
        '-o', 'IdentitiesOnly=yes',
        '-o', 'IdentityAgent=none',
        '-o', 'ForwardAgent=no',
        '-o', 'ClearAllForwardings=yes',
        '-o', 'PermitLocalCommand=no',
        '-o', 'ProxyCommand=none',
        '-o', 'ProxyJump=none',
        '-o', 'CanonicalizeHostname=no',
        '-o', 'VerifyHostKeyDNS=no',
        '-o', 'ControlMaster=no',
        '-o', 'ControlPath=none',
        '-o', 'ControlPersist=no',
        '-o', 'RequestTTY=no',
        '-o', 'LogLevel=ERROR',
        '-W', '%h:%p',
        "$($jump.User)@$($jump.Address)"
    ) -join ' ')
}

function Assert-FrozenKnownHosts {
    param([Parameter(Mandatory)][string]$Path)

    $item = Get-Item -LiteralPath $Path -Force
    if (-not ($item -is [System.IO.FileInfo])) {
        throw 'Frozen known_hosts path is not a regular file.'
    }
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw 'Frozen known_hosts must not be a link or reparse point.'
    }
    $observedHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($observedHash -ne $ExpectedKnownHostsSha256) {
        throw "Frozen known_hosts SHA-256 mismatch: $observedHash"
    }

    $entries = @(
        Get-Content -LiteralPath $Path -Encoding UTF8 |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ -and -not $_.StartsWith('#') }
    )
    foreach ($target in $Targets.Values) {
        $expected = "$($target.Address) $($target.HostKey)"
        if (@($entries | Where-Object { $_ -ceq $expected }).Count -ne 1) {
            throw "Frozen ED25519 host key is missing or ambiguous for $($target.Id)."
        }
    }
}

function New-StrictSshArguments {
    param([Parameter(Mandatory)][psobject]$Target)

    $arguments = @(
        '-F', 'NUL',
        '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=8',
        '-o', 'ConnectionAttempts=1',
        '-o', 'ServerAliveInterval=3',
        '-o', 'ServerAliveCountMax=2',
        '-o', 'StrictHostKeyChecking=yes',
        '-o', "UserKnownHostsFile=$KnownHosts",
        '-o', 'GlobalKnownHostsFile=NUL',
        '-o', 'KnownHostsCommand=none',
        '-o', "HostKeyAlias=$($Target.Address)",
        '-o', 'HostKeyAlgorithms=ssh-ed25519',
        '-o', 'UpdateHostKeys=no',
        '-o', 'CheckHostIP=yes',
        '-o', 'PasswordAuthentication=no',
        '-o', 'KbdInteractiveAuthentication=no',
        '-o', 'IdentitiesOnly=yes',
        '-o', 'IdentityAgent=none',
        '-o', 'ForwardAgent=no',
        '-o', 'ClearAllForwardings=yes',
        '-o', 'PermitLocalCommand=no',
        '-o', 'CanonicalizeHostname=no',
        '-o', 'VerifyHostKeyDNS=no',
        '-o', 'ControlMaster=no',
        '-o', 'ControlPath=none',
        '-o', 'ControlPersist=no',
        '-o', 'RequestTTY=no',
        '-o', 'LogLevel=ERROR'
    )
    if ($Target.UseAiJump) {
        $arguments += @('-o', "ProxyCommand=$(New-AiJumpProxyCommand)", '-o', 'ProxyJump=none')
    }
    else {
        $arguments += @('-o', 'ProxyCommand=none', '-o', 'ProxyJump=none')
    }
    $arguments += "$($Target.User)@$($Target.Address)"
    return $arguments
}

function Invoke-StreamingSsh {
    param([Parameter(Mandatory)][string[]]$Arguments)

    $captured = [System.Collections.Generic.List[string]]::new()
    $previousErrorActionPreference = $ErrorActionPreference
    $sshExitCode = -1
    try {
        $ErrorActionPreference = 'Continue'
        & ssh.exe @Arguments 2>&1 | ForEach-Object {
            $line = [string]$_
            $captured.Add($line)
            Write-Host $line
        }
        $sshExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($sshExitCode -ne 0) {
        throw "SSH command failed with exit code $sshExitCode"
    }
    return ,$captured.ToArray()
}

function Start-CapturedSshJob {
    param([Parameter(Mandatory)][string[]]$Arguments)

    return Start-Job -ScriptBlock {
        param([string[]]$SshArguments)

        $ErrorActionPreference = 'Continue'
        $captured = @(
            & ssh.exe @SshArguments 2>&1 | ForEach-Object { [string]$_ }
        )
        [pscustomobject]@{
            ExitCode = [int]$LASTEXITCODE
            Lines = [string[]]$captured
        }
    } -ArgumentList (,$Arguments)
}

Assert-FrozenKnownHosts -Path $KnownHosts

$plan = [ordered]@{
    schema_version = 'xrd-dual-arm-finals-v3-plan-v1'
    mode = if ($PlanOnly) { 'PLAN_ONLY' } elseif ($Execute) { 'EXECUTE' } else { 'VALIDATE_ONLY' }
    network = [ordered]@{
        lan = 'xrd-lab_5G'
        fixed_targets = @($Targets.Values | ForEach-Object { $_.Address })
        arm01_transport = 'fixed_ai_x5_192.0.2.103_jump'
        arm02_transport = 'fixed_ai_x5_192.0.2.103_jump'
        known_hosts_sha256 = $ExpectedKnownHostsSha256
        automatic_pc_network_changes = $false
        discovery_or_cache_used = $false
    }
    grind_cycles = $GrindCycles
    explicit_motion_authority = [bool]$Execute
}
if ($PlanOnly) {
    $plan | ConvertTo-Json -Depth 6
    exit 0
}

$leftSsh = New-StrictSshArguments -Target $Targets.arm01
$rightSsh = New-StrictSshArguments -Target $Targets.arm02
$aiSsh = New-StrictSshArguments -Target $Targets.ai_x5

$aiPreflight = 'test "$(id -un)" = sunrise && test "$(hostname)" = xrd-ai && test "$(cat /sys/class/net/wlan0/address)" = b4:2f:03:31:97:b9'
$leftPreflight = 'test "$(id -un)" = er && test "$(hostname)" = mycobot-arm-01 && test "$(cat /sys/class/net/wlan0/address)" = e4:5f:01:bf:de:a7 && grep -q 1000000092fb92d3 /proc/cpuinfo && test "$(systemctl is-active xrd-workcockpit.service 2>/dev/null || true)" = inactive && test "$(systemctl is-enabled xrd-workcockpit.service 2>/dev/null || true)" = disabled && test -z "$(lsof -t /dev/ttyAMA0 2>/dev/null)" && test "$(sha256sum /home/rdk/arm01_compact_front_transfer.py | awk ''{print $1}'')" = ' + $LeftFlowHash
$rightPreflight = 'test "$(id -un)" = er && test "$(hostname)" = er && test "$(cat /sys/class/net/wlan0/address)" = 98:fe:54:0c:94:07 && grep -q 10000000f08c41fc /proc/cpuinfo && test "$(systemctl is-active xrd-overhead-camera.service 2>/dev/null || true)" = inactive && test "$(systemctl is-enabled xrd-overhead-camera.service 2>/dev/null || true)" = disabled && test -z "$(lsof -t /dev/ttyAMA0 2>/dev/null)" && test -z "$(lsof -t /dev/video0 2>/dev/null)" && test "$(sha256sum /home/rdk/xrd/workstation/dual_arm/arm02_direct_grind_closed_loop.py | awk ''{print $1}'')" = ' + $RightFlowHash

Invoke-StreamingSsh -Arguments ($aiSsh + $aiPreflight) | Out-Null
Invoke-StreamingSsh -Arguments ($leftSsh + $leftPreflight) | Out-Null
Invoke-StreamingSsh -Arguments ($rightSsh + $rightPreflight) | Out-Null
Write-Host '[dual-arm] fixed host keys, identities, owners, services, and finals hashes verified'

if (-not $Execute) {
    Write-Host '[dual-arm] validate-only PASS; no motion command sent'
    exit 0
}

Write-Host '[dual-arm] LEFT phase: bag pickup, dish drop, vertical retract to dish clear top'
$leftOutput = Invoke-StreamingSsh -Arguments (
    $leftSsh + 'cd /home/rdk && timeout 240s python3 -u arm01_compact_front_transfer.py bag-drop-dish-top --speed 10 --timeout 90'
)
$leftText = $leftOutput -join "`n"
if ($leftText -notmatch '"flow":\s*"bag_drop_dish_top"' -or $leftText -notmatch '"result":\s*"completed_dish_clear_top"') {
    throw 'Left flow did not reach the dish-side clear top; right arm remains blocked.'
}

Write-Host '[dual-arm] OVERLAP phase: left returns START while right grinds'
$leftReturnArguments = $leftSsh + (
    'cd /home/rdk && timeout 150s python3 -u arm01_compact_front_transfer.py dish-top-return-start --speed 10 --timeout 90'
)
$leftReturnJob = Start-CapturedSshJob -Arguments $leftReturnArguments
$rightOutput = $null
$rightFailure = $null
try {
    $rightOutput = Invoke-StreamingSsh -Arguments (
        $rightSsh + "cd /home/rdk/xrd/workstation/dual_arm && timeout 180s python3 -u arm02_direct_grind_closed_loop.py --cycles $GrindCycles"
    )
}
catch {
    $rightFailure = $_
}

$leftReturnResult = Receive-Job -Job $leftReturnJob -Wait
Remove-Job -Job $leftReturnJob -Force
$leftReturnLines = @($leftReturnResult.Lines | ForEach-Object { [string]$_ })
$leftReturnLines | ForEach-Object { Write-Host $_ }
$leftReturnText = $leftReturnLines -join "`n"
if ($null -eq $leftReturnResult -or $leftReturnResult.ExitCode -ne 0) {
    throw "Concurrent left return failed with exit code $($leftReturnResult.ExitCode)."
}
if ($leftReturnText -notmatch '"result":\s*"completed_left_start"') {
    throw 'Concurrent left return did not emit completed_left_start.'
}
if ($null -ne $rightFailure) {
    throw $rightFailure
}
$rightText = $rightOutput -join "`n"
if ($rightText -notmatch '"event":\s*"CLOSED_LOOP_DONE"') {
    throw 'Right flow did not emit CLOSED_LOOP_DONE.'
}

Write-Host '[dual-arm] CLOSED_LOOP_DONE: left START and right START'
