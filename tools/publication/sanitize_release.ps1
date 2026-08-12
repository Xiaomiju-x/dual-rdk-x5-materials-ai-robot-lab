[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,

    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $RepositoryRoot).Path

if (-not (Test-Path -LiteralPath (Join-Path $root ".git"))) {
    throw "RepositoryRoot must be an isolated Git working tree."
}

$textExtensions = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
@(
    ".c", ".cc", ".cpp", ".h", ".hpp", ".py", ".ps1", ".sh",
    ".js", ".mjs", ".ts", ".tsx", ".vue", ".css", ".html",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".md", ".rst", ".txt", ".csv", ".sha256", ".xml", ".xacro",
    ".urdf", ".launch", ".rviz", ".service", ".uvprojx", ".uvoptx",
    ".ld", ".cmake", ".msg", ".srv", ".action"
) | ForEach-Object { [void]$textExtensions.Add($_) }

$specialNames = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
@("CMakeLists.txt", "package.xml", "requirements.txt", "Makefile") |
    ForEach-Object { [void]$specialNames.Add($_) }

$privateIpv4 = [regex]::new(
    "(?<![0-9])(?:" +
    "10(?:\.[0-9]{1,3}){3}|" +
    "192\.168(?:\.[0-9]{1,3}){2}|" +
    "172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2}" +
    ")(?![0-9])"
)
$hostKey = [regex]::new("ssh-ed25519\s+[A-Za-z0-9+/=]{20,}")
$apiKeyPattern = [regex]::new("(?<![A-Za-z0-9])sk-[A-Za-z0-9]{20,}")
$windowsUser = [regex]::new(('C:' + '\\Users\\' + '[^\\\s]+'))
$forwardWindowsUser = [regex]::new(('C:' + '/Users/' + '[^/\s]+'))
$linuxHome = [regex]::new(('/' + 'home/' + '(?!rdk(?:/|$))[^/\s]+'))
$privateDeviceLogin = [regex]::new(
    '(?<![A-Za-z0-9._-])[A-Za-z][A-Za-z0-9_-]{1,31}@' +
    '(?=(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.))'
)

$files = Get-ChildItem -LiteralPath $root -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object {
        $_.FullName -ne $PSCommandPath -and
        $_.FullName -notmatch "[\\/]\.git[\\/]" -and
        ($textExtensions.Contains($_.Extension) -or $specialNames.Contains($_.Name))
    }

$wouldChange = [System.Collections.Generic.List[string]]::new()
$utf8 = [System.Text.UTF8Encoding]::new($false)

foreach ($file in $files) {
    $content = [System.IO.File]::ReadAllText($file.FullName, $utf8)
    $updated = $apiKeyPattern.Replace($content, "REDACTED_API_KEY")
    $updated = $hostKey.Replace($updated, "ssh-ed25519 REPLACE_WITH_VERIFIED_HOST_KEY")
    $updated = $privateDeviceLogin.Replace($updated, 'rdk@')
    $updated = $privateIpv4.Replace($updated, {
        param($match)
        $octets = $match.Value.Split(".")
        $last = $octets[3]
        if ($octets[0] -eq "192") { return "192.0.2.$last" }
        if ($octets[0] -eq "10") { return "198.51.100.$last" }
        return "203.0.113.$last"
    })
    $updated = $windowsUser.Replace($updated, 'C:\Users\YOUR_USER')
    $updated = $forwardWindowsUser.Replace($updated, 'C:/Users/YOUR_USER')
    $updated = $linuxHome.Replace($updated, '/home/rdk')

    if ($updated -eq $content) {
        continue
    }

    $relative = $file.FullName.Substring($root.Length).TrimStart([char]92, [char]47)
    $wouldChange.Add($relative)
    if ($Apply) {
        [System.IO.File]::WriteAllText($file.FullName, $updated, $utf8)
    }
}

$verb = if ($Apply) { "Sanitized" } else { "Would sanitize" }
Write-Host "$verb $($wouldChange.Count) text files. Sensitive values were not printed."
if (-not $Apply) {
    Write-Host "Dry run only. Re-run with -Apply after reviewing the publication boundary."
}
