<#
.SYNOPSIS
Reclaims Windows disk space from a WSL 2 distribution.

.DESCRIPTION
Resolves the distribution VHD from the WSL registry, flushes and trims its
Linux filesystem, shuts down WSL, and compacts the detached VHD with DiskPart.

Run this script from an elevated Windows PowerShell. WSL remains stopped after
compaction and starts again on the next WSL command.

.PARAMETER DistributionName
The registered WSL distribution to compact. When omitted, the default
distribution is used.

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "\\wsl.localhost\Ubuntu\home\boaz\aigen\scripts\reclaim_wsl_disk.ps1"

.EXAMPLE
.\reclaim_wsl_disk.ps1 -DistributionName Ubuntu -WhatIf
#>

#Requires -Version 5.1

[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
param(
    [Parameter(Position = 0)]
    [string]$DistributionName
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Log {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    Write-Host "[aigen] $Message"
}

function Format-ByteCount {
    param(
        [Parameter(Mandatory = $true)]
        [long]$Bytes
    )

    return "{0:N2} GiB" -f ($Bytes / 1GB)
}

function Get-WslRegistration {
    param(
        [string]$Name
    )

    $lxssPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss"
    $lxss = Get-Item -LiteralPath $lxssPath

    if ([string]::IsNullOrWhiteSpace($Name)) {
        $defaultDistribution = [string]$lxss.GetValue("DefaultDistribution")
        if ([string]::IsNullOrWhiteSpace($defaultDistribution)) {
            throw "WSL has no default distribution."
        }

        $registration = Get-Item -LiteralPath (Join-Path $lxssPath $defaultDistribution)
    }
    else {
        $matches = @(
            Get-ChildItem -LiteralPath $lxssPath |
                Where-Object { $_.GetValue("DistributionName") -eq $Name }
        )
        if ($matches.Count -ne 1) {
            throw "Expected one registered WSL distribution named '$Name'; found $($matches.Count)."
        }

        $registration = $matches[0]
    }

    $resolvedName = [string]$registration.GetValue("DistributionName")
    $version = [int]$registration.GetValue("Version")
    if ($version -ne 2) {
        throw "Distribution '$resolvedName' uses WSL $version; only WSL 2 has a compactable VHD."
    }

    $basePath = [string]$registration.GetValue("BasePath")
    if ([string]::IsNullOrWhiteSpace($basePath)) {
        throw "Distribution '$resolvedName' has no registered BasePath."
    }

    $vhdFileName = [string]$registration.GetValue("VhdFileName")
    if ([string]::IsNullOrWhiteSpace($vhdFileName)) {
        $vhdFileName = "ext4.vhdx"
    }

    $vhdPath = [System.IO.Path]::GetFullPath(
        [System.IO.Path]::Combine($basePath, $vhdFileName)
    )
    $extension = [System.IO.Path]::GetExtension($vhdPath)
    if ($extension -notin @(".vhd", ".vhdx")) {
        throw "Distribution '$resolvedName' points to an unsupported disk image: $vhdPath"
    }

    $vhd = Get-Item -LiteralPath $vhdPath
    if ($vhd.PSIsContainer) {
        throw "Registered WSL disk path is not a file: $vhdPath"
    }

    return [pscustomobject]@{
        Name = $resolvedName
        VhdPath = $vhd.FullName
    }
}

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

$distribution = Get-WslRegistration -Name $DistributionName
$vhd = Get-Item -LiteralPath $distribution.VhdPath

Write-Log "distribution: $($distribution.Name)"
Write-Log "VHD: $($distribution.VhdPath)"
Write-Log "current VHD file size: $(Format-ByteCount -Bytes $vhd.Length)"

$operation = "trim '$($distribution.Name)', shut down all WSL distributions, and compact its VHD"
if (-not $PSCmdlet.ShouldProcess($distribution.VhdPath, $operation)) {
    return
}

if (-not (Test-Administrator)) {
    throw "Run this script from an elevated Windows PowerShell."
}

$wsl = (Get-Command wsl.exe -ErrorAction Stop).Source
$diskPart = Join-Path $env:SystemRoot "System32\diskpart.exe"
if (-not (Test-Path -LiteralPath $diskPart -PathType Leaf)) {
    throw "DiskPart is unavailable: $diskPart"
}

$beforeLength = (Get-Item -LiteralPath $distribution.VhdPath).Length

Write-Log "flush Linux filesystem"
& $wsl --distribution $distribution.Name --user root --exec sync
if ($LASTEXITCODE -ne 0) {
    throw "Linux filesystem sync failed with exit code $LASTEXITCODE."
}

Write-Log "trim unused Linux filesystem blocks"
& $wsl --distribution $distribution.Name --user root --exec fstrim --verbose /
if ($LASTEXITCODE -ne 0) {
    throw "Linux filesystem trim failed with exit code $LASTEXITCODE."
}

Write-Log "shut down all WSL distributions"
& $wsl --shutdown
if ($LASTEXITCODE -ne 0) {
    throw "WSL shutdown failed with exit code $LASTEXITCODE."
}

$diskPartScript = Join-Path $env:TEMP (
    "aigen-diskpart-{0}.txt" -f [guid]::NewGuid().ToString("N")
)

try {
    [System.IO.File]::WriteAllLines(
        $diskPartScript,
        @(
            "select vdisk file=`"$($distribution.VhdPath)`"",
            "compact vdisk"
        ),
        [Text.Encoding]::ASCII
    )

    Write-Log "compact detached WSL VHD"
    & $diskPart /s $diskPartScript
    if ($LASTEXITCODE -ne 0) {
        throw "DiskPart failed with exit code $LASTEXITCODE."
    }
}
finally {
    Remove-Item -LiteralPath $diskPartScript -ErrorAction SilentlyContinue
}

$afterLength = (Get-Item -LiteralPath $distribution.VhdPath).Length
$reclaimed = $beforeLength - $afterLength

Write-Log "VHD file size: $(Format-ByteCount -Bytes $beforeLength) -> $(Format-ByteCount -Bytes $afterLength)"
Write-Log "reclaimed: $(Format-ByteCount -Bytes $reclaimed)"
Write-Log "WSL remains stopped and will restart on the next WSL command"
