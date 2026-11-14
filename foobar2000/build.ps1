param([ValidateSet('All', 'x64', 'Win32')][string]$Platform = 'All')
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
function Invoke-Native([string]$File, [string[]]$Arguments) {
    $quoted = ($Arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $process = Start-Process $File -ArgumentList $quoted -PassThru -NoNewWindow
    $handle = $process.Handle
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) { throw "$File failed with exit code $($process.ExitCode)" }
}
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsOutput = [IO.Path]::GetTempFileName()
try {
    $lookup = Start-Process $vswhere -ArgumentList '-latest -products * -property installationPath' -Wait -PassThru -NoNewWindow -RedirectStandardOutput $vsOutput
    $vs = (Get-Content $vsOutput -Raw).Trim()
} finally { Remove-Item $vsOutput }
if (!$vs) { throw 'Install Visual Studio 2022 Build Tools with Desktop development with C++.' }
Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
$msbuild = "$vs\MSBuild\Current\Bin\MSBuild.exe"
$sdk = Join-Path $root 'vendor\sdk'
if (!(Test-Path "$sdk\foobar2000\SDK\foobar2000.h")) {
    $archive = Join-Path $root 'vendor\SDK-2026-10-01.7z'
    New-Item -ItemType Directory -Force (Split-Path $archive) | Out-Null
    Invoke-WebRequest 'https://www.foobar2000.org/downloads/SDK-2026-10-01.7z' -OutFile $archive
    $expectedHash = 'D4C55077336FAE81BF8DF0259B5B2748FA45EA84132C656EAD93EB123CBCDC26'
    if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne $expectedHash) { throw 'Unexpected SDK archive hash.' }
    $sevenZip = "$env:ProgramFiles\7-Zip\7z.exe"
    if (!(Test-Path $sevenZip)) { throw 'Install 7-Zip to extract the official foobar2000 SDK.' }
    Invoke-Native $sevenZip @('x', '-y', $archive, "-o$sdk")
}
$platforms = if ($Platform -eq 'All') { @('Win32', 'x64') } else { @($Platform) }
Push-Location $root
try {
    foreach ($arch in $platforms) {
        $target = if ($arch -eq 'x64') { 'x86_64-pc-windows-msvc' } else { 'i686-pc-windows-msvc' }
        Invoke-Native (Get-Command rustup.exe).Source @('target', 'add', '--toolchain', '1.100.0', $target)
        Invoke-Native (Get-Command cargo.exe).Source @('build', '--release', '--locked', '--manifest-path', 'rust/Cargo.toml', '--target', $target)
        Invoke-Native $msbuild @('foo_input_la.vcxproj', '/m', '/nodeReuse:false', '/nologo', '/v:minimal', '/p:Configuration=Release', "/p:Platform=$arch")
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $dist = Join-Path $root 'dist'
    New-Item -ItemType Directory -Force $dist | Out-Null
    $suffix = if ($Platform -eq 'All') { '' } else { "-$Platform" }
    $package = Join-Path $dist "foo_input_la$suffix.fb2k-component"
    if (Test-Path $package) { Remove-Item $package }
    $zip = [IO.Compression.ZipFile]::Open($package, 'Create')
    try {
        foreach ($arch in $platforms) {
            $entry = if ($arch -eq 'x64') { 'x64/foo_input_la.dll' } else { 'foo_input_la.dll' }
            [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, "$root\build\$arch\foo_input_la.dll", $entry) | Out-Null
        }
    } finally { $zip.Dispose() }
    Write-Output "Component ready: $package"
} finally { Pop-Location }
