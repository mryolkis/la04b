param(
    [string]$FoobarPath = "$env:ProgramFiles\foobar2000",
    [ValidateSet('x64', 'Win32')][string]$Platform = 'x64',
    [switch]$SeekBenchmark,
    [string]$DecoderPath
)
$ErrorActionPreference = 'Stop'
$component = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $component
$build = Join-Path $component 'build'
$player = Join-Path $build "test-player-$Platform"
if (!(Test-Path "$FoobarPath\foobar2000.exe")) { throw 'Specify an existing foobar2000 installation matching -Platform.' }
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$temp = [IO.Path]::GetTempFileName()
try {
    Start-Process $vswhere -ArgumentList '-latest -products * -property installationPath' -Wait -NoNewWindow -RedirectStandardOutput $temp
    $vs = (Get-Content $temp -Raw).Trim()
} finally { Remove-Item $temp }
$p = Start-Process "$vs\MSBuild\Current\Bin\MSBuild.exe" -ArgumentList @('"' + "$component\host_test.vcxproj" + '"', '/m', '/nodeReuse:false', '/nologo', '/v:minimal', '/p:Configuration=Release', "/p:Platform=$Platform") -PassThru -NoNewWindow
$handle = $p.Handle
$p.WaitForExit()
if ($p.ExitCode -ne 0) { throw 'Host test driver build failed.' }
New-Item -ItemType Directory -Force $player | Out-Null
Copy-Item "$FoobarPath\*.dll" $player -Force
Copy-Item "$FoobarPath\foobar2000.exe" $player -Force
Copy-Item "$FoobarPath\components" $player -Recurse -Force
if (Test-Path "$FoobarPath\runtime") { Copy-Item "$FoobarPath\runtime" $player -Recurse -Force }
[IO.File]::WriteAllText("$player\portable_mode_enabled", '')
$plugin_dir = if ($Platform -eq 'x64') { 'user-components-x64' } else { 'user-components' }
$plugins = "$player\profile\$plugin_dir"
New-Item -ItemType Directory -Force "$plugins\foo_input_la", "$plugins\foo_la_test" | Out-Null
if (!$DecoderPath) { $DecoderPath = "$build\$Platform\foo_input_la.dll" }
Copy-Item $DecoderPath "$plugins\foo_input_la" -Force
Copy-Item "$build\test-driver\$Platform\foo_la_test.dll" "$plugins\foo_la_test" -Force
$reportName = if ($SeekBenchmark) { 'seek-benchmark' } else { 'host-test-results' }
$report = "$build\$reportName-$Platform.txt"
if (Test-Path $report) { Remove-Item $report }
$savedBenchmark = $env:LA04_SEEK_BENCH
$env:LA04_SEEK_BENCH = if ($SeekBenchmark) { '1' } else { $null }
$savedRoot = $env:LA04_TEST_ROOT
$env:LA04_TEST_ROOT = $root
try {
    $process = Start-Process "$player\foobar2000.exe" -ArgumentList '/quiet' -PassThru
    if (!$process.WaitForExit(120000)) {
        $process.Kill()
        throw 'Portable test player did not finish within 120 seconds.'
    }
} finally { $env:LA04_TEST_ROOT = $savedRoot; $env:LA04_SEEK_BENCH = $savedBenchmark }
if (!(Test-Path $report)) { throw 'Host test did not run: no report. Check component loading in the portable test player.' }
Get-Content $report
if (!(Select-String -Path $report -Pattern '^SUCCESS:')) { throw 'Host integration tests failed.' }
