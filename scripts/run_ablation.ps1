param(
    [ValidateSet("local_8gb_large", "full_24gb")]
    [string]$Profile = "local_8gb_large",
    [string]$Seeds = "2026"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = $PSScriptRoot + "\.."

$pythonArguments = @("scripts/run_suite.py", "--profile", $Profile, "--seeds")
$pythonArguments += $Seeds.Split(",") | ForEach-Object { $_.Trim() }
& python @pythonArguments
if ($LASTEXITCODE -ne 0) {
    throw "Experiment suite failed for profile: $Profile"
}

Write-Host "Full-exposure validation complete. Results: artifacts\$Profile\recall\results.csv and artifacts\$Profile\ranking\results.csv"
