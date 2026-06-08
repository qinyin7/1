$ErrorActionPreference = "Stop"

$rawDir = Join-Path $PSScriptRoot "..\data\raw"
$extractDir = Join-Path $rawDir "extracted"
New-Item -ItemType Directory -Force -Path $rawDir, $extractDir | Out-Null

$files = @(
    @{
        Name = "KuaiRec.zip"
        Url = "https://zenodo.org/api/records/18164998/files/KuaiRec.zip/content"
        Md5 = "261550d472c48eff4990fb13c0e5bcf7"
    },
    @{
        Name = "video_raw_categories_multi.csv"
        Url = "https://zenodo.org/api/records/18164998/files/video_raw_categories_multi.csv/content"
        Md5 = "d05eea147135d2cdf7759fba5c0d70d4"
    },
    @{
        Name = "user_features_raw.csv"
        Url = "https://zenodo.org/api/records/18164998/files/user_features_raw.csv/content"
        Md5 = "3969b8120035e7ced36d56926a7cbd24"
    }
)

foreach ($file in $files) {
    $target = Join-Path $rawDir $file.Name
    if (-not (Test-Path $target)) {
        Write-Host "Downloading $($file.Name)..."
        curl.exe -L --retry 5 --retry-delay 3 -o $target $file.Url
    }
    $actual = (Get-FileHash -Algorithm MD5 $target).Hash.ToLowerInvariant()
    if ($actual -ne $file.Md5) {
        throw "Checksum mismatch for $($file.Name): expected $($file.Md5), got $actual"
    }
    Write-Host "Verified $($file.Name)"
}

$zipPath = Join-Path $rawDir "KuaiRec.zip"
if (-not (Test-Path (Join-Path $extractDir "KuaiRec 2.0\data\big_matrix.csv"))) {
    Write-Host "Extracting KuaiRec.zip..."
    Expand-Archive -LiteralPath $zipPath -DestinationPath $extractDir -Force
}

Write-Host "KuaiRec data is ready under data\raw."

