$uploadsDir = "c:\Users\tsaks\OneDrive\Desktop\proj_demand\uploads"
$allFiles = Get-ChildItem $uploadsDir | Sort-Object LastWriteTime -Descending
$total = $allFiles.Count
$keepCount = 6

Write-Host "Total files found: $total"

if ($total -le $keepCount) {
    Write-Host "Nothing to delete — only $total files exist."
    exit
}

$toDelete = $allFiles | Select-Object -Skip $keepCount
Write-Host "Keeping: $keepCount most recent files"
Write-Host "Deleting: $($toDelete.Count) old files..."

foreach ($f in $toDelete) {
    Remove-Item $f.FullName -Force
    Write-Host "  Deleted: $($f.Name)"
}

$remaining = (Get-ChildItem $uploadsDir).Count
Write-Host ""
Write-Host "Done! Remaining files: $remaining"
