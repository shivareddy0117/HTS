# PowerShell script to properly commit large image files to Git

Write-Host "=== HTS Image Commit Tool ===" -ForegroundColor Cyan
Write-Host "This script will help you add and commit your dashboard images to Git"
Write-Host ""

# Make sure we're in the project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $scriptPath "..")

# Check if Git is installed
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git is not installed. Please install Git first." -ForegroundColor Red
    exit 1
}

# Check if Git repo exists
if (!(Test-Path .git)) {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
}

Write-Host "Adding dashboard images..." -ForegroundColor Green
git add docs/images/dashboard_*.png

# Check if there are other placeholder images to replace
$placeholders = @(
    "docs/images/system_overview.png", 
    "docs/images/architecture.png",
    "docs/images/cv_comparison.png",
    "docs/images/nfq_architecture.png"
)

Write-Host ""
Write-Host "The following placeholder images still need to be replaced:" -ForegroundColor Yellow
foreach ($placeholder in $placeholders) {
    if (Test-Path $placeholder) {
        Write-Host "- $placeholder"
    }
}

Write-Host ""
Write-Host "Committing dashboard images..." -ForegroundColor Green
git commit -m "Add dashboard screenshots"

Write-Host ""
Write-Host "To push these changes to GitHub, run:" -ForegroundColor Cyan
Write-Host "git push origin main"
Write-Host ""
Write-Host "Remember that large files might take time to upload depending on your connection speed." -ForegroundColor Yellow
Write-Host "If you have issues with file size limits, consider using Git LFS for large image files." -ForegroundColor Yellow 