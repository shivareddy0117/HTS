#!/bin/bash
# Script to properly commit large image files to Git

echo "=== HTS Image Commit Tool ==="
echo "This script will help you add and commit your dashboard images to Git"
echo ""

# Make sure we're in the project root
cd "$(dirname "$0")/.." || exit 1

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Please install Git first."
    exit 1
fi

# Check if Git repo exists
if [ ! -d .git ]; then
    echo "Initializing Git repository..."
    git init
fi

echo "Adding dashboard images..."
git add docs/images/dashboard_*.png

# Check if there are other placeholder images to replace
PLACEHOLDERS=(
    "docs/images/system_overview.png" 
    "docs/images/architecture.png"
    "docs/images/cv_comparison.png"
    "docs/images/nfq_architecture.png"
)

echo ""
echo "The following placeholder images still need to be replaced:"
for placeholder in "${PLACEHOLDERS[@]}"; do
    if [ -f "$placeholder" ]; then
        echo "- $placeholder"
    fi
done

echo ""
echo "Committing dashboard images..."
git commit -m "Add dashboard screenshots"

echo ""
echo "To push these changes to GitHub, run:"
echo "git push origin main"
echo ""
echo "Remember that large files might take time to upload depending on your connection speed."
echo "If you have issues with file size limits, consider using Git LFS for large image files." 