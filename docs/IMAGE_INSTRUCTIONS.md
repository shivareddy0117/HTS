# Adding Dashboard Images to Your HTS Repository

These instructions will guide you through adding your dashboard images to your HTS project and displaying them correctly on GitHub.

## Step 1: Place Your Dashboard Screenshots in the Right Location

1. Ensure you have your three dashboard screenshots (1800 x 900 pixels):
   - Main dashboard view
   - Analytics panel view
   - Control interface view

2. Rename these files to:
   - `dashboard_main.png`
   - `dashboard_analytics.png`
   - `dashboard_controls.png`

3. Place these files in the `docs/images/` directory of your project

## Step 2: Generate Placeholder Images for the Other Diagrams

For the diagram images that you don't have yet, run the placeholder generation script:

```powershell
# Navigate to your project directory
cd C:\Users\SAHITHYAMOGILI\Desktop\Projects\Cursor_Projects\HTS

# Install Pillow if you don't have it
pip install Pillow

# Run the script to generate placeholder images
python scripts/generate_placeholder_images.py
```

This will create basic placeholder images for:
- `system_overview.png`
- `architecture.png`
- `cv_comparison.png`
- `nfq_architecture.png`

## Step 3: Commit and Push to GitHub

Run the PowerShell script to commit your images:

```powershell
# Navigate to your project directory
cd C:\Users\SAHITHYAMOGILI\Desktop\Projects\Cursor_Projects\HTS

# Run the commit script
pwsh scripts/commit_images.ps1
```

If you've already initialized your Git repository and want to push your changes:

```powershell
git push origin main
```

## Troubleshooting Large Image Files

If GitHub gives you errors about large file sizes:

1. Consider compressing your images with an online tool to reduce file size
2. Or set up Git LFS (Large File Storage) by installing it from https://git-lfs.com/

   ```powershell
   # Install Git LFS
   git lfs install
   
   # Track PNG files
   git lfs track "*.png"
   
   # Make sure .gitattributes is committed
   git add .gitattributes
   
   # Then add and commit your images
   git add docs/images/*.png
   git commit -m "Add images with Git LFS"
   
   # Push to GitHub
   git push origin main
   ```

## Notes on Image Display in GitHub

- The README uses responsive image tags, so your large images will display nicely on GitHub
- The `max-width: 1200px` setting ensures the images won't be too large on wide screens
- If images look too large or too small when displayed, you can adjust the `max-width` value in the README's HTML tags 