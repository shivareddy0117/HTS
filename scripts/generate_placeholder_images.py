#!/usr/bin/env python
"""
Generate placeholder PNG images for HTS documentation.
This script creates simple PNG images with text labels for use in documentation
until real diagrams and screenshots are available.
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont
import textwrap

# Check if Pillow is installed
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Pillow library is not installed. Please install it using:")
    print("pip install Pillow")
    sys.exit(1)

def create_placeholder_image(filename, title, width=1200, height=800, bg_color=(240, 240, 255)):
    """
    Create a placeholder image with the given title and dimensions
    
    Args:
        filename: Output filename (path to save the image)
        title: Title text to display on the image
        width: Image width in pixels
        height: Image height in pixels
        bg_color: Background color as RGB tuple
    """
    # Create image with background color
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw borders
    border_color = (180, 180, 200)
    border_width = 10
    draw.rectangle([0, 0, width-1, height-1], outline=border_color, width=border_width)
    
    # Try to use a nice font, but fall back to default if not available
    try:
        # Try to find a system font that might be available
        system_fonts = [
            'Arial.ttf', 'DejaVuSans.ttf', 'Verdana.ttf', 
            'FreeSans.ttf', 'Tahoma.ttf', 'LiberationSans-Regular.ttf'
        ]
        font = None
        for font_name in system_fonts:
            try:
                font = ImageFont.truetype(font_name, 48)
                break
            except IOError:
                continue
                
        if font is None:
            font = ImageFont.load_default()
            title_font_size = 20
        else:
            title_font_size = 48
            
    except Exception as e:
        print(f"Font error: {e}")
        font = ImageFont.load_default()
        title_font_size = 20
    
    # Draw the title at the top
    title_color = (50, 50, 150)
    draw.text((width//2, 60), title, fill=title_color, font=font, anchor="mm")
    
    # Draw HTS placeholder text
    placeholder_text = "HTS Tape Manufacturing Optimization System"
    draw.text((width//2, height//2 - 50), placeholder_text, 
              fill=(100, 100, 180), font=font, anchor="mm")
    
    # Draw "PLACEHOLDER IMAGE" text
    placeholder_text = "PLACEHOLDER IMAGE"
    draw.text((width//2, height//2 + 50), placeholder_text, 
              fill=(180, 0, 0), font=font, anchor="mm")
    
    # Add instruction text
    instruction_text = "Replace with actual image before deployment"
    try:
        small_font = ImageFont.truetype(font._family, 32)
    except:
        small_font = font
        
    draw.text((width//2, height - 80), instruction_text, 
              fill=(100, 100, 100), font=small_font, anchor="mm")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the image
    img.save(filename, 'PNG')
    print(f"Created placeholder image: {filename}")

def main():
    # Define the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(base_dir, "docs", "images")
    
    # Ensure the images directory exists
    os.makedirs(images_dir, exist_ok=True)
    
    # Define the images to create
    placeholders = [
        {"filename": "system_overview.png", "title": "System Overview", "width": 1200, "height": 800},
        {"filename": "architecture.png", "title": "Technical Architecture", "width": 1200, "height": 800},
        {"filename": "cv_comparison.png", "title": "Critical Current Optimization", "width": 1000, "height": 600},
        {"filename": "nfq_architecture.png", "title": "Neural Fitted Q-Iteration Architecture", "width": 1000, "height": 700},
        {"filename": "dashboard_main.png", "title": "Dashboard - Main View", "width": 1800, "height": 900},
        {"filename": "dashboard_analytics.png", "title": "Dashboard - Analytics Panel", "width": 1800, "height": 900},
        {"filename": "dashboard_controls.png", "title": "Dashboard - Control Interface", "width": 1800, "height": 900},
    ]
    
    # Create each placeholder image
    for p in placeholders:
        filepath = os.path.join(images_dir, p["filename"])
        create_placeholder_image(
            filepath, 
            p["title"],
            p.get("width", 1200),
            p.get("height", 800)
        )
    
    print("\nDone! Created placeholder images in:", images_dir)
    print("\nReplace these with actual images when available.")
    print("See docs/dashboard_screenshot_guide.md for instructions.")

if __name__ == "__main__":
    main() 