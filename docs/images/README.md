# HTS Visualization Images

This directory contains visualization images for the HTS Tape Manufacturing Optimization System.

## Required Images

For the project documentation, you'll need to add the following images:

1. `system_overview.png` - Overview diagram of the complete system
2. `architecture.png` - Technical architecture diagram
3. `cv_comparison.png` - Comparison of original vs. optimized CV values
4. `nfq_architecture.png` - Neural Fitted Q-Iteration architecture
5. `dashboard.png` - Screenshot of the monitoring dashboard

## How to Generate Images

### Dashboard Screenshots
Run the system dashboard and take screenshots:
```bash
python run_dashboard.py --data data/sample_hts_data.csv
```

### Plot Images
The system automatically saves plots when you run optimization:
```bash
python run_system.py --data data/sample_hts_data.csv --optimize
```

### System Diagrams
You can create system diagrams using tools like:
- [Lucidchart](https://www.lucidchart.com)
- [draw.io](https://app.diagrams.net/)
- [Miro](https://miro.com)

## Usage in Documentation

Images are referenced in the README.md and documentation files with the following format:
```markdown
![Image Description](docs/images/image_name.png)
``` 