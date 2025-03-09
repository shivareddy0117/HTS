# Dashboard Screenshot Guide

## How to Take Effective Dashboard Screenshots

For the HTS Tape Manufacturing Optimization System documentation, we need three distinct screenshots of the dashboard. Follow these steps to capture high-quality images:

## 1. Main Dashboard View (`dashboard_main.png`)

**What to capture:**
- The entire dashboard with all major components visible
- System status indicators
- Key performance metrics shown prominently

**How to capture:**
```bash
# Run the dashboard
python run_dashboard.py --data data/sample_hts_data.csv
```
- Wait for the dashboard to fully load
- Use your OS screenshot tool (Windows: Win+Shift+S, Mac: Cmd+Shift+4)
- Save as `dashboard_main.png` in the `docs/images/` directory
- **Image size**: 1800 x 900 pixels

## 2. Analytics Panel View (`dashboard_analytics.png`)

**What to capture:**
- Focus on the graphs and visualization components
- Make sure critical current plots are clearly visible
- Include the optimization progress visualization

**How to capture:**
- Scroll to the analytics section of the dashboard
- Capture just this section using your screenshot tool
- Save as `dashboard_analytics.png` in the `docs/images/` directory
- **Image size**: 1800 x 900 pixels

## 3. Control Interface View (`dashboard_controls.png`)

**What to capture:**
- Parameter adjustment controls
- Manufacturing process settings sliders/inputs
- Action buttons for optimization

**How to capture:**
- Scroll to the control panel section of the dashboard
- Capture just this section using your screenshot tool
- Save as `dashboard_controls.png` in the `docs/images/` directory
- **Image size**: 1800 x 900 pixels

## Tips for Good Screenshots

1. **Resolution**: Take screenshots at high resolution (at least 1200px wide)
2. **Clarity**: Make sure text is readable and not blurry
3. **Content**: Ensure all important UI elements are visible
4. **Clutter**: Close unnecessary browser tabs/windows before capturing
5. **Data**: Use sample data that shows interesting patterns or results

After taking your screenshots, replace the placeholder files in `docs/images/` and commit them to your repository. 