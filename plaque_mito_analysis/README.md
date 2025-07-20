# Plaque-Mitochondria Analysis Module

## Overview

This module performs spatial correlation analysis between amyloid plaques and mitochondrial calcium events in 3D brain tissue imaging data. It analyzes the spatial relationships between plaque locations and mitochondrial events to understand their spatial distribution patterns and proximity effects.

## File Structure

```
plaque_mito_analysis/
├── main.ipynb                 # Main analysis notebook
├── plaque.tif                 # 3D plaque channel TIFF file
├── mito_events/               # Mitochondrial events data directory
│   ├── mito_stat.mat          # Mitochondrial event statistics
│   ├── Label.mat              # Calcium event labels
│   ├── calcase_centroids.mat  # Calcium event centroids (output)
│   └── Z-axis.txt             # Z-axis position information
└── README.md                  # This file
```

## Data Format Requirements

### Input Files
1. **plaque.tif**: 3D TIFF stack of plaque channel
   - Each frame represents 1μm physical spacing
   - 3D volume data for plaque segmentation

2. **mito_stat.mat**: Mitochondrial event statistics
   - Structure array with ROI pixel coordinates (x, y)
   - Each ID corresponds to a mitochondrial calcium event

3. **Label.mat**: Calcium event classification
   - Predicted labels for calcium signaling events

4. **Z-axis.txt**: Z-coordinate information
   - Text file containing relative Z position in tissue block

## Usage

### Running the Analysis

1. **Prepare Data**: Ensure all required files are in the correct locations, Demo input files are available at [https://disk.pku.edu.cn/link/AA50645218EDC04252B9905A8CF8BB4CFF].
2. **Execute Notebook**: Run `main.ipynb` cell by cell.
3. **Review Results**: Check outputs for distance calculations and visualizations.

### Analysis Workflow

1. **Plaque Segmentation** (Cell 1)
   - Load and segment 3D plaque data
   - Apply morphological operations
   - Filter by volume threshold (≥50 voxels)
   - Extract centroid coordinates and volumes

2. **Mitochondrial Event Processing** (Cell 2)
   - Extract all mitochondrial ROI centroids
   - Identify calcium events
   - Calculate 2D centroids for each event type
   - Save processed data to MAT files

3. **3D Visualization** (Cell 3)
   - Create 3D scatter plots showing:
     - Plaque locations (blue spheres, size proportional to volume)
     - Calcium events (grey points)

4. **Distance Analysis** (Cell 4)
   - Calculate nearest plaque distances for each event
   - Classify events as close or distal based on median distance
   - Generate distance distribution statistics

## Output Analysis

### Key Metrics

1. **Distance Measurements**
   - Nearest plaque distance for each mitochondrial event
   - Distance distribution statistics (mean, median, range)

2. **Spatial Classification**
   - Events classified as "close" or "distal" based on median distance
   - Distribution analysis of events in each category

3. **Visualization**
   - 3D scatter plots showing spatial relationships
   - Distance distribution histograms
   - Summary statistics plots

## Dependencies

The same as in `mito-detect`.