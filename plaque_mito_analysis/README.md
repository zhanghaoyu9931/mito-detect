# Plaque-Mitochondria Analysis Module

## Overview

This module performs spatial correlation analysis between amyloid plaques and mitochondrial calcium events in 3D brain tissue imaging data. It analyzes the spatial relationships between plaque locations and mitochondrial events to understand their functional coupling and spatial distribution patterns.

## Functionality

### Core Analysis Features

1. **Plaque Segmentation and Analysis**
   - 3D segmentation of amyloid plaques from TIFF image stacks
   - Volume-based filtering to remove small artifacts
   - Extraction of 3D centroid coordinates and volumes

2. **Mitochondrial Event Processing**
   - Extraction of all mitochondrial ROI centroids
   - Identification of calcium signaling events
   - Detection of coupling events between mitochondria and calcium signals

3. **Spatial Distance Analysis**
   - Calculation of nearest plaque distances for each mitochondrial event
   - Weighted distance analysis (distance/volume ratio)
   - Classification of events as "close" or "distal" based on median distance

4. **Statistical Analysis**
   - Coupling event distribution analysis in close vs distal regions
   - Correlation analysis between coupling fidelity and plaque distance
   - Multi-slice data aggregation and visualization

## File Structure

```
plaque_mito_analysis/
├── main.ipynb                 # Main analysis notebook
├── plaque.tif                 # 3D plaque channel TIFF file
├── mito_events/               # Mitochondrial events data directory
│   ├── mito_stat.mat          # Mitochondrial event statistics
│   ├── has_coupling.mat       # Coupling event indicators
│   ├── fidelity.mat           # Coupling fidelity values
│   ├── Label.mat              # True calcium event labels
│   ├── couple_centroids.mat   # Coupled event centroids (output)
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

3. **has_coupling.mat**: Coupling event indicators
   - Boolean array indicating which events show coupling (true=1)

4. **fidelity.mat**: Coupling fidelity values
   - Numerical values for coupling fidelity of true events

5. **Label.mat**: Calcium event classification
   - Predicted labels for calcium signaling events

6. **Z-axis.txt**: Z-coordinate information
   - Text file containing relative Z position in tissue block

## Usage

### Running the Analysis

1. **Prepare Data**: Ensure all required files are in the correct locations, Demo input files are available at [http://].
2. **Execute Notebook**: Run `main.ipynb` cell by cell.
3. **Review Results**: Check outputs for distance calculations and visualizations.
