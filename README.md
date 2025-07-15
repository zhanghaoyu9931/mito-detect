# Deep Learning-based model for mitochondrial signal detection

A deep learning-based model for automatic detection of true mitochondrial signals from Suite2p output in calcium imaging data.

## Overview

Since many detected ROIs may represent noise-correlated pixels but not true calcium events, there is a high false-positive rate after the Suite2p analysis. To resolve this problem, we developed a deep learning based classifier model to distinguish genuine [Ca2+]mito events from artefact.

## Repository Structure

```
mito-detect/
├── mitoDetect.py          # Main detection script
├── model_weights.pth      # Pre-trained model weights
├── requirements.txt       # Python dependencies
├── run_demo.sh           # Demo execution script
├── demo/                 # Demo data directory
│   ├── Fall.mat          # Demo input data
│   ├── Label.mat         # Demo output results
│   └── checked_figs/     # Demo visualization plots
├── LICENSE               # License file
└── README.md            # This file
```

## Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)

### Python Dependencies
The project includes a `requirements.txt` file with all necessary dependencies. Key packages include:
- torch==2.0.1
- torchvision==0.15.2
- numpy==1.24.4
- scipy==1.10.1
- scikit-learn==1.3.2
- matplotlib==3.7.5
- tqdm==4.67.1

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/mito-detect.git
cd mito-detect
```

2. Install required packages:
```bash
conda create -n mitoDet python=3.8
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Your input data should be a `Fall.mat` file containing a matrix named `F` with shape `(n_rois, n_timepoints)`, where:
- `n_rois`: Number of regions of interest (ROIs)
- `n_timepoints`: Number of time points in the calcium imaging recording

The data should be the output from Suite2p processing.

### 2. Run Detection

Basic usage:
```bash
python mitoDetect.py --data_dir /path/to/your/datadir
```

With visualization control:
```bash
python mitoDetect.py --data_dir /path/to/your/datadir --visual_num 10
```

**Parameters:**
- `--data_dir`: Path to directory containing `Fall.mat` file (required)
- `--visual_num`: Number of sample plots to generate for detected mitochondrial signals (default: 3)

### 3. Demo Example

To test the system with provided demo data:

```bash
# Run the demo
bash run_demo.sh
```

Or manually:
```bash
python mitoDetect.py --data_dir ./demo --visual_num 5
```

**Demo Output:**
The demo will generate:
- `demo/Label.mat`: Prediction results
- `demo/checked_figs/`: Sample plots of detected mitochondrial signals

## Input/Output Specifications

### Input Format
- **File format**: MATLAB `.mat` file
- **Required variable**: `F` (matrix)
- **Data type**: Float32 or Float64
- **Dimensions**: `(n_rois, n_timepoints)`
- **Expected range**: Raw fluorescence values from Suite2p

### Output Format
- **File**: `Label.mat` in the input directory
- **Variable**: `predicted_label` (array)
- **Values**: 
  - `0`: Non-mitochondrial signal (noise/artifact)
  - `2`: True mitochondrial signal
- **Additional outputs**:
  - `checked_figs/`: Directory containing sample plots of detected mitochondrial signals
  - Console output with prediction statistics

### Processing Pipeline
1. **Preprocessing**: Gaussian filtering and signal normalization
2. **Feature Extraction**: CNN layers extract spatial features
3. **Temporal Modeling**: LSTM layers capture temporal dependencies
4. **Classification**: Final dense layers output binary predictions
5. **Visualization**: Sample plots of detected mitochondrial signals

## Model Architecture

The model consists of:
- **CNN Feature Extractor**: 3 convolutional layers with batch normalization and ReLU activation
- **LSTM Temporal Model**: 3-layer LSTM with dropout for sequence modeling
- **Classification Head**: Dense layers with dropout for final prediction


## Example Output

When running the demo, you'll see output similar to:
```
Using device: cuda
File dir: ./demo
The first 10 predictions: [0, 2, 2, 0, 0, 0, 0, 0, 0, 0]
The total number of signals: 300 The number of mitochondrial signals: 94
```

This indicates:
- 300 total signals processed
- 94 signals classified as mitochondrial
- Sample plots saved in `checked_figs/` directory

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{mitodetect2025,
  title={In vivo imaging of neuronal mitochondrial Ca2+ transients with two-photon microscopy in awake mice},
  author={Shan Qiu, Haoyu Zhang ..., Xianhua Wang},
  journal={Mitochondrial Communications},
  year={2025},
  volume={X},
  pages={XXX--XXX},
  doi={10.XXXX/XXXXX}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please contact: [hauserzhang@pku.edu.cn] or [qiushan@stu.pku.edu.cn]

## Acknowledgments

We thank the Suite2p developers for their excellent calcium imaging processing pipeline.
