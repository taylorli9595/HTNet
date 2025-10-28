
# Two-Stream Multiplicative Heavy-Tail Noise Despeckling Network With Truncation Loss (HTNet)

This repository provides an unofficial implementation of the paper:

**Two-Stream Multiplicative Heavy-Tail Noise Despeckling Network With Truncation Loss**  
By Cheng, Li; Guo, Zhichang; Li, Yao; Xing, Yuming.  
Published in *IEEE Transactions on Geoscience and Remote Sensing, 2023*.  
DOI: [10.1109/TGRS.2023.3302953](https://doi.org/10.1109/TGRS.2023.3302953)

### 🚀 Highlights
- **Two-Stream Network**: Handles structural and statistical aspects of heavy-tail noise.
- **Truncation Loss**: Eliminates extreme residuals for improved performance in noisy SAR images.
- **SAR Speckle Removal**: Optimized for despeckling synthetic aperture radar (SAR) images with heavy-tail noise.

### 📁 Project Structure

```
├── datasets/          # Dataset preparation scripts and sample data loaders
├── networks/          # Network architectures (two-stream CNN, truncation loss modules)
├── datasets.py        # Dataset definition and preprocessing (e.g., patch extraction, normalization)
├── loss.py            # Implementation of truncation loss and auxiliary loss functions
├── train_two.py       # Main training script for the two-stream HTNet model
├── test_CL.py         # Evaluation and visualization script (CL = contrastive or combined loss)
├── utils.py           # Utility functions (logging, metrics, checkpoint management)
└── README.md          # This file
```

### 🔧 Requirements

```bash
Python >= 3.8
PyTorch >= 1.10
torchvision
numpy
scipy
matplotlib
```

### 🧠 Training

```bash
python train_two.py --data_dir ./datasets/SAR --epochs 100 --batch_size 16
```

### 🔍 Testing

```bash
python test_CL.py --model_path ./checkpoints/htnet_best.pth --input ./test_images/
```

### 💾 Pretrained Weights

The pretrained model weights are available via Baidu Cloud:

> **Link:** [https://pan.baidu.com/s/1GFS7g0AmDCs83umVM2gZgA](https://pan.baidu.com/s/1GFS7g0AmDCs83umVM2gZgA)  
> **Extraction Code:** `q1h4`

Download the file `weight.zip` and extract it into the `./checkpoints/` directory before running the testing script.

```bash
unzip weight.zip -d ./checkpoints/
python test_CL.py --model_path ./checkpoints/htnet_best.pth
```

### 📚 Citation

If you use this repository or find our work helpful in your research, please consider citing the following paper:

```bibtex
@ARTICLE{10210392,
  author={Cheng, Li and Guo, Zhichang and Li, Yao and Xing, Yuming},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Two-Stream Multiplicative Heavy-Tail Noise Despeckling Network With Truncation Loss}, 
  year={2023},
  volume={61},
  number={},
  pages={1-17},
  keywords={Speckle; Heavily-tailed distribution; Radar polarimetry; Task analysis; Feature extraction; Estimation; Noise reduction; Hybrid truncation loss to eliminate multiplicative noise (HTNet); multiplicative denoising; synthetic aperture radar (SAR); texture image; truncation loss},
  doi={10.1109/TGRS.2023.3302953}
}
```

