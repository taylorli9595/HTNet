# HTNet
#code for Two-Stream Multiplicative Heavy-Tail Noise Despeckling Network With Truncation Loss[https://ieeexplore.ieee.org/document/10210392]

The pretrained model weights are available via Baidu Cloud:

Link: https://pan.baidu.com/s/1GFS7g0AmDCs83umVM2gZgA

Extraction Code: q1h4

Download the file weight.zip and extract it into the ./checkpoints/ directory before running the testing script.
unzip weight.zip -d ./checkpoints/
python test_CL.py --model_path ./checkpoints/htnet_best.pth


├── datasets/          # Dataset preparation scripts and sample data loaders
├── networks/          # Network architectures (two-stream CNN, truncation loss modules)
├── datasets.py        # Dataset definition and preprocessing (e.g., patch extraction, normalization)
├── loss.py            # Implementation of truncation loss and auxiliary loss functions
├── train_two.py       # Main training script for the two-stream HTNet model
├── test_CL.py         # Evaluation and visualization script (CL = contrastive or combined loss)
├── utils.py           # Utility functions (logging, metrics, checkpoint management)
└── README.md          # Documentation
⚙️ Usage
1️⃣ Training
bash
复制代码
python train_two.py --data_dir ./datasets/SAR --epochs 100 --batch_size 16
2️⃣ Testing
bash
复制代码
python test_CL.py --model_path ./checkpoints/htnet_best.pth --input ./test_images/

If you use this code or find our work helpful, please cite the following paper:
@ARTICLE{10210392,
  author={Cheng, Li and Guo, Zhichang and Li, Yao and Xing, Yuming},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Two-Stream Multiplicative Heavy-Tail Noise Despeckling Network With Truncation Loss}, 
  year={2023},
  volume={61},
  number={},
  pages={1-17},
  keywords={Speckle;Heavily-tailed distribution;Radar polarimetry;Task analysis;Feature extraction;Estimation;Noise reduction;Hybrid truncation loss to eliminate multiplicative noise (HTNet);multiplicative denoising;synthetic aperture radar (SAR);texture image;truncation loss},
  doi={10.1109/TGRS.2023.3302953}}

