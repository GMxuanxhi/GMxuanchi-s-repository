# GMxuanchi-s-repository
This repository contains implementations of federated learning algorithms applied to CIFAR-10 image classification under non-IID data distributions.

PREREQUISITES
- Python 3.7+
- PyTorch, torchvision, pandas, numpy, matplotlib, scikit-learn, PIL
- GPU with CUDA support (recommended: NVIDIA RTX 4090)

EXPERIMENT REPRODUCTION STEPS

1. GENERATE CIFAR-10 DATASET
   Run: python 01_CIFAR10_generate.py
   This downloads CIFAR-10 and saves images in dataset/CIFAR10/train and dataset/CIFAR10/test.

2. CREATE LABEL CSV FILES
   Run: python 02_label.py
   Generates CIFAR10Train.csv and CIFAR10Test.csv with image paths and labels.

3. SPLIT DATA INTO NON-IID CLIENTS
   Edit alpha value in 03_split_data.py (line 47) to desired Dirichlet parameter (0.1, 0.3, 1, or 3)
   Run: python 03_split_data.py
   Creates distribution folder (e.g., Distribution_CIFAR_0.1) with client datasets and test set.

4. VISUALIZE LABEL DISTRIBUTION
   Run: python 04_visualize_label.py
   Generates stacked bar chart showing label distribution across clients.

5. RUN FEDERATED LEARNING EXPERIMENTS

   A. FedAvg (Standard Federated Averaging)
   Edit in 05_FedAvg_normal.py:
   - Line 24: folder_name = 'FedAvg_CIFAR_0.1_1' (update alpha and run number)
   - Line 31: data_folder_path = "Distribution_CIFAR_0.1" (match distribution folder)
   Run: python 05_FedAvg_normal.py

   B. FedAvg+CS (with Compressed Sensing)
   Edit in 07_FedAvg_CS.py:
   - Line 24: folder_name = 'FedAvg_CS_0.5_1.5_CIFAR_0.1_1' (format: sparsity_ratio_alpha_run)
   - Line 31: data_folder_path = "Distribution_CIFAR_0.1"
   - Line 299: Modify server_sparsity and aggregation_ratio parameters
   Run: python 07_FedAvg_CS.py

   C. FedProx+CS (with Proximal Term and Compression)
   Edit in 08_FedProx_CS.py:
   - Lines 13-16: Set MU, SP, AR, AL hyperparameters
   Run: python 08_FedProx_CS.py

6. PROCESS MULTIPLE RUN RESULTS
   Edit in 09_data_processing.py:
   - Line 16: folders list with run directories
   - Line 18: result_dir with output directory name
   Run: python 09_data_processing.py
   Averages results from 5 runs for each configuration.

7. PLOT COMPARISON GRAPHS
   Edit in 10_draw.py:
   - Line 8: result_folders list with result directories to compare
   Run: python 10_draw.py
   Generates accuracy comparison plots in result folder.

EXPERIMENTAL CONFIGURATIONS
- Distributions: Dirichlet alpha = 0.1, 0.3, 1, 3
- Sparsity levels: 0.6, 0.5, 0.4, 0.3
- Aggregation ratios: 3, 5, 7, 10
- FedProx μ values: 0.01, 0.005, 0.001, 0.0005
- Each configuration should be run 5 times for statistical significance

NOTES
- The code automatically uses GPU if available
- Folder names must be consistent across scripts
