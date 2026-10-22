# Explainable CGCNN for Formation Energy Prediction

An explainable crystal graph convolutional neural network (CGCNN) for predicting materials formation energy directly from crystallographic information files (`.cif`). The project combines a PyTorch CGCNN pipeline with an interactive Streamlit application that visualizes crystal structures and highlights atoms and bonds influential to individual predictions.

## Overview

Crystal structures are naturally represented as graphs: atoms form the nodes, and spatially neighboring atoms form the edges. This project converts CIF structures into crystal graphs, trains a CGCNN to predict formation energy, and adds instance-level interpretation to make model predictions easier to inspect.

## Key Features

- **CIF-to-graph processing:** Uses `pymatgen` to read crystal structures. Element-specific atom embeddings are used as node features, and interatomic distances are expanded into Gaussian basis features for neighboring edges.
- **Large-scale training data:** Includes a notebook workflow for preparing a 130,000-structure subset of Matbench `mp_e_form`.
- **Configurable CGCNN training:** A PyTorch implementation with configurable model dimensions, Adam or SGD optimization, multistep learning-rate decay, and weight decay (L2 regularization). The best checkpoint is selected using validation MAE.
- **Instance-level explanations:** The dashboard optimizes node and edge masks in a GNNExplainer-style workflow to show which atoms and interactions most influence an individual prediction.
- **Interactive 3D visualization:** Explore the uploaded crystal structure and its explanation in a Streamlit dashboard built with Plotly.
- **Training analysis:** The Jupyter notebook orchestrates dataset generation and model training, then plots learning curves and evaluation metrics.

## Repository Structure

```text
.
├── cgcnn/
│   ├── data.py               # CIF dataset, graph construction, and batching
│   ├── model.py              # CGCNN layers and crystal-level prediction model
│   └── __init__.py
├── main.py                   # Training and validation entry point
├── train.ipynb               # Dataset generation, training, and learning curves
├── gnn_streamlit.py          # Explainability and interactive 3D dashboard
├── cgcnn_dataset/            # Generated CIF splits and atom feature metadata
├── results*/                 # Local training outputs (may not be tracked)
└── README.md
```

Large raw datasets, generated CIF files, training outputs, and model checkpoints may not be included in a Git checkout. Generate or obtain the required data and checkpoint locally before running the corresponding workflow.

## Dataset Information

The training workflow uses the **Matbench `mp_e_form`** task, which contains crystal structures and formation-energy targets. The notebook samples 130,000 structures and creates the following splits:

| Split | Structures |
| --- | ---: |
| Training | 80,000 |
| Validation | 10,000 |
| Test | 10,000 |
| Additional `test2` set | 30,000 |

Each generated split is expected to contain CIF files, an `id_prop.csv` mapping structure IDs to target values, and an `atom_init.json` file containing elemental feature vectors. The data loader builds a local neighbor graph for each structure (up to 12 neighbors within an 8 Å radius by default) and represents distances using Gaussian basis functions.

The dataset-generation notebook reads a local Matbench JSON file. Because the source dataset and generated structures are large, they may need to be downloaded or generated separately and are not necessarily tracked in this repository.

## Model Performance

The project reports formation-energy prediction performance of approximately **0.077 eV/atom MAE** and **0.99 R²**.

For context, the saved output in `train.ipynb` records a final-epoch **validation** MAE of **0.0772 eV/atom** and **R² of 0.9904**. These notebook values are validation metrics, not an independently reported test-set result. Use the held-out test split for a test estimate; results can vary with the split, checkpoint, and training configuration.

## Installation

Use Python 3.9 or later. A CUDA-enabled PyTorch installation is recommended for faster training, but the code can also run on CPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch numpy scikit-learn pymatgen pandas jupyter streamlit plotly
```

For GPU support, install the PyTorch build that matches your CUDA environment using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Usage

### 1. Prepare the dataset

Open `train.ipynb` in Jupyter and run the dataset-generation cells after placing the Matbench `mp_e_form` JSON file at the path configured in the notebook. The notebook writes train, validation, test, and `test2` CIF directories under `cgcnn_dataset/`.

Each training and validation directory must follow this layout:

```text
train_data/
├── atom_init.json
├── id_prop.csv
├── mat_<id>.cif
└── ...
```

The validation and test directories use the same structure. Ensure the generated directories exist before starting training.

### 2. Train the model

From the repository root, run:

```powershell
python main.py cgcnn_dataset --epochs 20 --batch-size 256 --optim Adam --lr 0.01 --lr-milestones 10 15 --weight-decay 0.00001
```

`main.py` reads `cgcnn_dataset/train_data/` and `cgcnn_dataset/val_data/`, trains the model, and saves the best validation checkpoint. Run `python main.py --help` to see all supported options. The included notebook also launches the training script as a subprocess and plots its output.

### 3. Launch the explainability dashboard

Before launching, configure `PROJECT_DIR`, `CHECKPOINT_PATH`, and `ATOM_INIT_PATH` near the top of `gnn_streamlit.py` to point to your local checkout, trained checkpoint, and atom initialization file. The dashboard currently expects a checkpoint compatible with the CGCNN architecture.

```powershell
streamlit run gnn_streamlit.py
```

Open the local URL printed by Streamlit, upload a CIF file, and inspect its predicted formation energy and 3D explanation. The dashboard also includes an optional batch validation action for available `test2` CIF files.

## Acknowledgements

This project builds on the CGCNN approach for learning material representations from crystal structures and uses the Matbench `mp_e_form` benchmark. See the original [CGCNN repository](https://github.com/txie-93/cgcnn) and [Matbench project](https://matbench.materialsproject.org/) for background and dataset details.
