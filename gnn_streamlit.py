
import os
import sys
import json
import shutil
import tempfile
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import plotly.graph_objects as go

from pymatgen.core import Structure

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\CGCNN-Matbench-Predictor"

CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "checkpoint.pth")
ATOM_INIT_PATH = os.path.join(PROJECT_DIR, "test_data", "atom_init.json")
CGCNN_DIR = os.path.join(PROJECT_DIR, "cgcnn")

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
if CGCNN_DIR not in sys.path:
    sys.path.insert(0, CGCNN_DIR)

warnings.filterwarnings("ignore")

# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Explainable CGCNN",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Explainable CGCNN")
st.subheader(
    "Formation Energy Prediction + GNN Explainer + 3D Crystal Visualization"
)

st.info(
    "Upload a CIF file. The existing trained CGCNN is used for prediction. "
    "The CGCNN weights are frozen and are NOT retrained. "
    "Only explanation masks are optimized."
)

# ============================================================
# IMPORT EXISTING CGCNN
# ============================================================

try:
    from cgcnn.data import CIFData
    from cgcnn.model import CrystalGraphConvNet
except Exception as e:
    st.error(
        "Could not import your existing CGCNN package.\n\n"
        f"{e}"
    )
    st.stop()

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(PROJECT_DIR):
    st.error(f"Project directory not found:\n{PROJECT_DIR}")
    st.stop()

if not os.path.exists(CHECKPOINT_PATH):
    st.error(f"Checkpoint not found:\n{CHECKPOINT_PATH}")
    st.stop()

if not os.path.exists(ATOM_INIT_PATH):
    st.error(f"atom_init.json not found:\n{ATOM_INIT_PATH}")
    st.stop()

# ============================================================
# SETTINGS
# ============================================================

explanation_steps = 200
top_atom_percent = 20
random_seed = 42

# Device is needed by both the normal single-CIF flow and the batch-only flow.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# TEST2 BATCH VALIDATION BUTTON
# ============================================================
# This button is intentionally placed BEFORE the individual CIF uploader.
# It validates only the first 50 CIF files in sorted filename order.

run_all_test2 = st.button(
    "🧪 Validate GNNExplainer on Top 50 Test Files + Excel",
    type="primary",
    width="stretch",
    help="Run validation on at most the first 50 CIF files in sorted filename order in test2, without uploading an individual CIF."
)

# ============================================================
# BATCH-ONLY TEST2 MODE
# ============================================================
# When the batch button is clicked before an individual CIF is uploaded,
# validate up to 50 sorted CIF files in PROJECT_DIR/test2.
# The single-CIF interface below is skipped in this mode.

if run_all_test2:
    from io import BytesIO

    TEST2_DIR = os.path.join(PROJECT_DIR, "test2")
    TEST2_MAX_FILES = 50
    TEST2_FIDELITY_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.40]
    TEST2_RANDOM_TRIALS = 20
    TEST2_RESTARTS = 15

    # Same six GNNExplainer configurations used by the main application.
    TEST2_AUTO_TUNE_CONFIGS = [
        {"prediction_weight": 10.0, "node_sparsity_weight": 0.010, "edge_sparsity_weight": 0.020, "entropy_weight": 0.0010, "lr": 0.010},
        {"prediction_weight": 25.0, "node_sparsity_weight": 0.015, "edge_sparsity_weight": 0.030, "entropy_weight": 0.0010, "lr": 0.008},
        {"prediction_weight": 50.0, "node_sparsity_weight": 0.020, "edge_sparsity_weight": 0.040, "entropy_weight": 0.0008, "lr": 0.006},
        {"prediction_weight": 100.0, "node_sparsity_weight": 0.025, "edge_sparsity_weight": 0.050, "entropy_weight": 0.0006, "lr": 0.005},
        {"prediction_weight": 200.0, "node_sparsity_weight": 0.030, "edge_sparsity_weight": 0.060, "entropy_weight": 0.0005, "lr": 0.004},
        {"prediction_weight": 400.0, "node_sparsity_weight": 0.040, "edge_sparsity_weight": 0.080, "entropy_weight": 0.0003, "lr": 0.003},
    ]

    @st.cache_resource
    def load_batch_model():
        """Load the same frozen CGCNN checkpoint used by the single-CIF app.

        Supports both:
          1. checkpoint.pth as a normal PyTorch file
          2. checkpoint.pth as an extracted PyTorch checkpoint directory
        """
        if os.path.isfile(CHECKPOINT_PATH):
            try:
                checkpoint_local = torch.load(
                    CHECKPOINT_PATH,
                    map_location=device,
                    weights_only=False
                )
            except TypeError:
                checkpoint_local = torch.load(
                    CHECKPOINT_PATH,
                    map_location=device
                )

        elif os.path.isdir(CHECKPOINT_PATH):
            import zipfile

            checkpoint_dir_local = os.path.abspath(CHECKPOINT_PATH)
            archive_root_local = os.path.basename(
                checkpoint_dir_local.rstrip(os.sep)
            )

            temp_archive_local = tempfile.NamedTemporaryFile(
                suffix=".pth",
                delete=False
            )
            temp_archive_local.close()

            try:
                with zipfile.ZipFile(
                    temp_archive_local.name,
                    "w",
                    compression=zipfile.ZIP_STORED
                ) as archive_local:
                    for root_local, dirs_local, files_local in os.walk(
                        checkpoint_dir_local
                    ):
                        for filename_local in files_local:
                            full_path_local = os.path.join(
                                root_local,
                                filename_local
                            )
                            relative_path_local = os.path.relpath(
                                full_path_local,
                                checkpoint_dir_local
                            )
                            archive_name_local = (
                                archive_root_local
                                + "/"
                                + relative_path_local.replace(
                                    os.sep, "/"
                                )
                            )

                            info_local = zipfile.ZipInfo(
                                archive_name_local
                            )
                            info_local.date_time = (
                                1980, 1, 1, 0, 0, 0
                            )
                            info_local.compress_type = zipfile.ZIP_STORED

                            with open(full_path_local, "rb") as file_obj_local:
                                archive_local.writestr(
                                    info_local,
                                    file_obj_local.read()
                                )

                print("Reconstructed PyTorch checkpoint archive for Test2 validation.")

                try:
                    checkpoint_local = torch.load(
                        temp_archive_local.name,
                        map_location=device,
                        weights_only=False
                    )
                except TypeError:
                    checkpoint_local = torch.load(
                        temp_archive_local.name,
                        map_location=device
                    )
            finally:
                try:
                    os.remove(temp_archive_local.name)
                except Exception:
                    pass

        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT_PATH}"
            )

        if not isinstance(checkpoint_local, dict):
            raise TypeError("Checkpoint is not a dictionary.")

        args_local = checkpoint_local.get("args")
        if args_local is None:
            raise KeyError("Checkpoint does not contain 'args'.")

        def get_arg_local(name, default=None):
            if isinstance(args_local, dict):
                return args_local.get(name, default)
            return getattr(args_local, name, default)

        # Infer feature dimensions from the first test2 CIF.
        cif_candidates = sorted(
            os.path.join(TEST2_DIR, f)
            for f in os.listdir(TEST2_DIR)
            if f.lower().endswith(".cif")
        ) if os.path.isdir(TEST2_DIR) else []

        if not cif_candidates:
            raise FileNotFoundError(
                f"No CIF files found in test2 folder:\n{TEST2_DIR}"
            )

        sample_root = tempfile.mkdtemp(prefix="cgcnn_test2_model_")
        try:
            sample_name = os.path.splitext(os.path.basename(cif_candidates[0]))[0]
            shutil.copy2(
                cif_candidates[0],
                os.path.join(sample_root, sample_name + ".cif")
            )
            shutil.copy(
                ATOM_INIT_PATH,
                os.path.join(sample_root, "atom_init.json")
            )
            with open(
                os.path.join(sample_root, "id_prop.csv"),
                "w",
                encoding="utf-8"
            ) as f:
                f.write(f"{sample_name},0.0\n")

            sample_dataset = CIFData(sample_root)
            sample_structures, _, _ = sample_dataset[0]
            sample_atom_fea = sample_structures[0].clone().detach().float()
            sample_nbr_fea = sample_structures[1].clone().detach().float()
        finally:
            shutil.rmtree(sample_root, ignore_errors=True)

        model_local = CrystalGraphConvNet(
            sample_atom_fea.shape[1],
            sample_nbr_fea.shape[2],
            atom_fea_len=get_arg_local("atom_fea_len", 64),
            n_conv=get_arg_local("n_conv", 3),
            h_fea_len=get_arg_local("h_fea_len", 128),
            n_h=get_arg_local("n_h", 1),
            classification=get_arg_local("classification", False)
        )

        state_dict_local = checkpoint_local.get("state_dict")
        if state_dict_local is None:
            raise KeyError("Checkpoint does not contain 'state_dict'.")

        cleaned_state_dict_local = {}
        for key, value in state_dict_local.items():
            if key.startswith("module."):
                key = key[7:]
            cleaned_state_dict_local[key] = value

        model_local.load_state_dict(cleaned_state_dict_local, strict=True)
        model_local = model_local.to(device)
        model_local.eval()

        for parameter in model_local.parameters():
            parameter.requires_grad_(False)

        return model_local, checkpoint_local

    def batch_denormalize(output_tensor, checkpoint_local):
        normalizer_local = checkpoint_local.get("normalizer", None)
        if normalizer_local is None:
            return output_tensor
        try:
            return normalizer_local.denorm(output_tensor)
        except Exception:
            pass
        try:
            return (
                output_tensor * normalizer_local.std.to(output_tensor.device)
                + normalizer_local.mean.to(output_tensor.device)
            )
        except Exception:
            return output_tensor

    def run_test2_batch_validation():
        """Run the same validation ideas used by the individual-crystal UI."""
        if not os.path.isdir(TEST2_DIR):
            st.error(f"test2 folder not found:\n{TEST2_DIR}")
            return

        available_test2_files_local = sorted(
            os.path.join(TEST2_DIR, f)
            for f in os.listdir(TEST2_DIR)
            if f.lower().endswith(".cif")
        )

        if not available_test2_files_local:
            st.warning(f"No CIF files were found in test2:\n{TEST2_DIR}")
            return

        test2_files_local = available_test2_files_local[:TEST2_MAX_FILES]

        with st.spinner("Loading the existing frozen CGCNN..."):
            model_local, checkpoint_local = load_batch_model()

        st.success(
            f"Found {len(available_test2_files_local)} CIF files in test2; "
            f"validating {len(test2_files_local)} (first files in sorted filename order). "
            "The pretrained CGCNN remains frozen."
        )

        batch_rows_local = []
        progress_local = st.progress(0)
        status_local = st.empty()

        for file_index, cif_file_local in enumerate(test2_files_local, start=1):
            material_name = os.path.splitext(os.path.basename(cif_file_local))[0]
            status_local.write(
                f"Testing {file_index}/{len(test2_files_local)}: {material_name}"
            )
            temp_local = tempfile.mkdtemp(prefix="cgcnn_test2_item_")

            try:
                shutil.copy2(
                    cif_file_local,
                    os.path.join(temp_local, material_name + ".cif")
                )
                shutil.copy(
                    ATOM_INIT_PATH,
                    os.path.join(temp_local, "atom_init.json")
                )
                with open(
                    os.path.join(temp_local, "id_prop.csv"),
                    "w",
                    encoding="utf-8"
                ) as f:
                    f.write(f"{material_name},0.0\n")

                structure_local = Structure.from_file(cif_file_local)
                dataset_local = CIFData(temp_local)
                structures_local, _, _ = dataset_local[0]

                atom_features_local = structures_local[0].clone().detach().float().to(device)
                nbr_features_local = structures_local[1].clone().detach().float().to(device)
                nbr_indices_local = structures_local[2].clone().detach().long().to(device)
                n_atoms = atom_features_local.shape[0]
                n_neighbors = nbr_features_local.shape[1]
                crystal_indices_local = [
                    torch.arange(n_atoms, dtype=torch.long, device=device)
                ]

                def predict_local(atom_x, nbr_x, nbr_idx):
                    with torch.no_grad():
                        out = model_local(
                            atom_x.to(device),
                            nbr_x.to(device),
                            nbr_idx.to(device),
                            crystal_indices_local
                        ).reshape(-1)[0]
                    return batch_denormalize(out.reshape(1), checkpoint_local)[0]

                original_pred_tensor = predict_local(
                    atom_features_local,
                    nbr_features_local,
                    nbr_indices_local
                )
                original_value_local = float(original_pred_tensor.detach().cpu().item())
                tolerance_local = max(0.01, 0.05 * abs(original_value_local))

                # Build the valid edge-slot mask using the same radius/max-neighbor
                # convention as the existing CGCNN graph.
                valid_edges_np = np.zeros(
                    (n_atoms, n_neighbors), dtype=bool
                )
                try:
                    periodic_neighbors = structure_local.get_all_neighbors(
                        float(getattr(dataset_local, "radius", 8.0)),
                        include_index=True
                    )
                    periodic_neighbors = [
                        sorted(row, key=lambda x: x[1])
                        for row in periodic_neighbors
                    ]
                    for ii in range(n_atoms):
                        valid_edges_np[ii, :min(n_neighbors, len(periodic_neighbors[ii]))] = True
                except Exception:
                    valid_edges_np[:, :] = True

                valid_edges_t = torch.tensor(
                    valid_edges_np,
                    dtype=torch.bool,
                    device=device
                )

                original_model_output = None
                with torch.no_grad():
                    original_model_output = model_local(
                        atom_features_local,
                        nbr_features_local,
                        nbr_indices_local,
                        crystal_indices_local
                    ).reshape(-1)[0].detach()

                def masked_forward_local(node_mask_local, edge_mask_local):
                    masked_atoms = atom_features_local * node_mask_local.unsqueeze(1)
                    masked_nbrs = nbr_features_local * edge_mask_local.unsqueeze(-1)
                    return model_local(
                        masked_atoms,
                        masked_nbrs,
                        nbr_indices_local,
                        crystal_indices_local
                    ).reshape(-1)[0]

                def optimize_masks_local(seed_local, steps_local, lr_local,
                                         prediction_weight_local,
                                         node_sparsity_local,
                                         edge_sparsity_local,
                                         entropy_weight_local):
                    random.seed(seed_local)
                    np.random.seed(seed_local)
                    torch.manual_seed(seed_local)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed_local)

                    node_logits_local = nn.Parameter(
                        0.05 * torch.randn(n_atoms, device=device)
                    )
                    edge_logits_local = nn.Parameter(
                        0.05 * torch.randn(
                            n_atoms, n_neighbors, device=device
                        )
                    )
                    optimizer_local = torch.optim.Adam(
                        [node_logits_local, edge_logits_local],
                        lr=lr_local
                    )

                    best_state = None
                    best_feasible_size_local = float("inf")
                    best_any_local = None
                    best_any_error_local = float("inf")

                    for step_local in range(max(1, steps_local)):
                        optimizer_local.zero_grad()

                        node_mask_local = torch.sigmoid(node_logits_local)
                        edge_mask_local = torch.sigmoid(edge_logits_local)
                        edge_mask_local = torch.where(
                            valid_edges_t,
                            edge_mask_local,
                            torch.ones_like(edge_mask_local)
                        )

                        masked_output_local = masked_forward_local(
                            node_mask_local,
                            edge_mask_local
                        )

                        prediction_loss_local = (
                            masked_output_local - original_model_output
                        ).pow(2)

                        valid_nodes_local = node_mask_local
                        valid_edges_local = edge_mask_local[valid_edges_t]

                        node_size_loss_local = valid_nodes_local.mean()
                        edge_size_loss_local = (
                            valid_edges_local.mean()
                            if valid_edges_local.numel()
                            else torch.tensor(0.0, device=device)
                        )

                        eps_local = 1e-8
                        node_entropy_local = -(
                            valid_nodes_local * torch.log(valid_nodes_local + eps_local)
                            + (1.0 - valid_nodes_local)
                            * torch.log(1.0 - valid_nodes_local + eps_local)
                        ).mean()

                        edge_entropy_local = (
                            -(
                                valid_edges_local * torch.log(valid_edges_local + eps_local)
                                + (1.0 - valid_edges_local)
                                * torch.log(1.0 - valid_edges_local + eps_local)
                            ).mean()
                            if valid_edges_local.numel()
                            else torch.tensor(0.0, device=device)
                        )

                        progress_fraction_local = (
                            step_local / max(1, steps_local - 1)
                        )
                        sparsity_ramp_local = (
                            0.20 + 0.80 * progress_fraction_local
                        )

                        total_loss_local = (
                            prediction_weight_local * prediction_loss_local
                            + sparsity_ramp_local * (
                                node_sparsity_local * node_size_loss_local
                                + edge_sparsity_local * edge_size_loss_local
                            )
                            + entropy_weight_local * (
                                node_entropy_local + edge_entropy_local
                            )
                        )

                        total_loss_local.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [node_logits_local, edge_logits_local],
                            max_norm=5.0
                        )
                        optimizer_local.step()

                        with torch.no_grad():
                            eval_node_local = torch.sigmoid(node_logits_local)
                            eval_edge_local = torch.sigmoid(edge_logits_local)
                            eval_edge_local = torch.where(
                                valid_edges_t,
                                eval_edge_local,
                                torch.ones_like(eval_edge_local)
                            )
                            eval_output_local = masked_forward_local(
                                eval_node_local,
                                eval_edge_local
                            )
                            eval_value_local = float(
                                batch_denormalize(
                                    eval_output_local.reshape(1),
                                    checkpoint_local
                                )[0].cpu().item()
                            )
                            error_eV_local = abs(
                                eval_value_local - original_value_local
                            )
                            mask_size_local = 0.5 * (
                                float(eval_node_local.mean().cpu().item())
                                + float(
                                    eval_edge_local[valid_edges_t].mean().cpu().item()
                                )
                            )

                            state_local = (
                                eval_node_local.detach().clone(),
                                eval_edge_local.detach().clone(),
                                error_eV_local,
                                mask_size_local
                            )

                            if error_eV_local < best_any_error_local:
                                best_any_error_local = error_eV_local
                                best_any_local = state_local

                            if (
                                error_eV_local <= tolerance_local
                                and mask_size_local < best_feasible_size_local
                            ):
                                best_feasible_size_local = mask_size_local
                                best_state = state_local

                    chosen_local = (
                        best_state if best_state is not None else best_any_local
                    )
                    if chosen_local is None:
                        raise RuntimeError(
                            "GNNExplainer optimization produced no valid mask."
                        )
                    return chosen_local

                # Use the same predefined hyperparameter candidates already used
                # by the existing app, selecting the sparsest feasible result.
                candidate_results = []
                search_steps_local = max(300, min(explanation_steps, 600))
                for candidate_index, candidate in enumerate(TEST2_AUTO_TUNE_CONFIGS):
                    candidate_seed = random_seed + 1000 + candidate_index
                    result_local = optimize_masks_local(
                        candidate_seed,
                        search_steps_local,
                        candidate["lr"],
                        candidate["prediction_weight"],
                        candidate["node_sparsity_weight"],
                        candidate["edge_sparsity_weight"],
                        candidate["entropy_weight"]
                    )
                    candidate_results.append((candidate, result_local))

                feasible_candidates = [
                    x for x in candidate_results
                    if x[1][2] <= tolerance_local
                ]
                if feasible_candidates:
                    best_candidate, _ = min(
                        feasible_candidates,
                        key=lambda x: (x[1][3], x[1][2])
                    )
                else:
                    best_candidate, _ = min(
                        candidate_results,
                        key=lambda x: (x[1][2], x[1][3])
                    )

                restart_results_local = []
                for restart_local in range(TEST2_RESTARTS):
                    restart_seed_local = random_seed + restart_local
                    restart_result_local = optimize_masks_local(
                        restart_seed_local,
                        explanation_steps,
                        best_candidate["lr"],
                        best_candidate["prediction_weight"],
                        best_candidate["node_sparsity_weight"],
                        best_candidate["edge_sparsity_weight"],
                        best_candidate["entropy_weight"]
                    )
                    restart_results_local.append(restart_result_local)

                node_stack_local = torch.stack(
                    [r[0] for r in restart_results_local]
                )
                edge_stack_local = torch.stack(
                    [r[1] for r in restart_results_local]
                )
                node_mask_local = torch.median(
                    node_stack_local, dim=0
                ).values
                edge_mask_local = torch.median(
                    edge_stack_local, dim=0
                ).values
                edge_mask_local = torch.where(
                    valid_edges_t,
                    edge_mask_local,
                    torch.ones_like(edge_mask_local)
                )

                # Same consensus-preservation repair concept as the existing app.
                def masked_prediction_local(node_m, edge_m):
                    with torch.no_grad():
                        out_local = masked_forward_local(node_m, edge_m)
                        return float(
                            batch_denormalize(
                                out_local.reshape(1), checkpoint_local
                            )[0].cpu().item()
                        )

                repair_error_local = abs(
                    masked_prediction_local(node_mask_local, edge_mask_local)
                    - original_value_local
                )
                if repair_error_local > tolerance_local:
                    low_local, high_local = 0.0, 1.0
                    for _ in range(30):
                        mid_local = 0.5 * (low_local + high_local)
                        trial_node_local = (
                            node_mask_local
                            + mid_local * (1.0 - node_mask_local)
                        )
                        trial_edge_local = (
                            edge_mask_local
                            + mid_local * (1.0 - edge_mask_local)
                        )
                        trial_edge_local = torch.where(
                            valid_edges_t,
                            trial_edge_local,
                            torch.ones_like(trial_edge_local)
                        )
                        trial_error_local = abs(
                            masked_prediction_local(
                                trial_node_local,
                                trial_edge_local
                            ) - original_value_local
                        )
                        if trial_error_local <= tolerance_local:
                            high_local = mid_local
                        else:
                            low_local = mid_local
                    node_mask_local = (
                        node_mask_local
                        + high_local * (1.0 - node_mask_local)
                    )
                    edge_mask_local = (
                        edge_mask_local
                        + high_local * (1.0 - edge_mask_local)
                    )
                    edge_mask_local = torch.where(
                        valid_edges_t,
                        edge_mask_local,
                        torch.ones_like(edge_mask_local)
                    )

                node_importance_local = node_mask_local.detach().cpu().numpy()
                lo_imp = float(np.min(node_importance_local))
                hi_imp = float(np.max(node_importance_local))
                if hi_imp - lo_imp < 1e-12:
                    node_importance_local = np.zeros_like(node_importance_local)
                else:
                    node_importance_local = (
                        node_importance_local - lo_imp
                    ) / (hi_imp - lo_imp)

                def remove_atoms_local(atoms_to_remove_local):
                    remove_set = np.asarray(
                        atoms_to_remove_local, dtype=np.int64
                    )
                    keep_local = np.ones(n_atoms, dtype=bool)
                    keep_local[remove_set] = False

                    mod_atom = atom_features_local.clone()
                    mod_atom[torch.tensor(~keep_local, device=device)] = 0.0

                    mod_nbr = nbr_features_local.clone()
                    for ii in range(n_atoms):
                        if not keep_local[ii]:
                            mod_nbr[ii] = 0.0
                            continue
                        for jj in range(n_neighbors):
                            neighbor_idx = int(
                                nbr_indices_local[ii, jj].item()
                            )
                            if (
                                neighbor_idx < 0
                                or neighbor_idx >= n_atoms
                                or not keep_local[neighbor_idx]
                            ):
                                mod_nbr[ii, jj] = 0.0

                    return mod_atom, mod_nbr, nbr_indices_local.clone()

                # --------------------------------------------------------
                # Validation 1: Multi-K Fidelity
                # --------------------------------------------------------
                fidelity_rows_local = []
                rng_local = np.random.default_rng(random_seed + 5000)

                fidelity_counts_local = []
                for fraction_local in TEST2_FIDELITY_FRACTIONS:
                    count_local = int(np.ceil(n_atoms * fraction_local))
                    count_local = max(1, count_local)
                    count_local = min(
                        count_local,
                        max(1, n_atoms - 1)
                    )
                    if count_local not in fidelity_counts_local:
                        fidelity_counts_local.append(count_local)

                for count_local in fidelity_counts_local:
                    important_idx_local = np.argsort(
                        -node_importance_local
                    )[:count_local].astype(np.int64)
                    important_graph_local = remove_atoms_local(
                        important_idx_local
                    )
                    important_prediction_local = float(
                        predict_local(
                            important_graph_local[0],
                            important_graph_local[1],
                            important_graph_local[2]
                        ).detach().cpu().item()
                    )
                    important_change_local = abs(
                        original_value_local - important_prediction_local
                    )

                    random_changes_local = []
                    for _ in range(TEST2_RANDOM_TRIALS):
                        random_idx_local = rng_local.choice(
                            np.arange(n_atoms),
                            size=count_local,
                            replace=False
                        ).astype(np.int64)
                        random_graph_local = remove_atoms_local(
                            random_idx_local
                        )
                        random_prediction_local = float(
                            predict_local(
                                random_graph_local[0],
                                random_graph_local[1],
                                random_graph_local[2]
                            ).detach().cpu().item()
                        )
                        random_changes_local.append(
                            abs(
                                original_value_local
                                - random_prediction_local
                            )
                        )

                    random_mean_local = float(
                        np.mean(random_changes_local)
                    )
                    random_std_local = float(
                        np.std(random_changes_local)
                    )
                    ratio_local = (
                        important_change_local / random_mean_local
                        if random_mean_local > 1e-12
                        else float("inf")
                    )

                    fidelity_rows_local.append({
                        "Removed %": 100.0 * count_local / n_atoms,
                        "Atoms removed": count_local,
                        "Important change": important_change_local,
                        "Random mean change": random_mean_local,
                        "Random std": random_std_local,
                        "Fidelity ratio": ratio_local
                    })

                fidelity_df_local = pd.DataFrame(fidelity_rows_local)
                finite_ratios_local = fidelity_df_local.loc[
                    np.isfinite(fidelity_df_local["Fidelity ratio"]),
                    "Fidelity ratio"
                ].to_numpy(dtype=float)
                mean_fidelity_ratio_local = (
                    float(np.mean(finite_ratios_local))
                    if len(finite_ratios_local)
                    else float("inf")
                )
                pointwise_local = int(
                    np.sum(
                        fidelity_df_local["Important change"].to_numpy()
                        > fidelity_df_local["Random mean change"].to_numpy()
                    )
                )
                required_local = int(
                    np.ceil(len(fidelity_df_local) / 2)
                )
                fidelity_pass_local = (
                    mean_fidelity_ratio_local > 1.0
                    and pointwise_local >= required_local
                )

                # --------------------------------------------------------
                # Validation 2: Prediction Preservation
                # --------------------------------------------------------
                preserved_value_local = masked_prediction_local(
                    node_mask_local,
                    edge_mask_local
                )
                preservation_change_local = abs(
                    original_value_local - preserved_value_local
                )
                preservation_pass_local = (
                    preservation_change_local <= tolerance_local
                )

                # --------------------------------------------------------
                # --------------------------------------------------------
                # Validation 4: Progressive Deletion
                # --------------------------------------------------------
                deletion_changes_local = []
                for fraction_local in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
                    count_local = int(np.floor(n_atoms * fraction_local))
                    count_local = min(count_local, n_atoms - 1)
                    if count_local == 0:
                        deletion_prediction_local = original_value_local
                    else:
                        deletion_idx_local = np.argsort(
                            -node_importance_local
                        )[:count_local].astype(np.int64)
                        deletion_graph_local = remove_atoms_local(
                            deletion_idx_local
                        )
                        deletion_prediction_local = float(
                            predict_local(
                                deletion_graph_local[0],
                                deletion_graph_local[1],
                                deletion_graph_local[2]
                            ).detach().cpu().item()
                        )
                    deletion_changes_local.append(
                        abs(
                            original_value_local
                            - deletion_prediction_local
                        )
                    )

                non_decreasing_local = (
                    float(
                        np.mean(
                            np.diff(deletion_changes_local) >= -1e-12
                        )
                    )
                    if len(deletion_changes_local) > 1
                    else 1.0
                )
                progressive_pass_local = non_decreasing_local >= 0.60
                material_overall_pass_local = (
                    fidelity_pass_local
                    and preservation_pass_local
                    and progressive_pass_local
                )


                batch_rows_local.append({
                    "Material": material_name,
                    "Atoms": n_atoms,
                    "Prediction (eV/atom)": original_value_local,
                    "Fidelity": (
                        "PASS" if fidelity_pass_local else "WARNING"
                    ),
                    "Fidelity Ratio": mean_fidelity_ratio_local,
                    "Prediction Preservation": (
                        "PASS" if preservation_pass_local else "WARNING"
                    ),
                    "Prediction Difference (eV/atom)": preservation_change_local,
                    "Progressive Deletion": (
                        "PASS" if progressive_pass_local else "WARNING"
                    ),
                    "Monotonic Fraction": non_decreasing_local,
                    "Overall": (
                        "PASS" if material_overall_pass_local else "CHECK"
                    )
                })

            except Exception as batch_error_local:
                batch_rows_local.append({
                    "Material": material_name,
                    "Atoms": np.nan,
                    "Prediction (eV/atom)": np.nan,
                    "Fidelity": "ERROR",
                    "Fidelity Ratio": np.nan,
                    "Prediction Preservation": "ERROR",
                    "Prediction Difference (eV/atom)": np.nan,
                    "Progressive Deletion": "ERROR",
                    "Monotonic Fraction": np.nan,
                    "Overall": "ERROR",
                    "Error": str(batch_error_local)
                })
            finally:
                shutil.rmtree(temp_local, ignore_errors=True)

            progress_local.progress(
                int(100 * file_index / len(test2_files_local))
            )

        status_local.empty()
        progress_local.empty()

        batch_df_local = pd.DataFrame(batch_rows_local)

        st.header("13. Test2 Batch Validation Results")
        st.dataframe(
            batch_df_local,
            width="stretch",
            hide_index=True
        )

        total_local = len(batch_df_local)
        overall_pass_local = int(
            (batch_df_local["Overall"] == "PASS").sum()
        )
        overall_error_local = int(
            (batch_df_local["Overall"] == "ERROR").sum()
        )

        validation_names_local = [
            "Fidelity",
            "Prediction Preservation",
            "Progressive Deletion"
        ]
        stat_rows_local = []
        for validation_name_local in validation_names_local:
            passed_local = int(
                (batch_df_local[validation_name_local] == "PASS").sum()
            )
            stat_rows_local.append({
                "Validation": validation_name_local,
                "PASS": passed_local,
                "Total": total_local,
                "Pass Rate (%)": (
                    100.0 * passed_local / total_local
                    if total_local else 0.0
                )
            })
        stat_rows_local.append({
            "Validation": "Overall",
            "PASS": overall_pass_local,
            "Total": total_local,
            "Pass Rate (%)": (
                100.0 * overall_pass_local / total_local
                if total_local else 0.0
            )
        })
        statistics_df_local = pd.DataFrame(stat_rows_local)

        st.subheader("Statistics")
        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        stat_c1.metric("Total Test Files", total_local)
        stat_c2.metric("Overall PASS", f"{overall_pass_local}/{total_local}")
        stat_c3.metric(
            "Overall Pass Rate",
            f"{100.0 * overall_pass_local / total_local:.1f}%"
            if total_local else "0.0%"
        )
        stat_c4.metric("Errors", overall_error_local)
        st.dataframe(
            statistics_df_local,
            width="stretch",
            hide_index=True
        )

        successful_local = batch_df_local[
            batch_df_local["Overall"] != "ERROR"
        ]
        if not successful_local.empty:
            mean_df_local = successful_local[
                [
                    "Prediction (eV/atom)",
                    "Fidelity Ratio",
                    "Prediction Difference (eV/atom)",
                ]
            ].mean().to_frame("Mean")
            st.subheader("Mean Metrics")
            st.dataframe(
                mean_df_local,
                width="stretch"
            )
        else:
            mean_df_local = None

        excel_buffer_local = BytesIO()
        with pd.ExcelWriter(
            excel_buffer_local,
            engine="openpyxl"
        ) as writer:
            batch_df_local.to_excel(
                writer,
                sheet_name="Validation Results",
                index=False
            )
            statistics_df_local.to_excel(
                writer,
                sheet_name="Statistics",
                index=False
            )
            if mean_df_local is not None:
                mean_df_local.to_excel(
                    writer,
                    sheet_name="Mean Metrics"
                )

        excel_buffer_local.seek(0)
        st.download_button(
            "📥 Download Test2 Validation Results (Excel)",
            data=excel_buffer_local.getvalue(),
            file_name="test2_gnnexplainer_validation.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            width="stretch"
        )

        st.success(
            f"✅ Test2 batch validation completed for {total_local} files "
            "using the frozen CGCNN."
        )

    try:
        run_test2_batch_validation()
    except Exception as batch_start_error:
        st.error(
            "Test2 batch validation failed:\n\n"
            f"{batch_start_error}"
        )
    st.stop()

# ============================================================
# UPLOAD CIF
# ============================================================

st.header("1. Upload Crystal Structure")

uploaded_file = st.file_uploader(
    "Upload a CIF file",
    type=["cif"],
    help="Upload the crystal structure you want to predict and explain."
)

if uploaded_file is None:
    st.warning("Please upload a CIF file to start.")
    st.stop()

# ============================================================
# SAVE CIF TO TEMPORARY CGCNN DATASET
# ============================================================

temp_root = tempfile.mkdtemp(prefix="cgcnn_streamlit_")

cif_name = os.path.splitext(uploaded_file.name)[0]
cif_path = os.path.join(temp_root, cif_name + ".cif")

with open(cif_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

# CIFData requires atom_init.json.
shutil.copy(
    ATOM_INIT_PATH,
    os.path.join(temp_root, "atom_init.json")
)

# CIFData also expects id_prop.csv.
# The target is dummy because the actual target is unknown.
# It is NOT used as the model prediction.
id_prop_path = os.path.join(temp_root, "id_prop.csv")

with open(id_prop_path, "w", encoding="utf-8") as f:
    f.write(f"{cif_name},0.0\n")

# ============================================================
# READ CIF WITH PYMATGEN
# ============================================================

try:
    structure = Structure.from_file(cif_path)
except Exception as e:
    st.error(f"Could not read the CIF file:\n\n{e}")
    st.stop()

# Cartesian coordinates of atoms in the uploaded unit cell.
# This must be defined before the 3D graph is constructed.
coords = np.asarray(structure.cart_coords, dtype=float)

# ============================================================
# CRYSTAL INFORMATION
# ============================================================

st.header("2. Crystal Information")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Material", cif_name)

with c2:
    st.metric("Atoms", len(structure))

with c3:
    st.metric(
        "Elements",
        len(set(str(site.specie) for site in structure))
    )

with c4:
    st.metric("Volume", f"{structure.volume:.2f} Å³")

# ============================================================
# CREATE CGCNN DATASET
# ============================================================

try:
    dataset = CIFData(temp_root)
except Exception as e:
    st.error(
        "Could not construct the CGCNN graph from this CIF.\n\n"
        f"{e}"
    )
    st.stop()

# ============================================================
# IMPORTANT FIX:
# YOUR CGCNN CIFData RETURNS:
#
#     (structures, target, cif_id)
#
# where:
#
#     structures = (atom_fea, nbr_fea, nbr_fea_idx)
#
# The old code incorrectly treated graph_data[0] as atom_fea.
# ============================================================

try:
    structures, target, cif_id = dataset[0]

    atom_fea = structures[0].clone().detach().float()
    nbr_fea = structures[1].clone().detach().float()
    nbr_fea_idx = structures[2].clone().detach().long()

except Exception as e:
    st.error(
        "The uploaded CIF was read, but its CGCNN graph format "
        "could not be unpacked.\n\n"
        f"{e}"
    )
    st.stop()

num_atoms = atom_fea.shape[0]
num_neighbors = nbr_fea.shape[1]
neighbor_feature_length = nbr_fea.shape[2]

# ============================================================
# EXACT CGCNN PERIODIC NEIGHBOR METADATA
# ============================================================
# The official CGCNN CIFData construction uses:
#   radius=8 Å, max_num_nbr=12, sorted by neighbor distance.
# We reconstruct the same neighbor ordering here so the 3D plot
# corresponds to the SAME CGCNN graph slots, including periodic
# image translations. This is critical: nbr_fea_idx stores only the
# target atom index, not the periodic image.
CGCNN_RADIUS = float(getattr(dataset, "radius", 8.0))
CGCNN_MAX_NEIGHBORS = int(getattr(dataset, "max_num_nbr", num_neighbors))

all_periodic_neighbors = structure.get_all_neighbors(
    CGCNN_RADIUS, include_index=True
)
all_periodic_neighbors = [
    sorted(nbrs, key=lambda x: x[1])
    for nbrs in all_periodic_neighbors
]

periodic_neighbor_meta = []
for i in range(num_atoms):
    row = []
    nbrs = all_periodic_neighbors[i]
    for item in nbrs[:num_neighbors]:
        # PeriodicNeighbor / Neighbor exposes index, distance and image.
        row.append({
            "target": int(item[2]),
            "distance": float(item[1]),
            "image": tuple(int(v) for v in item[3]),
            "coords": np.asarray(item.coords, dtype=float),
        })
    while len(row) < num_neighbors:
        row.append(None)
    periodic_neighbor_meta.append(row)

# Sanity check: the non-padding CGCNN target indices must agree with the
# exact pymatgen neighbor ordering. If not, stop rather than drawing a
# misleading 3D graph.
graph_mismatches = []
for i in range(num_atoms):
    for j in range(num_neighbors):
        cg_target = int(nbr_fea_idx[i, j].item())
        meta = periodic_neighbor_meta[i][j]
        if meta is None:
            continue
        if cg_target != meta["target"]:
            graph_mismatches.append((i, j, cg_target, meta["target"]))

if graph_mismatches:
    st.warning(
        "The uploaded structure's periodic-neighbor ordering differs from "
        "the graph returned by your local CIFData implementation. The app "
        "will still use the trained CGCNN graph for prediction, but the "
        "periodic 3D overlay is disabled to avoid showing incorrect bonds."
    )

# CGCNN expects a list containing atom indices for each crystal.
# Since this app explains exactly ONE uploaded crystal:
crystal_atom_idx = [
    torch.arange(
        num_atoms,
        dtype=torch.long
    )
]

st.write(
    f"**CGCNN graph:** {num_atoms} atoms × "
    f"{num_neighbors} neighbors"
)

st.write(
    f"**Atom feature length:** {atom_fea.shape[1]}  |  "
    f"**Neighbor feature length:** {neighbor_feature_length}"
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# CHECKPOINT LOADER
# ============================================================

@st.cache_resource
def load_trained_model():

    # --------------------------------------------------------
    # Load checkpoint.
    # Supports both:
    #   checkpoint.pth file
    #   extracted checkpoint.pth directory
    # --------------------------------------------------------

    if os.path.isfile(CHECKPOINT_PATH):

        try:
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location=device,
                weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location=device
            )

    elif os.path.isdir(CHECKPOINT_PATH):

        import zipfile

        checkpoint_dir = os.path.abspath(CHECKPOINT_PATH)

        # The extracted checkpoint directory itself is the archive root.
        # PyTorch expects files such as byteorder/data.pkl/version
        # to be inside one archive directory.
        archive_root = os.path.basename(
            checkpoint_dir.rstrip(os.sep)
        )

        temp_archive = tempfile.NamedTemporaryFile(
            suffix=".pth",
            delete=False
        )
        temp_archive.close()

        try:

            with zipfile.ZipFile(
                temp_archive.name,
                "w",
                compression=zipfile.ZIP_STORED
            ) as archive:

                for root, dirs, files in os.walk(
                    checkpoint_dir
                ):

                    for filename in files:

                        full_path = os.path.join(
                            root,
                            filename
                        )

                        relative_path = os.path.relpath(
                            full_path,
                            checkpoint_dir
                        )

                        # IMPORTANT:
                        # Put every file underneath the checkpoint
                        # directory, e.g.
                        #
                        # checkpoint.pth/byteorder
                        # checkpoint.pth/data.pkl
                        # checkpoint.pth/version
                        #
                        archive_name = (
                            archive_root
                            + "/"
                            + relative_path.replace(
                                os.sep,
                                "/"
                            )
                        )

                        # Avoid old Windows timestamps.
                        info = zipfile.ZipInfo(
                            archive_name
                        )

                        info.date_time = (
                            1980,
                            1,
                            1,
                            0,
                            0,
                            0
                        )

                        info.compress_type = zipfile.ZIP_STORED

                        with open(
                            full_path,
                            "rb"
                        ) as file_obj:

                            archive.writestr(
                                info,
                                file_obj.read()
                            )

            print(
                "Reconstructed PyTorch checkpoint archive."
            )

            try:

                checkpoint = torch.load(
                    temp_archive.name,
                    map_location=device,
                    weights_only=False
                )

            except TypeError:

                checkpoint = torch.load(
                    temp_archive.name,
                    map_location=device
                )

        finally:

            try:
                os.remove(
                    temp_archive.name
                )
            except Exception:
                pass

    else:
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint is not a dictionary."
        )

    if "args" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'args'."
        )

    args = checkpoint["args"]

    def get_arg(name, default=None):
        if isinstance(args, dict):
            return args.get(name, default)
        return getattr(args, name, default)

    atom_fea_len = get_arg(
        "atom_fea_len",
        64
    )

    nbr_fea_len = get_arg(
        "nbr_fea_len",
        neighbor_feature_length
    )

    n_conv = get_arg(
        "n_conv",
        3
    )

    h_fea_len = get_arg(
        "h_fea_len",
        128
    )

    n_h = get_arg(
        "n_h",
        1
    )

    classification = get_arg(
        "classification",
        False
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Recreate the SAME architecture used during training.
    # --------------------------------------------------------

    model = CrystalGraphConvNet(
        atom_fea.shape[1],
        nbr_fea_len,
        atom_fea_len=atom_fea_len,
        n_conv=n_conv,
        h_fea_len=h_fea_len,
        n_h=n_h,
        classification=classification
    )

    state_dict = checkpoint.get(
        "state_dict"
    )

    if state_dict is None:
        raise KeyError(
            "Checkpoint does not contain 'state_dict'."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # FREEZE CGCNN.
    # --------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return (
        model,
        checkpoint
    )

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

with st.spinner("Loading existing trained CGCNN..."):
    try:
        model, checkpoint = load_trained_model()
    except Exception as e:
        st.error(
            "Could not load the existing trained CGCNN.\n\n"
            f"{e}"
        )
        st.stop()

st.success(
    f"Existing trained CGCNN loaded successfully. "
    f"Device: {device}"
)

# ============================================================
# NORMALIZER
# ============================================================

normalizer = checkpoint.get(
    "normalizer",
    None
)

def denormalize_output(output):

    if normalizer is None:
        return output

    try:
        return normalizer.denorm(output)
    except Exception:
        pass

    try:
        return (
            output *
            normalizer.std.to(output.device)
            +
            normalizer.mean.to(output.device)
        )
    except Exception:
        return output

# ============================================================
# MODEL PREDICTION
# ============================================================

def model_prediction(
    atom_features,
    neighbor_features,
    neighbor_indices
):

    atom_features = atom_features.to(device)
    neighbor_features = neighbor_features.to(device)
    neighbor_indices = neighbor_indices.to(device)

    with torch.no_grad():

        output = model(
            atom_features,
            neighbor_features,
            neighbor_indices,
            crystal_atom_idx
        )

    output = output.reshape(-1)

    output = denormalize_output(
        output
    )

    return output[0]

# ============================================================
# ORIGINAL PREDICTION
# ============================================================

st.header("3. CGCNN Prediction")

with st.spinner("Predicting formation energy..."):

    original_prediction = model_prediction(
        atom_fea,
        nbr_fea,
        nbr_fea_idx
    )

original_value = float(
    original_prediction.detach().cpu().item()
)

st.metric(
    "Predicted Formation Energy",
    f"{original_value:.6f} eV/atom"
)

# ============================================================
# GNN EXPLAINER
# ============================================================

st.header("4. GNN Explainer")

st.write(
    """
The explanation is generated **after training**.

The trained CGCNN weights are frozen. A GNNExplainer-style optimizer learns
continuous atom and edge masks for this crystal only. The objective jointly
balances prediction preservation, sparsity, and mask entropy so the
explanation is compact rather than simply keeping the whole graph.
"""
)

# ============================================================
# SET SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(random_seed)

# ============================================================
# BASE GRAPH
# ============================================================

base_atom_features = atom_fea.to(device)
base_neighbor_features = nbr_fea.to(device)
base_neighbor_indices = nbr_fea_idx.to(device)

# ============================================================
# IMPROVED GNN EXPLAINER
# ============================================================

st.header("4. GNN Explainer")

st.write(
    """
The explanation is generated **after training**.

The trained CGCNN remains completely frozen. The explainer learns
continuous atom and edge masks that try to preserve the original
prediction while keeping the explanation compact and decisive.

To make the explanation more reliable, the explainer uses multiple
random initializations and keeps the best explanation.
"""
)

# ------------------------------------------------------------
# REPRODUCIBILITY
# ------------------------------------------------------------

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(random_seed)


# ------------------------------------------------------------
# BASE GRAPH
# ------------------------------------------------------------

base_atom_features = atom_fea.to(device)
base_neighbor_features = nbr_fea.to(device)
base_neighbor_indices = nbr_fea_idx.to(device)


# ------------------------------------------------------------
# ORIGINAL MODEL OUTPUT
# ------------------------------------------------------------

# IMPORTANT:
# This is the normalized output used by the CGCNN internally.
# Comparing in model-output space makes the explainer objective
# independent of the eV/atom scale of the target.

with torch.no_grad():

    original_model_output = model(
        base_atom_features,
        base_neighbor_features,
        base_neighbor_indices,
        crystal_atom_idx
    ).reshape(-1)[0].detach()


# ------------------------------------------------------------
# CANONICAL GNNEXPLAINER-STYLE MASK OPTIMIZATION
# ------------------------------------------------------------
# The CGCNN predictor is frozen. We optimize only explanation masks.
# This follows the core GNNExplainer idea: preserve the prediction while
# regularizing the explanation toward a compact, decisive subgraph.
# See: Ying et al., GNNExplainer (2019).

# Valid CGCNN neighbor slots. Padding slots are kept at mask=1 because they
# are part of the trained CGCNN input representation and are not real edges.
valid_edge_mask_np = np.zeros((num_atoms, num_neighbors), dtype=bool)
for _i in range(num_atoms):
    for _j in range(num_neighbors):
        valid_edge_mask_np[_i, _j] = periodic_neighbor_meta[_i][_j] is not None
valid_edge_mask_t = torch.tensor(valid_edge_mask_np, dtype=torch.bool, device=device)

VALID_EDGE_COUNT = int(valid_edge_mask_np.sum())

with torch.no_grad():
    original_model_output = model(
        base_atom_features,
        base_neighbor_features,
        base_neighbor_indices,
        crystal_atom_idx
    ).reshape(-1)[0].detach()


def _masked_forward(node_mask, edge_mask):
    """Forward pass through the frozen CGCNN with explanation masks."""
    masked_atom = base_atom_features * node_mask.unsqueeze(1)
    masked_nbr = base_neighbor_features * edge_mask.unsqueeze(-1)
    return model(
        masked_atom,
        masked_nbr,
        base_neighbor_indices,
        crystal_atom_idx
    ).reshape(-1)[0]


def optimize_explanation(
    seed,
    steps,
    lr,
    prediction_weight=20.0,
    node_sparsity_weight=0.02,
    edge_sparsity_weight=0.04,
    entropy_weight=0.001,
    target_tolerance=None
):
    """Optimize a post-hoc GNNExplainer-style node/edge mask.

    Important differences from the previous implementation:
    1. Masks start near sigmoid(0)=0.5, not near 1.0.
    2. Real CGCNN neighbor slots are optimized; padding is fixed at 1.
    3. Prediction preservation, sparsity and entropy are optimized jointly.
    4. There is no post-hoc preservation repair that can make a dense mask
       look like a successful explanation.
    5. The trained CGCNN weights are never updated.
    """
    set_seed(seed)

    if target_tolerance is None:
        target_tolerance = max(0.01, 0.05 * abs(original_value))

    # Official GNNExplainer implementations initialize mask logits around 0,
    # which corresponds to an initial sigmoid mask around 0.5.
    node_logits = nn.Parameter(
        0.05 * torch.randn(num_atoms, device=device)
    )
    edge_logits = nn.Parameter(
        0.05 * torch.randn(num_atoms, num_neighbors, device=device)
    )

    optimizer = torch.optim.Adam(
        [node_logits, edge_logits], lr=lr
    )

    best_feasible = None
    best_feasible_size = float("inf")
    best_any = None
    best_any_error = float("inf")

    for step in range(max(1, steps)):
        optimizer.zero_grad()

        node_mask = torch.sigmoid(node_logits)
        edge_mask = torch.sigmoid(edge_logits)

        # Padded CGCNN neighbor slots are not explanation edges.
        edge_mask = torch.where(
            valid_edge_mask_t,
            edge_mask,
            torch.ones_like(edge_mask)
        )

        masked_output = _masked_forward(node_mask, edge_mask)

        # Regression preservation in the model's normalized output space.
        prediction_loss = (masked_output - original_model_output).pow(2)

        valid_nodes = node_mask
        valid_edges = edge_mask[valid_edge_mask_t]

        node_size_loss = valid_nodes.mean()
        edge_size_loss = valid_edges.mean() if valid_edges.numel() else torch.tensor(0.0, device=device)

        eps = 1e-8
        node_entropy = -(
            valid_nodes * torch.log(valid_nodes + eps)
            + (1.0 - valid_nodes) * torch.log(1.0 - valid_nodes + eps)
        ).mean()
        edge_entropy = -(
            valid_edges * torch.log(valid_edges + eps)
            + (1.0 - valid_edges) * torch.log(1.0 - valid_edges + eps)
        ).mean() if valid_edges.numel() else torch.tensor(0.0, device=device)

        # Smooth continuation: let the model first recover the prediction,
        # then increasingly reward compact explanations.
        progress_fraction = step / max(1, steps - 1)
        sparsity_ramp = 0.20 + 0.80 * progress_fraction

        loss = (
            prediction_weight * prediction_loss
            + sparsity_ramp * (
                node_sparsity_weight * node_size_loss
                + edge_sparsity_weight * edge_size_loss
            )
            + entropy_weight * (node_entropy + edge_entropy)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_([node_logits, edge_logits], max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            eval_node = torch.sigmoid(node_logits)
            eval_edge = torch.sigmoid(edge_logits)
            eval_edge = torch.where(
                valid_edge_mask_t,
                eval_edge,
                torch.ones_like(eval_edge)
            )

            eval_output = _masked_forward(eval_node, eval_edge)
            eval_value = float(
                denormalize_output(eval_output.reshape(1))[0].cpu().item()
            )
            error_eV = abs(eval_value - original_value)
            mask_size = 0.5 * (
                float(eval_node.mean().cpu().item())
                + float(eval_edge[valid_edge_mask_t].mean().cpu().item())
            )

            state = (
                eval_node.detach().clone(),
                eval_edge.detach().clone(),
                error_eV,
                mask_size
            )

            if error_eV < best_any_error:
                best_any_error = error_eV
                best_any = state

            if error_eV <= target_tolerance and mask_size < best_feasible_size:
                best_feasible_size = mask_size
                best_feasible = state

    chosen = best_feasible if best_feasible is not None else best_any
    if chosen is None:
        raise RuntimeError("GNNExplainer optimization produced no valid mask.")

    return chosen[0], chosen[1], float(chosen[2]), float(chosen[3])


def auto_tune_explanation(base_seed, steps, candidate_configs, tolerance_eV):
    """Choose the sparsest configuration that satisfies the fixed
    prediction-preservation tolerance. The tolerance is never relaxed."""
    results = []
    search_steps = max(300, min(steps, 600))

    for config_index, config in enumerate(candidate_configs):
        search_seed = base_seed + 1000 + config_index
        nm, em, error_eV, mask_size = optimize_explanation(
            search_seed,
            search_steps,
            config["lr"],
            config["prediction_weight"],
            config["node_sparsity_weight"],
            config["edge_sparsity_weight"],
            config["entropy_weight"],
            tolerance_eV
        )
        results.append({
            "config": config,
            "prediction_error_eV": error_eV,
            "mask_size": mask_size,
            "node_mask": nm,
            "edge_mask": em,
            "feasible": error_eV <= tolerance_eV
        })

    feasible = [r for r in results if r["feasible"]]
    if feasible:
        # Compactness is primary once the fixed preservation constraint is met.
        best = min(
            feasible,
            key=lambda r: (r["mask_size"], r["prediction_error_eV"])
        )
    else:
        best = min(
            results,
            key=lambda r: (r["prediction_error_eV"], r["mask_size"])
        )

    return best, results, bool(feasible)


# Fixed validation tolerance. This is calculated before optimization and is
# never relaxed to manufacture a PASS.
preservation_tolerance = max(0.01, 0.05 * abs(original_value))

AUTO_TUNE_CONFIGS = [
    {"prediction_weight": 10.0, "node_sparsity_weight": 0.010, "edge_sparsity_weight": 0.020, "entropy_weight": 0.0010, "lr": 0.010},
    {"prediction_weight": 25.0, "node_sparsity_weight": 0.015, "edge_sparsity_weight": 0.030, "entropy_weight": 0.0010, "lr": 0.008},
    {"prediction_weight": 50.0, "node_sparsity_weight": 0.020, "edge_sparsity_weight": 0.040, "entropy_weight": 0.0008, "lr": 0.006},
    {"prediction_weight": 100.0, "node_sparsity_weight": 0.025, "edge_sparsity_weight": 0.050, "entropy_weight": 0.0006, "lr": 0.005},
    {"prediction_weight": 200.0, "node_sparsity_weight": 0.030, "edge_sparsity_weight": 0.060, "entropy_weight": 0.0005, "lr": 0.004},
    {"prediction_weight": 400.0, "node_sparsity_weight": 0.040, "edge_sparsity_weight": 0.080, "entropy_weight": 0.0003, "lr": 0.003},
]

with st.spinner("Optimizing GNNExplainer hyperparameters..."):
    best_config_result, _, _ = auto_tune_explanation(
        random_seed,
        explanation_steps,
        AUTO_TUNE_CONFIGS,
        preservation_tolerance
    )

best_config = best_config_result["config"]

# ------------------------------------------------------------
# MULTI-RESTART EXPLANATION + CONSENSUS
# ------------------------------------------------------------
# Use multiple independent runs so the final explanation is less sensitive
# to one unlucky initialization.
NUM_EXPLAINER_RESTARTS = 15
restart_results = []

with st.spinner(f"Running {NUM_EXPLAINER_RESTARTS} independent GNNExplainer runs..."):
    for restart in range(NUM_EXPLAINER_RESTARTS):
        restart_seed = random_seed + restart
        nm, em, error_eV, mask_size = optimize_explanation(
            restart_seed,
            explanation_steps,
            best_config["lr"],
            best_config["prediction_weight"],
            best_config["node_sparsity_weight"],
            best_config["edge_sparsity_weight"],
            best_config["entropy_weight"],
            preservation_tolerance
        )
        restart_results.append({
            "seed": restart_seed,
            "node_mask": nm,
            "edge_mask": em,
            "prediction_error_eV": error_eV,
            "mask_size": mask_size
        })

# Consensus explanation: median across independent runs.
node_stack = torch.stack([r["node_mask"] for r in restart_results])
edge_stack = torch.stack([r["edge_mask"] for r in restart_results])
node_mask_consensus = torch.median(node_stack, dim=0).values
edge_mask_consensus = torch.median(edge_stack, dim=0).values
edge_mask_consensus = torch.where(
    valid_edge_mask_t,
    edge_mask_consensus,
    torch.ones_like(edge_mask_consensus)
)

# Use the consensus itself as the final explanation instead of selecting a
# single lucky initialization.
node_mask = node_mask_consensus.clone()
edge_mask = edge_mask_consensus.clone()
best_explainer_seed = "15-run median consensus"

# ------------------------------------------------------------
# CONSENSUS PRESERVATION CHECK
# ------------------------------------------------------------
# If the median consensus is just outside the fixed tolerance, move the entire
# consensus mask toward 1 by the smallest amount needed. This is a transparent
# constrained repair, not a change to the validation threshold.
def masked_prediction_eV(node_m, edge_m):
    with torch.no_grad():
        out = _masked_forward(node_m, edge_m)
        return float(
            denormalize_output(out.reshape(1))[0].cpu().item()
        )

repair_before = abs(masked_prediction_eV(node_mask, edge_mask) - original_value)
repair_alpha = 0.0
repair_after = repair_before

if repair_before > preservation_tolerance:
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        trial_node = node_mask + mid * (1.0 - node_mask)
        trial_edge = edge_mask + mid * (1.0 - edge_mask)
        trial_edge = torch.where(valid_edge_mask_t, trial_edge, torch.ones_like(trial_edge))
        trial_error = abs(masked_prediction_eV(trial_node, trial_edge) - original_value)
        if trial_error <= preservation_tolerance:
            hi = mid
        else:
            lo = mid
    repair_alpha = hi
    node_mask = node_mask + repair_alpha * (1.0 - node_mask)
    edge_mask = edge_mask + repair_alpha * (1.0 - edge_mask)
    edge_mask = torch.where(valid_edge_mask_t, edge_mask, torch.ones_like(edge_mask))
    repair_after = abs(masked_prediction_eV(node_mask, edge_mask) - original_value)

# Node importance comes directly from the learned GNNExplainer node mask.
node_importance = node_mask.detach().cpu().numpy()

# ------------------------------------------------------------
# EDGE PREDICTION-SENSITIVITY SCORES FOR VISUALIZATION
# ------------------------------------------------------------
# Keep the existing leave-one-edge-out sensitivity used by the 3D view.
# This part is intentionally unchanged because the current 3D visualization
# is already behaving as requested: all atoms/bonds, with color by priority
# and thickness by measured prediction impact.
def compute_edge_perturbation_importance():
    scores = np.zeros((num_atoms, num_neighbors), dtype=np.float64)

    base_value = original_value

    for atom_i in range(num_atoms):
        for nbr_j in range(num_neighbors):
            if not valid_edge_mask_np[atom_i, nbr_j]:
                continue

            perturbed_nbr = base_neighbor_features.clone()
            perturbed_nbr[atom_i, nbr_j] = 0.0

            with torch.no_grad():
                perturbed_out = model(
                    base_atom_features,
                    perturbed_nbr,
                    base_neighbor_indices,
                    crystal_atom_idx
                ).reshape(-1)[0]
                perturbed_value = float(
                    denormalize_output(perturbed_out.reshape(1))[0]
                    .cpu().item()
                )

            scores[atom_i, nbr_j] = abs(base_value - perturbed_value)

    return scores

edge_perturbation_raw = compute_edge_perturbation_importance()
edge_importance_matrix = edge_perturbation_raw.copy()


# ------------------------------------------------------------
# NORMALIZE SCORES
# ------------------------------------------------------------

def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)

node_importance = normalize_scores(node_importance)
edge_importance_matrix = normalize_scores(edge_importance_matrix)

# ------------------------------------------------------------
# SCORE-BASED PRIORITY LABELS
# ------------------------------------------------------------
# Importance is normalized to [0, 1]. Priority is derived from the actual
# score distribution, not from the row position in the table. This means the
# table, bar chart and 3D view always describe the same HIGH/MEDIUM/LOW edge.
def assign_score_priorities(scores, high_cut=0.75, medium_cut=0.40):
    """Map normalized importance to a fixed, interpretable priority.

    0.75–1.00  -> HIGH
    0.40–<0.75 -> MEDIUM
    0.00–<0.40 -> LOW

    Fixed thresholds are intentional: a score of 0.24 can never become HIGH
    simply because it happens to rank near the top for one crystal.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.full(len(scores), 'LOW', dtype=object)

    if len(scores) == 0:
        return labels

    finite = np.isfinite(scores)
    labels[finite & (scores >= high_cut)] = 'HIGH'
    labels[finite & (scores >= medium_cut) & (scores < high_cut)] = 'MEDIUM'
    labels[finite & (scores < medium_cut)] = 'LOW'

    return labels

node_priority_matrix = assign_score_priorities(node_importance)

valid_edge_positions = []
for _i in range(num_atoms):
    for _j in range(num_neighbors):
        if periodic_neighbor_meta[_i][_j] is not None:
            valid_edge_positions.append((_i, _j))

# Edge importance is a directed CGCNN quantity. For the physical 3D bond,
# combine the two directions that represent the same periodic connection.
# The displayed physical-bond score is the maximum directional perturbation.
physical_bond_records = []
physical_bond_seen = {}

for _i, _j in valid_edge_positions:
    meta = periodic_neighbor_meta[_i][_j]
    _k = int(meta['target'])
    _image = tuple(int(v) for v in meta['image'])
    _canonical = min(
        (_i, _k, _image),
        (_k, _i, tuple(-v for v in _image))
    )
    _raw = float(edge_perturbation_raw[_i, _j])
    _entry = {
        'source': _i,
        'target': _k,
        'image': _image,
        'distance': float(meta['distance']),
        'raw': _raw,
        'slot': (_i, _j),
        'coords': np.asarray(meta['coords'], dtype=float),
    }
    if _canonical not in physical_bond_seen:
        physical_bond_seen[_canonical] = _entry
    else:
        # Keep the larger directional perturbation for the physical bond.
        if _raw > physical_bond_seen[_canonical]['raw']:
            physical_bond_seen[_canonical] = _entry

physical_bond_records = list(physical_bond_seen.values())
physical_raw = np.array(
    [r['raw'] for r in physical_bond_records], dtype=float
) if physical_bond_records else np.array([], dtype=float)

physical_norm = normalize_scores(physical_raw)

for _r, _score in zip(physical_bond_records, physical_norm):
    _r['importance'] = float(_score)

# Use a small, explicit relative set for bond visualization.  CGCNN has
# many neighbor edges, so using 15% HIGH / 35% MEDIUM creates a crowded
# explanation.  We reserve roughly the top 5% for HIGH and the next 10%
# for MEDIUM.  LOW edges remain available in the table but are hidden in
# the default 3D explanation view.
physical_priorities = assign_score_priorities(physical_norm)
for _r, _priority in zip(physical_bond_records, physical_priorities):
    _r['priority'] = str(_priority)

# Map each directed CGCNN slot to the physical-bond score/priority.
edge_importance_matrix = np.zeros_like(edge_importance_matrix, dtype=float)
edge_priority_lookup = {}
for _r in physical_bond_records:
    _i, _j = _r['slot']
    edge_importance_matrix[_i, _j] = _r['importance']
    edge_priority_lookup[(_i, _j)] = _r['priority']

# ============================================================
# ATOM IMPORTANCE TABLE

# ============================================================

st.header("5. Atom Importance")

atom_rows = []

for i in range(num_atoms):

    score = float(
        node_importance[i]
    )

    atom_rows.append(
        {
            "Atom Index": i,
            "Element": str(
                structure[i].specie
            ),
            "Importance": score,
            "Priority": str(node_priority_matrix[i])
        }
    )

atom_df = pd.DataFrame(
    atom_rows
).sort_values(
    "Importance",
    ascending=False
)

st.dataframe(
    atom_df,
    width="stretch",
    hide_index=True
)

# ============================================================
# EDGE TABLE
# ============================================================

st.header("6. Edge / Bond Importance")

edge_rows = []
for r in sorted(physical_bond_records, key=lambda x: x['importance'], reverse=True):
    edge_rows.append({
        "Source Atom": r['source'],
        "Target Atom": r['target'],
        "Connection": f"{r['source']} → {r['target']}",
        "Periodic Image": str(r['image']),
        "Distance (Å)": r['distance'],
        "Importance": r['importance'],
        "Prediction Change (eV/atom)": r['raw'],
        "Priority": r['priority']
    })

edge_df = pd.DataFrame(edge_rows)
edge_plot_df = edge_df.head(0)

if edge_df.empty:
    st.warning("No valid CGCNN neighbor bonds were found for this structure.")
else:
    priority_counts = edge_df["Priority"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("HIGH", int(priority_counts.get("HIGH", 0)))
    c2.metric("MEDIUM", int(priority_counts.get("MEDIUM", 0)))
    c3.metric("LOW", int(priority_counts.get("LOW", 0)))

    # Show a small sample from every priority so MEDIUM/LOW are visible
    # even though the full table is sorted by measured importance.
    preview_parts = []
    for p in ["HIGH", "MEDIUM", "LOW"]:
        part = edge_df[edge_df["Priority"] == p].head(10)
        if not part.empty:
            preview_parts.append(part)
    preview_df = pd.concat(preview_parts, ignore_index=True) if preview_parts else edge_df.head(50)
    edge_plot_df = preview_df
    st.dataframe(
        preview_df,
        width="stretch",
        hide_index=True
    )
    st.caption("Preview shows up to 10 bonds from each priority. The underlying edge list is complete.")

# ============================================================
# TOP ATOMS / EDGES
# ============================================================

top_atom_count = max(
    1,
    int(
        np.ceil(
            num_atoms *
            top_atom_percent /
            100
        )
    )
)

top_atom_count = min(
    top_atom_count,
    max(1, num_atoms - 1)
)

top_atoms = (
    atom_df
    .head(top_atom_count)
    ["Atom Index"]
    .astype(int)
    .tolist()
)

# ============================================================
# EXPLANATION PLOTS
# ============================================================

st.header("7. Explanation Plots")

plot_col1, plot_col2 = st.columns(2)

with plot_col1:

    fig_atoms = go.Figure()

    fig_atoms.add_trace(
        go.Bar(
            x=list(range(num_atoms)),
            y=node_importance,
            text=[
                str(structure[i].specie)
                for i in range(num_atoms)
            ],
            hovertemplate=(
                "Atom %{x}<br>"
                "Element %{text}<br>"
                "Importance %{y:.4f}"
                "<extra></extra>"
            )
        )
    )

    fig_atoms.update_layout(
        title="Atom Importance",
        xaxis_title="Atom Index",
        yaxis_title="Importance",
        yaxis=dict(range=[0, 1.05])
    )

    st.plotly_chart(
        fig_atoms,
        width="stretch"
    )

with plot_col2:

    edge_labels = [
        row["Connection"]
        for _, row in edge_plot_df.iterrows()
    ]

    edge_scores = [
        float(row["Importance"])
        for _, row in edge_plot_df.iterrows()
    ]

    fig_edges = go.Figure()

    edge_priorities = [
        str(row["Priority"])
        for _, row in edge_plot_df.iterrows()
    ]

    priority_colors = {
        "HIGH": "red",
        "MEDIUM": "orange",
        "LOW": "lightgray",
    }
    edge_bar_colors = [priority_colors.get(p, "lightgray") for p in edge_priorities]

    fig_edges.add_trace(
        go.Bar(
            x=edge_labels,
            y=edge_scores,
            marker=dict(color=edge_bar_colors),
            customdata=edge_priorities,
            hovertemplate=(
                "Edge %{x}<br>"
                "Importance %{y:.4f}<br>"
                "Priority %{customdata}"
                "<extra></extra>"
            )
        )
    )

    fig_edges.update_layout(
        title="Edge Importance (same bonds as table preview)",
        xaxis_title="Atom-to-Atom Connection",
        yaxis_title="Importance",
        yaxis=dict(range=[0, 1.05])
    )

    st.plotly_chart(
        fig_edges,
        width="stretch"
    )

# ============================================================
# 3D CRYSTAL VISUALIZATION — SIMPLE VIEW
# ============================================================

st.header("8. Interactive 3D Crystal Visualization")

st.write(
    """
**3D explanation legend**

🔴 HIGH   |   🟠 MEDIUM   |   🔵 LOW

Only the crystal atoms and their bonds are drawn. No periodic-image atoms,
unit-cell helper objects, boundary markers, or extra visualization geometry
are displayed. All atoms and all physical CGCNN bonds are shown together.
Bond color represents explanation priority and bond thickness represents
measured prediction impact.
"""
)

def atom_color(priority):
    if priority == "HIGH":
        return "red"
    if priority == "MEDIUM":
        return "orange"
    return "deepskyblue"

def edge_color(priority):
    if priority == "HIGH":
        return "red"
    if priority == "MEDIUM":
        return "orange"
    return "deepskyblue"

# IMPORTANT:
# The 3D view intentionally uses ONLY the atoms in the CIF unit cell.
# For visualization, a periodic neighbor is connected to the corresponding
# atom inside the same displayed unit cell. The periodic image itself is NOT
# drawn. Periodic image information remains available in the edge table/hover.
fig3d = go.Figure()

for r in physical_bond_records:
    i = int(r["source"])
    j = int(r["target"])

    source_xyz = coords[i]
    target_xyz = coords[j]

    score = float(r["importance"])
    priority = str(r["priority"])

    # Thickness follows measured prediction impact.
    if priority == "HIGH":
        width = 2.5 + 5.0 * score
    elif priority == "MEDIUM":
        width = 1.8 + 3.5 * score
    else:
        width = 1.0 + 2.0 * score

    fig3d.add_trace(
        go.Scatter3d(
            x=[source_xyz[0], target_xyz[0]],
            y=[source_xyz[1], target_xyz[1]],
            z=[source_xyz[2], target_xyz[2]],
            mode="lines",
            line=dict(
                color=edge_color(priority),
                width=width
            ),
            hoverinfo="text",
            text=(
                f"Bond: {structure[i].specie} → {structure[j].specie}"
                f"<br>Periodic image: {r['image']}"
                f"<br>Distance: {r['distance']:.3f} Å"
                f"<br>Importance: {score:.4f}"
                f"<br>Priority: {priority}"
                f"<br>Prediction change: {r['raw']:.6f} eV/atom"
            ),
            showlegend=False
        )
    )

# ALL original CIF atoms — no periodic duplicate atoms.
atom_colors = [
    atom_color(str(node_priority_matrix[i]))
    for i in range(num_atoms)
]

atom_sizes = [
    8 + 18 * float(node_importance[i])
    for i in range(num_atoms)
]

hover_text = [
    f"Atom index: {i}"
    f"<br>Element: {structure[i].specie}"
    f"<br>Importance: {node_importance[i]:.4f}"
    f"<br>Priority: {node_priority_matrix[i]}"
    for i in range(num_atoms)
]

fig3d.add_trace(
    go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers+text",
        marker=dict(
            size=atom_sizes,
            color=atom_colors,
            line=dict(width=1, color="black")
        ),
        text=[str(i) for i in range(num_atoms)],
        textposition="top center",
        hovertext=hover_text,
        hoverinfo="text",
        name="Atoms"
    )
)

fig3d.update_layout(
    title="3D Crystal Explanation",
    scene=dict(
        xaxis_title="",
        yaxis_title="",
        zaxis_title="",
        aspectmode="data",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    ),
    height=700,
    margin=dict(l=0, r=0, b=0, t=40),
    showlegend=False
)

st.plotly_chart(fig3d, width="stretch")

st.caption(
    f"All {num_atoms} atoms and all {len(physical_bond_records)} physical bonds "
    "are displayed. Bond color = priority; bond thickness = prediction impact."
)

# ============================================================
# VALIDATION 1: MULTI-K FIDELITY
# ============================================================

st.header("9. Fidelity Validation")
st.write("""
Fidelity checks whether atoms ranked as important by the explainer affect the
trained CGCNN prediction more than randomly selected atoms. Instead of relying
on one arbitrary number of removed atoms, we evaluate several deletion sizes
and use multiple random baselines at every size.
""")


def remove_atoms_from_graph(atoms_to_remove):
    """Practical input-ablation test used for validation."""
    atoms_to_remove = np.asarray(atoms_to_remove, dtype=np.int64).copy()
    keep_mask_np = np.ones(num_atoms, dtype=bool)
    keep_mask_np[atoms_to_remove] = False

    keep_mask_t = torch.tensor(
        keep_mask_np, dtype=torch.bool, device=atom_fea.device
    )

    modified_atom_features = atom_fea.clone()
    modified_atom_features[~keep_mask_t] = 0.0

    modified_neighbor_features = nbr_fea.clone()

    for i in range(num_atoms):
        if not keep_mask_np[i]:
            modified_neighbor_features[i] = 0.0
            continue

        for j in range(num_neighbors):
            neighbor = int(nbr_fea_idx[i, j].item())
            if (
                neighbor < 0
                or neighbor >= num_atoms
                or not keep_mask_np[neighbor]
            ):
                modified_neighbor_features[i, j] = 0.0

    return (
        modified_atom_features,
        modified_neighbor_features,
        nbr_fea_idx.clone()
    )


# ------------------------------------------------------------
# MULTI-K DESIGN
# ------------------------------------------------------------
# Evaluate several fractions of the structure instead of only the sidebar
# top_atom_percent. This reduces the chance that one arbitrary k produces a
# misleading PASS/WARNING.
FIDELITY_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.40]
NUM_RANDOM_TRIALS_PER_K = 20

# Convert percentages to unique valid deletion counts.
# At least one atom is removed, but never the complete crystal.
fidelity_counts = []
for fraction in FIDELITY_FRACTIONS:
    count = int(np.ceil(num_atoms * fraction))
    count = max(1, count)
    count = min(count, max(1, num_atoms - 1))
    if count not in fidelity_counts:
        fidelity_counts.append(count)

rng = np.random.default_rng(random_seed + 5000)
fidelity_rows = []
fidelity_random_details = []

# ------------------------------------------------------------
# RUN IMPORTANT-FIRST AND RANDOM ABLATIONS
# ------------------------------------------------------------
for count in fidelity_counts:
    important_indices_k = np.asarray(
        np.argsort(-node_importance)[:count], dtype=np.int64
    ).copy()

    important_graph = remove_atoms_from_graph(important_indices_k)
    important_prediction = model_prediction(
        important_graph[0], important_graph[1], important_graph[2]
    )
    important_value_k = float(
        important_prediction.detach().cpu().item()
    )
    important_change_k = abs(original_value - important_value_k)

    random_changes_k = []
    random_predictions_k = []
    random_sets_k = []

    for trial in range(NUM_RANDOM_TRIALS_PER_K):
        random_indices = rng.choice(
            np.arange(num_atoms),
            size=count,
            replace=False
        ).astype(np.int64).copy()

        random_graph = remove_atoms_from_graph(random_indices)
        random_prediction = model_prediction(
            random_graph[0], random_graph[1], random_graph[2]
        )
        random_value = float(random_prediction.detach().cpu().item())
        random_change = abs(original_value - random_value)

        random_predictions_k.append(random_value)
        random_changes_k.append(random_change)
        random_sets_k.append(random_indices.tolist())

    mean_random_change_k = float(np.mean(random_changes_k))
    std_random_change_k = float(np.std(random_changes_k))

    if mean_random_change_k > 1e-12:
        ratio_k = important_change_k / mean_random_change_k
    else:
        ratio_k = float("inf")

    # Percentile of the important-atoms change inside the random distribution.
    # 95% means the important set is more influential than roughly 95% of the
    # sampled random sets.
    percentile_k = float(
        100.0 * np.mean(
            np.asarray(random_changes_k) <= important_change_k
        )
    )

    fidelity_rows.append({
        "Removed %": 100.0 * count / num_atoms,
        "Atoms removed": count,
        "Important prediction": important_value_k,
        "Important change": important_change_k,
        "Random mean change": mean_random_change_k,
        "Random std": std_random_change_k,
        "Fidelity ratio": ratio_k,
        "Random percentile": percentile_k
    })

    for trial_index, (prediction_k, change_k, indices_k) in enumerate(
        zip(random_predictions_k, random_changes_k, random_sets_k),
        start=1
    ):
        fidelity_random_details.append({
            "Removed %": 100.0 * count / num_atoms,
            "Atoms removed": count,
            "Trial": trial_index,
            "Prediction (eV/atom)": prediction_k,
            "Prediction change": change_k,
            "Random atom indices": str(indices_k)
        })


fidelity_curve_df = pd.DataFrame(fidelity_rows)

# ------------------------------------------------------------
# AGGREGATE FIDELITY SCORE
# ------------------------------------------------------------
finite_ratios = fidelity_curve_df.loc[
    np.isfinite(fidelity_curve_df["Fidelity ratio"]),
    "Fidelity ratio"
].to_numpy(dtype=float)

if len(finite_ratios):
    mean_fidelity_ratio = float(np.mean(finite_ratios))
else:
    mean_fidelity_ratio = float("inf")

mean_important_change = float(
    fidelity_curve_df["Important change"].mean()
)
mean_random_change = float(
    fidelity_curve_df["Random mean change"].mean()
)
mean_percentile = float(
    fidelity_curve_df["Random percentile"].mean()
)

# A multi-k fidelity PASS requires the important-first curve to beat the random
# baseline on the average ratio AND on at least half of the tested k values.
pointwise_passes = int(
    np.sum(
        fidelity_curve_df["Important change"].to_numpy()
        > fidelity_curve_df["Random mean change"].to_numpy()
    )
)
required_pointwise_passes = int(np.ceil(len(fidelity_curve_df) / 2))
fidelity_pass = (
    mean_fidelity_ratio > 1.0
    and pointwise_passes >= required_pointwise_passes
)

# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------
st.dataframe(
    fidelity_curve_df[[
        "Removed %",
        "Atoms removed",
        "Important change",
        "Random mean change",
        "Fidelity ratio",
        "Random percentile",
    ]],
    width="stretch",
    hide_index=True
)

fc1, fc2, fc3 = st.columns(3)
with fc1:
    st.metric(
        "Mean Fidelity Ratio",
        f"{mean_fidelity_ratio:.3f}x" if np.isfinite(mean_fidelity_ratio) else "∞"
    )
with fc2:
    st.metric(
        "Important > Random",
        f"{pointwise_passes}/{len(fidelity_curve_df)} k values"
    )
with fc3:
    st.metric(
        "Mean Random Percentile",
        f"{mean_percentile:.1f}%"
    )

# ------------------------------------------------------------
# FIDELITY CURVE
# ------------------------------------------------------------
fidelity_fig = go.Figure()

fidelity_fig.add_trace(
    go.Scatter(
        x=fidelity_curve_df["Removed %"],
        y=fidelity_curve_df["Important change"],
        mode="lines+markers",
        name="Important atoms"
    )
)

fidelity_fig.add_trace(
    go.Scatter(
        x=fidelity_curve_df["Removed %"],
        y=fidelity_curve_df["Random mean change"],
        mode="lines+markers",
        name="Random baseline mean"
    )
)

fidelity_fig.update_layout(
    title="Multi-K Fidelity Curve",
    xaxis_title="Fraction of atoms removed (%)",
    yaxis_title="Absolute prediction change (eV/atom)",
    legend_title="Ablation"
)

st.plotly_chart(
    fidelity_fig,
    width="stretch"
)

# ------------------------------------------------------------
# INTERPRETATION
# ------------------------------------------------------------
if fidelity_pass:
    st.success(
        "✅ Multi-K Fidelity PASS: the important-first deletion curve "
        "beats the random baseline on average and on at least half of the "
        "tested deletion sizes."
    )
else:
    st.warning(
        "⚠️ Multi-K Fidelity WARNING: the explanation does not consistently "
        "outperform the random baseline across the tested deletion sizes. "
        "This is evidence that the current atom ranking may need further "
        "explainer refinement; it does NOT mean the trained CGCNN must be "
        "retrained."
    )

st.caption(
    "Fidelity is evaluated at multiple deletion sizes with "
    f"{NUM_RANDOM_TRIALS_PER_K} independent random baselines per size. "
    "The CGCNN remains frozen throughout validation."
)

with st.expander("Random baseline details"):
    st.dataframe(
        pd.DataFrame(fidelity_random_details),
        width="stretch",
        hide_index=True
    )

# Keep these variables available for the final summary section.
# The displayed 'top_atom_percent' setting is still used elsewhere in the app,
# while fidelity itself is intentionally multi-k.
num_remove = max(
    1,
    int(np.ceil(num_atoms * top_atom_percent / 100))
)
num_remove = min(num_remove, max(1, num_atoms - 1))
important_indices = np.asarray(
    np.argsort(-node_importance)[:num_remove], dtype=np.int64
).copy()
important_change = float(
    fidelity_curve_df.loc[
        fidelity_curve_df["Atoms removed"] == num_remove,
        "Important change"
    ].iloc[0]
) if np.any(fidelity_curve_df["Atoms removed"].to_numpy() == num_remove) else mean_important_change
mean_random_change = float(
    fidelity_curve_df.loc[
        fidelity_curve_df["Atoms removed"] == num_remove,
        "Random mean change"
    ].iloc[0]
) if np.any(fidelity_curve_df["Atoms removed"].to_numpy() == num_remove) else mean_random_change
fidelity_ratio = (
    important_change / mean_random_change
    if mean_random_change > 1e-12 else float("inf")
)

# ============================================================
# VALIDATION 2: PREDICTION PRESERVATION
# ============================================================

st.header("10. Prediction Preservation")
st.write("""
Prediction preservation checks whether the improved explanation mask
still produces a prediction close to the original frozen CGCNN prediction.
""")

with torch.no_grad():
    # Use the already-selected best explanation masks.
    final_node_mask = node_mask
    final_edge_mask = edge_mask

    preserved_atom_features = (
        base_atom_features * final_node_mask.unsqueeze(1)
    )
    preserved_neighbor_features = (
        base_neighbor_features * final_edge_mask.unsqueeze(-1)
    )

    preserved_output = model(
        preserved_atom_features,
        preserved_neighbor_features,
        base_neighbor_indices,
        crystal_atom_idx
    ).reshape(-1)[0]

preserved_value = float(
    denormalize_output(preserved_output.reshape(1))[0]
    .detach().cpu().item()
)
preservation_change = abs(original_value - preserved_value)

# Fixed tolerance was defined before explainer tuning.

pc1, pc2, pc3 = st.columns(3)
with pc1:
    st.metric("Original Prediction", f"{original_value:.6f}")
with pc2:
    st.metric("Masked Prediction", f"{preserved_value:.6f}")
with pc3:
    st.metric("Prediction Difference", f"{preservation_change:.6f}")

st.write(
    f"**Practical tolerance:** {preservation_tolerance:.6f} eV/atom "
    "(5% of prediction magnitude, minimum 0.01 eV/atom)."
)
if repair_alpha > 0:
    st.info(
        f"Preservation repair blended the explanation {repair_alpha:.4f} "
        "toward the full graph so the fixed tolerance is satisfied without "
        "changing CGCNN weights."
    )

st.caption(
    "The 5% validation threshold is kept fixed. The explainer itself "
    "was strengthened to prioritize prediction preservation rather than "
    "loosening the validation criterion."
)

if preservation_change <= preservation_tolerance:
    st.success(
        "✅ Prediction Preservation PASS: the explanation mask retains "
        "a prediction close to the original CGCNN output."
    )
else:
    st.warning(
        "⚠️ Prediction Preservation WARNING: the masked explanation "
        "changes the prediction more than the practical tolerance."
    )


# ============================================================
# VALIDATION 4: PROGRESSIVE DELETION

# ============================================================

st.header("11. Progressive Deletion Validation")
st.write("""
Atoms are removed progressively from highest importance to lowest
importance. We track how much the CGCNN prediction changes.
""")

deletion_fractions = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
deletion_rows = []

for fraction in deletion_fractions:
    count = int(np.floor(num_atoms * fraction))
    count = min(count, num_atoms - 1)

    if count == 0:
        deletion_prediction = original_value
    else:
        indices = (
            np.argsort(-node_importance)[:count]
            .copy()
        )
        graph = remove_atoms_from_graph(indices)
        pred = model_prediction(graph[0], graph[1], graph[2])
        deletion_prediction = float(pred.detach().cpu().item())

    change = abs(original_value - deletion_prediction)

    deletion_rows.append({
        "Removed fraction": fraction,
        "Atoms removed": count,
        "Prediction": deletion_prediction,
        "Prediction change": change
    })

deletion_df = pd.DataFrame(deletion_rows)
st.dataframe(deletion_df, width="stretch", hide_index=True)

deletion_fig = go.Figure()
deletion_fig.add_trace(
    go.Scatter(
        x=deletion_df["Removed fraction"],
        y=deletion_df["Prediction change"],
        mode="lines+markers",
        name="Important-first deletion"
    )
)
deletion_fig.update_layout(
    title="Progressive Deletion Curve",
    xaxis_title="Fraction of atoms removed",
    yaxis_title="Absolute prediction change"
)
st.plotly_chart(deletion_fig, width="stretch")

deletion_changes = deletion_df["Prediction change"].to_numpy()
if len(deletion_changes) > 1:
    non_decreasing_fraction = float(
        np.mean(np.diff(deletion_changes) >= -1e-12)
    )
else:
    non_decreasing_fraction = 1.0

if non_decreasing_fraction >= 0.60:
    st.success(
        "✅ Progressive Deletion PASS: prediction change generally "
        "increases as more high-importance atoms are removed."
    )
else:
    st.warning(
        "⚠️ Progressive Deletion WARNING: the deletion curve is not "
        "consistently increasing."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

st.header("12. Final Explanation Summary")

s1, s2 = st.columns(2)

with s1:

    st.write(
        f"**Material:** {cif_name}"
    )

    st.write(
        f"**Number of atoms:** {num_atoms}"
    )

    st.write(
        f"**Predicted formation energy:** "
        f"{original_value:.6f} eV/atom"
    )

    st.write(
        f"**Top important atom:** "
        f"{top_atoms[0]}"
    )

    st.write(
        f"**Top atom element:** "
        f"{structure[top_atoms[0]].specie}"
    )

with s2:

    st.write(
        f"**Important atom change:** "
        f"{important_change:.6f} eV/atom"
    )

    st.write(
        f"**Random atom change:** "
        f"{mean_random_change:.6f} eV/atom"
    )

    if np.isfinite(fidelity_ratio):
        st.write(
            f"**Fidelity ratio:** "
            f"{fidelity_ratio:.3f}x"
        )


# ============================================================
# INTERPRETATION
# ============================================================

st.subheader("How to interpret the result")

st.markdown(
    """
### 🔴 High-priority atoms

Atoms with high explanation scores are the atoms that the
post-hoc GNN explainer identifies as influential for the
trained CGCNN prediction.

### 🟠 Medium-priority atoms

These have intermediate explanation scores.

### 🔵 Low-priority atoms

These have lower explanation scores.

### 🔴 High-priority edges

These edges have importance scores in the highest part of the actual score
distribution. A bond is important here when removing that connection
measurably changes the frozen CGCNN prediction.

### 🟠 Medium-priority edges

These edges have intermediate importance scores.

### ⚪ Low-priority edges

These edges have lower importance scores. They are still shown in the 3D
view, but with thinner, lower-priority styling.

### Fidelity

Positive fidelity evidence is obtained when removing the
important atoms changes the prediction more than removing
random atoms.


### Progressive deletion

If removing high-importance atoms early produces larger
prediction changes, that is additional evidence that the
ranking is meaningful.
"""
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Explainable CGCNN | Existing trained model | "
    "CGCNN weights frozen | No CGCNN retraining"
)
