import argparse
import difflib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models.finetune_model import standard_finetune
from suiren_datasets.org_mol2d import from_smiles


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
ALLOWED_ELEMENTS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for a single SMILES string or a CSV file (interactive mode)."
    )
    parser.add_argument(
        "task_pos",
        nargs="?",
        default=None,
        help="Task name. Usage: python inference.py aqsol",
    )
    parser.add_argument(
        "--task",
        dest="task_flag",
        type=str,
        default=None,
        help="Task name under checkpoints/, for example aqsol or bbb.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Exact checkpoint path. If omitted, the latest checkpoint under checkpoints/<task>/ is used.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=str(DEFAULT_CHECKPOINT_ROOT),
        help="Root directory of downstream checkpoints.",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default=None,
        help="CSV column name containing SMILES. If omitted, the script auto-detects it.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size for CSV input.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path. Results are printed to console if not provided.",
    )

    args = parser.parse_args()


    args.task = args.task_flag or args.task_pos
    args.input = None  # 统一使用交互式输入

    if args.task is None and args.checkpoint is None:
        parser.error("Must provide a task name or --checkpoint.")

    return args


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("No available CUDA device in the current environment.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_torch_file(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_state_dict(checkpoint_obj) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["state_dict"]
        meta = checkpoint_obj
    elif isinstance(checkpoint_obj, dict):
        state_dict = checkpoint_obj
        meta = checkpoint_obj
    else:
        raise TypeError("Unsupported checkpoint format.")

    normalized = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        normalized[new_key] = value
    return normalized, meta


def infer_task_type(state_dict: Dict[str, torch.Tensor], checkpoint_path: Path) -> Tuple[str, int]:
    head_key = "proj_2d_glob.2.weight"
    if head_key in state_dict:
        output_dim = int(state_dict[head_key].shape[0])
        if output_dim == 1:
            return "regression", 1
        return "classification", output_dim

    checkpoint_name = checkpoint_path.name.lower()
    if "classification" in checkpoint_name:
        return "classification", 2
    return "regression", 1


def list_available_tasks(checkpoint_root: Path) -> List[str]:
    if not checkpoint_root.is_dir():
        return []
    return sorted(path.name for path in checkpoint_root.iterdir() if path.is_dir())


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    if not args.task:
        raise ValueError("Must provide either --checkpoint or --task.")

    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    task_dir = checkpoint_root / args.task
    if not task_dir.is_dir():
        available_tasks = list_available_tasks(checkpoint_root)
        similar_tasks = difflib.get_close_matches(args.task, available_tasks, n=5, cutoff=0.5)
        message = f"No matching property: {args.task}"
        if similar_tasks:
            message += f"\nDid you mean: {', '.join(similar_tasks)}"
        raise FileNotFoundError(message)

    candidates = [
        path for path in task_dir.rglob("*.pt")
        if not path.name.endswith("_ema.pt")
    ]
    if not candidates:
        raise FileNotFoundError(f"No available checkpoint found under {task_dir}.")

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def infer_task_name(checkpoint_path: Path) -> str:
    if checkpoint_path.parent.parent.name:
        return checkpoint_path.parent.parent.name

    stem = checkpoint_path.stem
    stem = stem.replace("_classification", "").replace("_regression", "")
    return stem


def to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint_obj = load_torch_file(checkpoint_path)
    state_dict, meta = normalize_state_dict(checkpoint_obj)
    task_type, class_num = infer_task_type(state_dict, checkpoint_path)

    model = standard_finetune(
        class_flag=(task_type == "classification"),
        class_num=class_num if task_type == "classification" else 2,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    norm_factor = None
    if task_type == "regression" and isinstance(meta, dict) and "norm_factor" in meta:
        norm_values = meta["norm_factor"]
        if isinstance(norm_values, (list, tuple)) and len(norm_values) == 2:
            mean, std = norm_values
            norm_factor = (to_float(mean), to_float(std))


    return model, task_type, class_num, norm_factor


def detect_smiles_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"Specified column does not exist in CSV: {preferred}")
        return preferred

    exact_candidates = [
        "SMILES",
        "smiles",
        "Smiles",
        "canonical_smiles",
        "Canonical_SMILES",
    ]
    for column in exact_candidates:
        if column in df.columns:
            return column

    fuzzy_candidates = [column for column in df.columns if "smiles" in str(column).lower()]
    if len(fuzzy_candidates) == 1:
        return fuzzy_candidates[0]

    if len(df.columns) == 1:
        return df.columns[0]

    raise ValueError(
        "Unable to auto-detect SMILES column, please use --smiles-column to specify explicitly."
    )


def looks_like_csv_path(input_value: str) -> bool:
    return Path(input_value).suffix.lower() == ".csv"


def load_inputs(input_value: str, smiles_column: Optional[str]) -> Tuple[str, pd.DataFrame, str, Optional[Path]]:
    input_path = Path(input_value).expanduser()

    if input_path.is_file():
        df = pd.read_csv(input_path)
        smiles_col = detect_smiles_column(df, smiles_column)
        return "csv", df.copy(), smiles_col, input_path.resolve()

    if looks_like_csv_path(input_value):
        raise FileNotFoundError(f"Input CSV not found: {input_path.resolve()}")

    df = pd.DataFrame({"SMILES": [input_value]})
    return "smiles", df, "SMILES", None


def build_graph(smiles: str) -> Tuple[Optional[Data], Optional[str]]:
    if pd.isna(smiles):
        return None, "empty_smiles"

    smiles = str(smiles).strip()
    if not smiles:
        return None, "empty_smiles"

    graph_tuple, mol_flag = from_smiles(smiles, with_hydrogen=True)
    if not mol_flag:
        return None, "invalid_smiles"

    x, edge_index, edge_attr, edge_index_all = graph_tuple
    atom_types = set(x[:, 0].tolist())
    if not atom_types.issubset(ALLOWED_ELEMENTS):
        return None, "unsupported_elements"

    data = Data(
        x=x.to(torch.long),
        edge_index=edge_index.to(torch.long),
        edge_attr=edge_attr.to(torch.long),
        edge_index_all=edge_index_all.to(torch.long),
        smiles=smiles,
    )
    return data, None


def run_inference(
    model: torch.nn.Module,
    task_type: str,
    class_num: int,
    norm_factor: Optional[Tuple[float, float]],
    data_list: Sequence[Data],
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, object]]:
    loader = DataLoader(list(data_list), batch_size=batch_size, shuffle=False)
    outputs: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)

            if task_type == "classification":
                labels = logits.argmax(dim=-1).detach().cpu()
                for row_label in labels:
                    outputs.append({"pred_label": int(row_label.item())})
            else:
                preds = logits.view(-1).detach().cpu()
                if norm_factor is not None:
                    mean, std = norm_factor
                    preds = preds * std + mean
                for pred in preds:
                    outputs.append({"prediction": float(pred.item())})

    return outputs


def attach_predictions(
    df: pd.DataFrame,
    records: List[Optional[Dict[str, object]]],
) -> pd.DataFrame:
    result_df = df.copy()
    result_df["status"] = "invalid"
    result_df["error"] = None

    discovered_columns: List[str] = []
    for record in records:
        if record:
            for key in record.keys():
                if key not in {"status", "error"} and key not in discovered_columns:
                    discovered_columns.append(key)

    for column in discovered_columns:
        result_df[column] = None

    for idx, record in enumerate(records):
        if not record:
            continue
        result_df.at[idx, "status"] = record.get("status", "invalid")
        result_df.at[idx, "error"] = record.get("error")
        for key, value in record.items():
            if key in {"status", "error"}:
                continue
            result_df.at[idx, key] = value

    return result_df


def main() -> None:
    args = parse_args()
    
    if args.input is None:
        try:
            args.input = input("Enter SMILES or CSV path: ").strip()
        except EOFError:
            raise ValueError("No input provided.")
        if not args.input:
            raise ValueError("Input cannot be empty.")
    
    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint(args)
    task_name = args.task or infer_task_name(checkpoint_path)

    model, task_type, class_num, norm_factor = load_model(checkpoint_path, device)
    input_kind, input_df, smiles_column, input_path = load_inputs(args.input, args.smiles_column)

    data_list: List[Data] = []
    valid_indices: List[int] = []
    records: List[Optional[Dict[str, object]]] = [None] * len(input_df)

    for idx, smiles in enumerate(input_df[smiles_column].tolist()):
        graph, error = build_graph(smiles)
        if graph is None:
            records[idx] = {"status": "invalid", "error": error}
            continue

        data_list.append(graph)
        valid_indices.append(idx)

    if data_list:
        predictions = run_inference(
            model=model,
            task_type=task_type,
            class_num=class_num,
            norm_factor=norm_factor,
            data_list=data_list,
            device=device,
            batch_size=args.batch_size,
        )
        for row_idx, pred in zip(valid_indices, predictions):
            pred["status"] = "ok"
            pred["error"] = None
            records[row_idx] = pred

    result_df = attach_predictions(input_df, records)

    if input_kind == "smiles":
        row = result_df.iloc[0]
        result = {"SMILES": row.get(smiles_column)}
        if task_type == "regression":
            prediction = row.get("prediction")
            result["prediction"] = None if pd.isna(prediction) else float(prediction)
        else:
            pred_label = row.get("pred_label")
            result["pred_label"] = None if pd.isna(pred_label) else int(pred_label)

        message = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
            output_path.write_text(message, encoding="utf-8")
            print(f"Results saved to: {output_path}")
        else:
            print(message)
        return

    value_column: List[Optional[object]] = []
    value_key = "prediction" if task_type == "regression" else "pred_label"
    for _, row in result_df.iterrows():
        value = row.get(value_key)
        if pd.isna(value):
            value_column.append(None)
        elif task_type == "regression":
            value_column.append(float(value))
        else:
            value_column.append(int(value))

    output_df = input_df.copy()
    output_df["value"] = value_column

    output_path = Path(args.output).expanduser().resolve() if args.output else input_path
    if output_path is None:
        raise ValueError("No writable source file path found for CSV input.")

    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Task: {task_name}")
    print(f"Type: {task_type}")
    print(f"Total rows: {len(output_df)}")
    print(f"Result file: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, TypeError) as exc:
        print(str(exc))
        raise SystemExit(1)

