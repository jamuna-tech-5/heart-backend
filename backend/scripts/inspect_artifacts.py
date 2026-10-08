"""Inspect CAD model artifacts. Run before implementing or changing the API.

Prints metadata, encoder types/classes, model type/classes_, feature count/order,
and library versions. Does not train or modify any artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

EXCLUDED_FEATURES = ("Cath", "LAD", "LCX", "RCA")


def detect_loader(path: Path) -> str:
    raw = path.read_bytes()
    if b"joblib.numpy_pickle" in raw:
        return "joblib"
    return "pickle"


def load_artifact(path: Path):
    loader = detect_loader(path)
    if loader == "joblib":
        return joblib.load(path), loader
    with path.open("rb") as handle:
        return pickle.load(handle), loader


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def feature_table_markdown(feature_names: list[str], encoders: dict) -> str:
    lines = [
        "# CAD model feature contract",
        "",
        "Generated from `CAD_XGBoost_Metadata.pkl` and `CAD_Categorical_Encoders.pkl`.",
        "Numeric ranges are **NEEDS_CONFIRMATION** (not present in metadata).",
        "Missing-value handling is **NEEDS_CONFIRMATION**; default: all 52 features required.",
        "",
        "| # | Feature | Data type | Categorical / Numeric | Allowed values / range | Required |",
        "|---|---------|-----------|------------------------|------------------------|----------|",
    ]
    for index, name in enumerate(feature_names, start=1):
        if name in encoders:
            classes = [str(c) for c in encoders[name].classes_]
            allowed = ", ".join(f"`{c}`" for c in classes)
            row_type = "string"
            kind = "categorical"
        else:
            allowed = "NEEDS_CONFIRMATION (not in metadata)"
            row_type = "number"
            kind = "numeric"
        lines.append(
            f"| {index} | `{name}` | {row_type} | {kind} | {allowed} | Yes |"
        )
    lines.extend(
        [
            "",
            "## Excluded (never prediction inputs)",
            "",
            "`Cath`, `LAD`, `LCX`, `RCA` (and any other key not in the 52-feature list).",
            "",
            "## Open items for the team",
            "",
            "1. Confirm numeric valid ranges.",
            "2. Confirm missing values are rejected (current default: all 52 required).",
            "3. Confirm vessel models (LAD/LCX/RCA) do not exist in this workspace.",
            "4. Provide one real dataset row for the golden `predict_proba` test.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        default=str(
            Path(__file__).resolve().parents[1] / "app" / "ml" / "artifacts"
        ),
    )
    parser.add_argument(
        "--write-features",
        default=str(Path(__file__).resolve().parents[1] / "docs" / "FEATURES.md"),
        help="Path to write FEATURES.md. Empty string to skip.",
    )
    args = parser.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    model_path = artifacts_dir / "CAD_XGBoost_Model.pkl"
    metadata_path = artifacts_dir / "CAD_XGBoost_Metadata.pkl"
    encoders_path = artifacts_dir / "CAD_Categorical_Encoders.pkl"

    print("=== Library versions (runtime) ===")
    print(f"python: {sys.version}")
    print(f"xgboost: {xgboost.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"joblib: {joblib.__version__}")
    print()

    print("=== Artifact files ===")
    for path in (model_path, metadata_path, encoders_path):
        print(
            f"{path.name}: exists={path.exists()} size={path.stat().st_size if path.exists() else 'MISSING'} "
            f"sha256={sha256(path) if path.exists() else 'n/a'} detected_loader={detect_loader(path) if path.exists() else 'n/a'}"
        )
    print()

    metadata, metadata_loader = load_artifact(metadata_path)
    print("=== Metadata ===")
    print(f"loader: {metadata_loader}")
    print(f"type: {type(metadata)}")
    if not isinstance(metadata, dict):
        print(metadata)
        return 1
    for key, value in metadata.items():
        print(f"{key}: {value}")
    feature_names = list(metadata["feature_names"])
    print(f"feature_count: {len(feature_names)}")
    excluded_present = [name for name in feature_names if name in EXCLUDED_FEATURES]
    print(f"excluded_features_in_list: {excluded_present}")
    print()

    encoders, encoders_loader = load_artifact(encoders_path)
    print("=== Encoders ===")
    print(f"loader: {encoders_loader}")
    print(f"type: {type(encoders)}")
    print(f"encoder_count: {len(encoders)}")
    sklearn_versions = set()
    for name, encoder in encoders.items():
        classes = [repr(c) for c in list(encoder.classes_)]
        sk_ver = getattr(encoder, "_sklearn_version", None)
        if sk_ver:
            sklearn_versions.add(sk_ver)
        print(
            f"- {name}: type={type(encoder).__module__}.{type(encoder).__name__} "
            f"sklearn={sk_ver} classes={classes}"
        )
        # round-trip each class
        for cls in encoder.classes_:
            encoded = encoder.transform([cls])
            decoded = encoder.inverse_transform(encoded)
            if decoded[0] != cls:
                print(f"  ROUNDTRIP FAIL {name} {cls!r} -> {encoded} -> {decoded}")
    print(f"encoder_sklearn_versions: {sorted(sklearn_versions)}")
    missing_encoders = [name for name in encoders if name not in feature_names]
    print(f"encoders_not_in_feature_list: {missing_encoders}")
    cats_without_encoder = [
        name for name in feature_names if name in encoders
    ]
    print(f"categorical_features_from_encoders: {cats_without_encoder}")
    print()

    model, model_loader = load_artifact(model_path)
    print("=== Model ===")
    print(f"loader: {model_loader}")
    print(f"type: {type(model).__module__}.{type(model).__name__}")
    print(f"classes_: {getattr(model, 'classes_', None)}")
    print(f"n_classes_: {getattr(model, 'n_classes_', None)}")
    print(f"n_features_in_: {getattr(model, 'n_features_in_', None)}")
    print(f"feature_names_in_: {getattr(model, 'feature_names_in_', None)}")
    print(f"objective: {getattr(model, 'objective', None)}")
    print(f"n_estimators: {getattr(model, 'n_estimators', None)}")
    booster = model.get_booster()
    print(f"booster.num_feature: {booster.num_feature()}")
    print(f"booster.feature_names: {booster.feature_names}")
    print()

    target_mapping = metadata.get("target_mapping", {})
    print("=== Target mapping vs model.classes_ ===")
    print(f"target_mapping: {target_mapping}")
    print(f"model.classes_: {list(model.classes_)}")
    print()

    if args.write_features:
        out = Path(args.write_features)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(feature_table_markdown(feature_names, encoders), encoding="utf-8")
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
