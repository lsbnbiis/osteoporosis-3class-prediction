#!/usr/bin/env python3
"""使用無 SNP 的 3 分類 ensemble 模型預測骨質狀態。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

CATEGORICAL_FEATURES = ["MENSTRURATION_CURR"]
NUMERIC_FEATURES = [
    "AGE",
    "FASTING_GLUCOSE",
    "SGPT",
    "GAMMA_GT",
    "BMI",
    "MICROALB",
    "RBC",
    "BODY_WEIGHT",
    "T_BILIRUBIN",
]
REQUIRED_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# 與模型訓練時相同的檢驗值特殊處理規則。
BELOW_DETECTION_LIMITS = {
    "SGPT": {"<5": "0"},
    "GAMMA_GT": {"<3.3": "0"},
    "MICROALB": {"<2.09": "0"},
}

DIAGNOSIS = {
    0: "正常",
    1: "骨鬆",
    2: "輕微骨鬆",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="讀取 CSV，正規化數值特徵，並預測骨質狀態。"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=BASE_DIR / "nonsnp_example.csv",
        help="輸入 CSV（預設：nonsnp_example.csv）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=BASE_DIR / "nonsnp_predictions.csv",
        help="輸出 CSV（預設：nonsnp_predictions.csv）",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=BASE_DIR / "model",
        help="模型目錄（預設：model）",
    )
    return parser.parse_args()


def prepare_features(input_df: pd.DataFrame, scaler: object) -> np.ndarray:
    """驗證並依訓練流程建立模型使用的 10 維特徵。"""
    missing_columns = [c for c in REQUIRED_FEATURES if c not in input_df.columns]
    if missing_columns:
        raise ValueError("輸入 CSV 缺少欄位：" + ", ".join(missing_columns))

    features = input_df.loc[:, REQUIRED_FEATURES].copy()

    # 去除字串前後與中間空白，再套用訓練資料的特殊值轉換。
    for column in REQUIRED_FEATURES:
        features[column] = features[column].str.replace(r"\s+", "", regex=True)
        features[column] = features[column].replace("", np.nan)

    for column, replacements in BELOW_DETECTION_LIMITS.items():
        features[column] = features[column].replace(replacements)

    numeric_features = features.apply(pd.to_numeric, errors="coerce")
    invalid_mask = numeric_features.isna()
    if invalid_mask.any(axis=None):
        details = []
        for row_index, column in zip(*np.where(invalid_mask.to_numpy())):
            original_value = input_df.iloc[row_index][column]
            details.append(f"第 {row_index + 2} 列 {column}={original_value!r}")
        preview = "；".join(details[:10])
        if len(details) > 10:
            preview += f"；另有 {len(details) - 10} 個錯誤"
        raise ValueError("欄位含有缺值或非數值：" + preview)

    scaler_feature_names = list(getattr(scaler, "feature_names_in_", []))
    if scaler_feature_names and scaler_feature_names != NUMERIC_FEATURES:
        raise ValueError(
            "scaler 的欄位順序與程式不符：" + ", ".join(scaler_feature_names)
        )

    scaled_numeric = scaler.transform(numeric_features[NUMERIC_FEATURES])
    menstruation = numeric_features[CATEGORICAL_FEATURES].to_numpy(dtype=np.float32)

    # 模型訓練時的順序：MENSTRURATION_CURR + 9 個正規化數值特徵。
    return np.concatenate([menstruation, scaled_numeric], axis=1)


def load_artifact(path: Path, artifact_name: str) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"找不到{artifact_name}：{path}")
    try:
        return joblib.load(path)
    except (TypeError, ValueError) as exc:
        if np.lib.NumpyVersion(np.__version__) < "2.0.0":
            raise RuntimeError(
                f"無法載入{artifact_name}；此模型需要 NumPy 2.x，"
                f"目前版本為 {np.__version__}。請先安裝 requirements-prediction.txt。"
            ) from exc
        raise


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    model_dir = args.model_dir.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"找不到輸入檔：{input_path}")

    input_df = pd.read_csv(input_path, dtype=str)
    if input_df.empty:
        raise ValueError("輸入 CSV 沒有任何資料列。")

    scaler = load_artifact(model_dir / "sub_feature_oste_scaler.pkl", "scaler")
    model = load_artifact(
        model_dir / "sub_feature_nosnp_ensemble(3class).pkl", "ensemble 模型"
    )

    model_input = prepare_features(input_df, scaler)
    expected_feature_count = getattr(model, "n_features_in_", model_input.shape[1])
    if model_input.shape[1] != expected_feature_count:
        raise ValueError(
            f"模型需要 {expected_feature_count} 個特徵，"
            f"但程式建立了 {model_input.shape[1]} 個。"
        )

    predictions = np.asarray(model.predict(model_input), dtype=int)
    unknown_labels = sorted(set(predictions) - set(DIAGNOSIS))
    if unknown_labels:
        raise ValueError(f"模型輸出未知類別：{unknown_labels}")

    output_df = input_df.copy()
    output_df["PREDICTION"] = predictions
    output_df["DIAGNOSIS"] = [DIAGNOSIS[label] for label in predictions]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    counts = pd.Series(predictions).value_counts().sort_index()
    print(f"已完成 {len(output_df)} 筆預測，輸出至：{output_path}")
    for label, diagnosis in DIAGNOSIS.items():
        print(f"{label}（{diagnosis}）：{int(counts.get(label, 0))} 筆")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        sys.exit(1)
