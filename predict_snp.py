#!/usr/bin/env python3
"""計算 TPMI PRS，並使用含 SNP 特徵的 SVM 預測骨質狀態。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

PRS_MODEL_FEATURE = "TPMI_Osteoporosis_osteopenia and pathological fracture"
PRS_OUTPUT_COLUMN = "TPMI_prs"

CATEGORICAL_FEATURES = ["MENSTRURATION_CURR"]
NUMERIC_INPUT_FEATURES = [
    "AGE",
    "FASTING_GLUCOSE",
    "SGPT",
    "HB",
    "TG",
    "HCT",
    "BMI",
    "BODY_WEIGHT",
]
SCALER_FEATURES = NUMERIC_INPUT_FEATURES + [PRS_MODEL_FEATURE]
REQUIRED_INPUT_FEATURES = CATEGORICAL_FEATURES + NUMERIC_INPUT_FEATURES

PRS_COLUMNS = [
    "rs_id",
    "effect_size",
    "ALT_allele(A1)",
    "ref_allele(A2)",
    "source",
]

DIAGNOSIS = {
    0: "正常",
    1: "骨鬆",
    2: "輕微骨鬆",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="從 SNP 基因型計算 TPMI PRS，並以 SVM 預測骨質狀態。"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=BASE_DIR / "snp_example.csv",
        help="輸入 CSV（預設：snp_example.csv）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=BASE_DIR / "snp_predictions.csv",
        help="輸出 CSV（預設：snp_predictions.csv）",
    )
    parser.add_argument(
        "--prs",
        type=Path,
        default=BASE_DIR / "prs.txt",
        help="PRS 權重檔（預設：prs.txt）",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=BASE_DIR / "model",
        help="模型目錄（預設：model）",
    )
    return parser.parse_args()


def read_prs_file(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 PRS 權重檔：{path}")

    prs_df = pd.read_csv(path, sep="\t", dtype=str)
    missing_columns = [column for column in PRS_COLUMNS if column not in prs_df.columns]
    if missing_columns:
        raise ValueError("PRS 權重檔缺少欄位：" + ", ".join(missing_columns))

    prs_df = prs_df.loc[prs_df["source"] == PRS_MODEL_FEATURE, PRS_COLUMNS].copy()
    if prs_df.empty:
        raise ValueError(f"PRS 權重檔沒有模型需要的 source：{PRS_MODEL_FEATURE}")

    prs_df["effect_size"] = pd.to_numeric(prs_df["effect_size"], errors="coerce")
    if prs_df["effect_size"].isna().any():
        bad_rows = (prs_df.index[prs_df["effect_size"].isna()] + 2).tolist()
        raise ValueError(f"PRS 權重檔的 effect_size 不是數值，列號：{bad_rows}")

    duplicate_rs = prs_df.loc[prs_df["rs_id"].duplicated(), "rs_id"].unique().tolist()
    if duplicate_rs:
        raise ValueError("PRS 權重檔含重複 rs_id：" + ", ".join(duplicate_rs))

    return prs_df


def calculate_tpmi_prs(input_df: pd.DataFrame, prs_df: pd.DataFrame) -> pd.Series:
    """依 notebook 邏輯計算 sum(效果等位基因數量 * effect_size)。"""
    required_rs_ids = prs_df["rs_id"].tolist()
    missing_rs_ids = [rs_id for rs_id in required_rs_ids if rs_id not in input_df.columns]
    if missing_rs_ids:
        raise ValueError("輸入 CSV 缺少 PRS SNP 欄位：" + ", ".join(missing_rs_ids))

    prs_values = pd.Series(0.0, index=input_df.index, dtype=float)
    errors: list[str] = []

    for rule in prs_df.itertuples(index=False, name=None):
        rs_id, effect_size, alt_allele, ref_allele, _source = rule
        valid_alleles = {alt_allele, ref_allele}
        alt_counts = []

        for row_index, genotype in input_df[rs_id].items():
            # 與 notebook 相同：缺失基因型的效果等位基因數量視為 0。
            if pd.isna(genotype) or not str(genotype).strip():
                alt_counts.append(0)
                continue

            alleles = [allele.strip() for allele in str(genotype).split("_")]
            if len(alleles) != 2 or any(allele not in valid_alleles for allele in alleles):
                errors.append(
                    f"第 {row_index + 2} 列 {rs_id}={genotype!r}；"
                    f"應為兩個等位基因，以底線分隔，可用值：{sorted(valid_alleles)}"
                )
                alt_counts.append(0)
                continue

            alt_counts.append(sum(allele == alt_allele for allele in alleles))

        prs_values += pd.Series(alt_counts, index=input_df.index) * float(effect_size)

    if errors:
        preview = "\n".join(errors[:10])
        if len(errors) > 10:
            preview += f"\n另有 {len(errors) - 10} 個基因型錯誤"
        raise ValueError("基因型格式錯誤：\n" + preview)

    return prs_values


def prepare_model_input(
    input_df: pd.DataFrame, prs_values: pd.Series, scaler: object
) -> np.ndarray:
    missing_columns = [
        column for column in REQUIRED_INPUT_FEATURES if column not in input_df.columns
    ]
    if missing_columns:
        raise ValueError("輸入 CSV 缺少欄位：" + ", ".join(missing_columns))

    features = input_df.loc[:, REQUIRED_INPUT_FEATURES].copy()
    for column in REQUIRED_INPUT_FEATURES:
        features[column] = features[column].str.replace(r"\s+", "", regex=True)
        features[column] = features[column].replace("", np.nan)

    # 與 notebook 的訓練前處理一致。
    features["SGPT"] = features["SGPT"].replace({"<5": "0"})

    numeric_features = features.apply(pd.to_numeric, errors="coerce")
    invalid_mask = numeric_features.isna()
    if invalid_mask.any(axis=None):
        details = []
        for row_index, column_index in zip(*np.where(invalid_mask.to_numpy())):
            column = numeric_features.columns[column_index]
            original_value = input_df.iloc[row_index][column]
            details.append(f"第 {row_index + 2} 列 {column}={original_value!r}")
        preview = "；".join(details[:10])
        if len(details) > 10:
            preview += f"；另有 {len(details) - 10} 個錯誤"
        raise ValueError("欄位含有缺值或非數值：" + preview)

    scaler_input = numeric_features[NUMERIC_INPUT_FEATURES].copy()
    scaler_input[PRS_MODEL_FEATURE] = prs_values

    scaler_feature_names = list(getattr(scaler, "feature_names_in_", []))
    if scaler_feature_names and scaler_feature_names != SCALER_FEATURES:
        raise ValueError(
            "scaler 的欄位順序與程式不符：" + ", ".join(scaler_feature_names)
        )

    scaled_numeric = scaler.transform(scaler_input[SCALER_FEATURES])
    menstruation = numeric_features[CATEGORICAL_FEATURES].to_numpy(dtype=np.float32)

    # 模型訓練順序：MENSTRURATION_CURR + 9 個正規化數值特徵。
    return np.concatenate([menstruation, scaled_numeric], axis=1)


def load_artifact(path: Path, artifact_name: str) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"找不到{artifact_name}：{path}")
    return joblib.load(path)


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

    prs_df = read_prs_file(args.prs.resolve())
    prs_values = calculate_tpmi_prs(input_df, prs_df)

    scaler = load_artifact(model_dir / "sub_snp_feature_oste_scaler.pkl", "scaler")
    model = load_artifact(
        model_dir / "sub_snp_feature_svm(3class).pkl", "SVM 模型"
    )

    model_input = prepare_model_input(input_df, prs_values, scaler)
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
    output_df[PRS_OUTPUT_COLUMN] = prs_values
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
