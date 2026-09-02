from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


FEATURE_COLUMNS = [
    "Daily_Return",
    "EMA_20",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Momentum_10",
    "Rolling_Volatility_20",
]
TARGET_COLUMN = "Target_Return"

FEATURE_NAMES = {
    "Daily_Return": "Daily Return",
    "EMA_20": "the 20-day EMA",
    "RSI_14": "RSI",
    "MACD": "MACD",
    "MACD_Signal": "the MACD signal line",
    "MACD_Histogram": "the MACD histogram",
    "Momentum_10": "momentum",
    "Rolling_Volatility_20": "volatility",
}


def phrase_feature(feature: str, shap_value: float, feature_value: float, threshold: float) -> str:
    name = FEATURE_NAMES.get(feature, feature)
    magnitude = "high" if feature_value >= threshold else "low"
    if shap_value > 0:
        if magnitude == "high":
            return f"High {name.lower()} increased the predicted return."
        return f"Low {name.lower()} increased the predicted return."
    if shap_value < 0:
        if magnitude == "high":
            return f"High {name.lower()} reduced the predicted return."
        return f"Low {name.lower()} reduced the predicted return."
    return f"{name.title()} had little effect on the predicted return."


def run_explanation_report(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    processed_path = Path(config["data"]["processed_path"])
    experiments_path = Path(config.get("experiments", {}).get("path", "experiments"))
    results_path = Path(config.get("results", {}).get("path", "results"))

    tables_dir = results_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    test_file = processed_path / "test.csv"
    shap_rf_file = experiments_path / "shap" / "random_forest_shap_values.csv"
    shap_xgb_file = experiments_path / "shap" / "xgboost_shap_values.csv"
    tabpfn_imp_file = tables_dir / "tabpfn_permutation_importance.csv"

    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return

    test = pd.read_csv(test_file)
    test = test.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)

    thresholds = test[FEATURE_COLUMNS].median()

    rows = []
    for idx, row in test.iterrows():
        date = row["Date"]
        ticker = row["Ticker"]

        if shap_rf_file.exists():
            shap_rf = pd.read_csv(shap_rf_file)
            if len(shap_rf) == len(test):
                shap_row = shap_rf.iloc[idx]
                top = shap_row.reindex(FEATURE_COLUMNS).abs().nlargest(3).index
                sentences = [
                    phrase_feature(f, shap_row[f], row[f], thresholds[f]) for f in top
                ]
                rows.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Model": "RandomForest",
                    "Explanation": " ".join(sentences),
                })
            else:
                print("RF SHAP row count mismatch; skipping.")
        else:
            print(f"RF SHAP file not found: {shap_rf_file}")

        if shap_xgb_file.exists():
            shap_xgb = pd.read_csv(shap_xgb_file)
            if len(shap_xgb) == len(test):
                shap_row = shap_xgb.iloc[idx]
                top = shap_row.reindex(FEATURE_COLUMNS).abs().nlargest(3).index
                sentences = [
                    phrase_feature(f, shap_row[f], row[f], thresholds[f]) for f in top
                ]
                rows.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Model": "XGBoost",
                    "Explanation": " ".join(sentences),
                })
            else:
                print("XGBoost SHAP row count mismatch; skipping.")
        else:
            print(f"XGBoost SHAP file not found: {shap_xgb_file}")

        if tabpfn_imp_file.exists():
            importance = pd.read_csv(tabpfn_imp_file)
            top = importance.sort_values("Importance_Mean", ascending=False).head(3)["Feature"].tolist()
            sentences = [
                f"{FEATURE_NAMES.get(f, f)} was among the most important features for TabPFN."
                for f in top
            ]
            rows.append({
                "Date": date,
                "Ticker": ticker,
                "Model": "TabPFN",
                "Explanation": " ".join(sentences),
            })
        else:
            print(f"TabPFN importance file not found: {tabpfn_imp_file}")

    if not rows:
        print("No explanations generated.")
        return

    report = pd.DataFrame(rows)
    output_file = tables_dir / "prediction_explanations.csv"
    report.to_csv(output_file, index=False)
    print(f"Saved prediction explanations to {output_file}")
    print(report.head().to_string(index=False))


if __name__ == "__main__":
    run_explanation_report()
