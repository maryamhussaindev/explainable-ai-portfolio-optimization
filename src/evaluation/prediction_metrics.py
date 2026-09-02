from pathlib import Path

import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


MODEL_FILES = {
    "RandomForest": ("experiments", "rf_predictions.csv"),
    "XGBoost": ("experiments", "xgb_predictions.csv"),
    "TabPFN": ("experiments", "tabpfn_predictions.csv"),
}

ACTUAL_COLUMN = "Actual_Return"
PREDICTED_COLUMN = "Predicted_Return"

COLORS = {
    "MAE": (66, 133, 244),
    "RMSE": (234, 67, 53),
}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_model_metrics(df: pd.DataFrame) -> dict:
    actual = df[ACTUAL_COLUMN]
    predicted = df[PREDICTED_COLUMN]
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": root_mean_squared_error(actual, predicted),
    }


def _draw_bar_chart(comparison: pd.DataFrame, output_path: Path) -> None:
    models = comparison["Model"].tolist()
    metrics = ["MAE", "RMSE"]
    n_models = len(models)

    img_w, img_h = 900, 500
    margin_l, margin_r, margin_t, margin_b = 80, 30, 60, 80
    chart_w = img_w - margin_l - margin_r
    chart_h = img_h - margin_t - margin_b

    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_sm = ImageFont.truetype("arial.ttf", 12)
        font_lg = ImageFont.truetype("arialbd.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font
        font_lg = font

    all_values = [comparison[m].values for m in metrics]
    max_val = max(max(v) for v in all_values) * 1.25

    group_width = chart_w / n_models
    bar_width = group_width * 0.28
    gap = bar_width * 0.2

    for i, model in enumerate(models):
        group_x = margin_l + i * group_width
        for j, metric in enumerate(metrics):
            val = comparison.loc[comparison["Model"] == model, metric].values[0]
            bar_h = (val / max_val) * chart_h
            x0 = group_x + (group_width - (bar_width * 2 + gap)) / 2 + j * (bar_width + gap)
            y0 = margin_t + chart_h - bar_h
            x1 = x0 + bar_width
            y1 = margin_t + chart_h
            draw.rectangle([x0, y0, x1, y1], fill=COLORS[metric])
            label = f"{val:.4f}"
            bbox = draw.textbbox((0, 0), label, font=font_sm)
            tw = bbox[2] - bbox[0]
            draw.text((x0 + (bar_width - tw) / 2, y0 - 18), label, fill=(0, 0, 0), font=font_sm)

        label = model
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((group_x + (group_width - tw) / 2, margin_t + chart_h + 10), label, fill=(0, 0, 0), font=font)

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = margin_t + chart_h - tick * chart_h
        draw.line([(margin_l, y), (margin_l + chart_w, y)], fill=(200, 200, 200), width=1)
        val = tick * max_val
        label = f"{val:.4f}"
        bbox = draw.textbbox((0, 0), label, font=font_sm)
        tw = bbox[2] - bbox[0]
        draw.text((margin_l - tw - 8, y - 7), label, fill=(100, 100, 100), font=font_sm)

    draw.line([(margin_l, margin_t), (margin_l, margin_t + chart_h)], fill=(0, 0, 0), width=2)
    draw.line([(margin_l, margin_t + chart_h), (margin_l + chart_w, margin_t + chart_h)], fill=(0, 0, 0), width=2)

    title = "Model Comparison: MAE and RMSE"
    bbox = draw.textbbox((0, 0), title, font=font_lg)
    tw = bbox[2] - bbox[0]
    draw.text(((img_w - tw) / 2, 15), title, fill=(0, 0, 0), font=font_lg)

    legend_x = margin_l + chart_w - 180
    legend_y = margin_t + 10
    for i, metric in enumerate(metrics):
        y = legend_y + i * 24
        draw.rectangle([legend_x, y, legend_x + 16, y + 16], fill=COLORS[metric])
        draw.text((legend_x + 22, y), metric, fill=(0, 0, 0), font=font)

    img.save(str(output_path))


def evaluate_all(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    experiments_path = Path(config.get("experiments", {}).get(
        "path", "experiments"
    ))
    results_path = Path(config.get("results", {}).get(
        "path", "results"
    ))

    tables_path = results_path / "tables"
    figures_path = results_path / "figures"
    tables_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, (folder, filename) in MODEL_FILES.items():
        file_path = Path(folder) / filename
        if not file_path.exists():
            print(f"Prediction file not found for {model_name}: {file_path}")
            continue
        df = pd.read_csv(file_path)
        metrics = compute_model_metrics(df)
        rows.append({
            "Model": model_name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
        })

    if not rows:
        print("No prediction files found to evaluate.")
        return

    comparison = pd.DataFrame(rows)
    tables_file = tables_path / "model_comparison.csv"
    comparison.to_csv(tables_file, index=False)
    print(f"Saved comparison table to {tables_file}")
    print(comparison.to_string(index=False))

    figures_file = figures_path / "model_comparison.png"
    _draw_bar_chart(comparison, figures_file)
    print(f"Saved comparison chart to {figures_file}")


if __name__ == "__main__":
    evaluate_all()
