from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_max_drawdown(cumulative_returns):
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()


def run_backtest(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    experiments_path = Path(config.get("experiments", {}).get("path", "experiments"))
    results_path = Path(config.get("results", {}).get("path", "results"))

    tables_path = results_path / "tables"
    figures_path = results_path / "figures"
    tables_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    weights_file = experiments_path / "portfolio_weights.csv"
    if not weights_file.exists():
        print(f"Weights file not found: {weights_file}")
        return

    weights_df = pd.read_csv(weights_file, index_col=0)

    model_predictions = {
        "RandomForest": experiments_path / "rf_predictions.csv",
        "XGBoost": experiments_path / "xgb_predictions.csv",
        "TabPFN": experiments_path / "tabpfn_predictions.csv",
    }

    risk_free_annual = config.get("portfolio", {}).get("risk_free_rate", 0.02)

    rows = []
    cumulative_data = {}

    for model_name in weights_df.columns:
        pred_file = model_predictions.get(model_name)
        if pred_file is None or not pred_file.exists():
            print(f"Predictions not found for {model_name}: {pred_file}")
            continue

        preds = pd.read_csv(pred_file)
        preds = preds.sort_values("Date").reset_index(drop=True)
        returns_pivot = preds.pivot_table(
            index="Date", columns="Ticker", values="Actual_Return"
        )

        model_weights = weights_df[model_name]
        tickers = model_weights.index.tolist()
        returns_aligned = returns_pivot.reindex(columns=tickers).fillna(0)

        daily_portfolio_returns = returns_aligned.values @ model_weights.values
        cumulative_returns = np.cumprod(1 + daily_portfolio_returns)

        n_days = len(daily_portfolio_returns)
        ann_return = (cumulative_returns[-1] ** (252 / n_days)) - 1
        ann_vol = np.std(daily_portfolio_returns, ddof=1) * np.sqrt(252)
        sharpe = (
            (ann_return - risk_free_annual) / ann_vol if ann_vol > 0 else 0.0
        )
        max_dd = compute_max_drawdown(cumulative_returns)

        rows.append({
            "Model": model_name,
            "Portfolio Return": round(ann_return, 6),
            "Sharpe Ratio": round(sharpe, 6),
            "Max Drawdown": round(max_dd, 6),
            "Annual Volatility": round(ann_vol, 6),
        })
        cumulative_data[model_name] = cumulative_returns

    if not rows:
        print("No models evaluated.")
        return

    metrics_df = pd.DataFrame(rows)
    metrics_file = tables_path / "portfolio_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Saved portfolio metrics to {metrics_file}")
    print(metrics_df.to_string(index=False))

    img_w, img_h = 900, 500
    margin_l, margin_r, margin_t, margin_b = 80, 30, 60, 80
    chart_w = img_w - margin_l - margin_r
    chart_h = img_h - margin_t - margin_b

    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_lg = ImageFont.truetype("arialbd.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        font_lg = font

    colors = [(66, 133, 244), (234, 67, 53), (52, 168, 83)]

    all_cum = np.concatenate(list(cumulative_data.values()))
    y_min = all_cum.min() * 0.95
    y_max = all_cum.max() * 1.05
    x_max = max(len(v) for v in cumulative_data.values())

    for i, (model_name, cum_returns) in enumerate(cumulative_data.items()):
        n = len(cum_returns)
        points = []
        for j in range(n):
            x = (
                margin_l + (j / (x_max - 1)) * chart_w
                if x_max > 1
                else margin_l
            )
            y = margin_t + chart_h - (
                (cum_returns[j] - y_min) / (y_max - y_min)
            ) * chart_h
            points.append((x, y))
        for k in range(len(points) - 1):
            draw.line(
                [points[k], points[k + 1]],
                fill=colors[i % len(colors)],
                width=2,
            )

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = margin_t + chart_h - tick * chart_h
        draw.line(
            [(margin_l, y), (margin_l + chart_w, y)],
            fill=(200, 200, 200),
            width=1,
        )
        val = y_min + tick * (y_max - y_min)
        label = f"{val:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (margin_l - tw - 8, y - 7),
            label,
            fill=(100, 100, 100),
            font=font,
        )

    draw.line(
        [(margin_l, margin_t), (margin_l, margin_t + chart_h)],
        fill=(0, 0, 0),
        width=2,
    )
    draw.line(
        [(margin_l, margin_t + chart_h), (margin_l + chart_w, margin_t + chart_h)],
        fill=(0, 0, 0),
        width=2,
    )

    title = "Cumulative Returns"
    bbox = draw.textbbox((0, 0), title, font=font_lg)
    tw = bbox[2] - bbox[0]
    draw.text(
        ((img_w - tw) / 2, 15), title, fill=(0, 0, 0), font=font_lg
    )

    legend_x = margin_l + chart_w - 180
    legend_y = margin_t + 10
    for i, model_name in enumerate(cumulative_data.keys()):
        y = legend_y + i * 24
        draw.rectangle(
            [legend_x, y, legend_x + 16, y + 16],
            fill=colors[i % len(colors)],
        )
        draw.text(
            (legend_x + 22, y), model_name, fill=(0, 0, 0), font=font
        )

    figures_file = figures_path / "cumulative_returns.png"
    img.save(str(figures_file))
    print(f"Saved cumulative returns chart to {figures_file}")


if __name__ == "__main__":
    run_backtest()
