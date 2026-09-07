### Setup (recommended)

Create and activate a virtual environment, then install the dependencies:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

From here on, use `python -m ...` while the virtual environment is active.

---

## Configure the Project

Before running the project, update `config.yaml` with the stock tickers.

```yaml
data:
  tickers:
    - AAPL
    - MSFT
    - GOOGL
    - AMZN
    - NVDA

  start_date: "2020-01-01"
  end_date: "2025-01-01"
```

---

## Run the Project (in order)

### 1. Download raw stock data

```bash
python -m src.data.download
```

### 2. Clean and preprocess the data

```bash
python -m src.data.cleaning
```

### 3. Generate technical features

```bash
python -m src.features.builder
```

### 4. Explore the dataset (optional)

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 5. Create rolling train/validation/test splits

```bash
python -m src.utils.split
```

### 6. Train Random Forest

```bash
python -m src.models.random_forest
```

### 7. Train XGBoost

```bash
python -m src.models.xgboost_model
```

### 8. Train TabPFN

```bash
python -m src.models.tabpfm_model
```

### 9. Compare prediction models

```bash
python -m src.evaluation.prediction_metrics
```

### 10. Explain the models

Random Forest and XGBoost use SHAP (summary plot, feature importance, waterfall plot, and SHAP values CSV):

```bash
python -m src.explainability.shap_analysis
```

TabPFN uses permutation importance instead of SHAP (run on the complete rolling test dataset):

```bash
python -m src.explainability.tabpfn_importance
```

Generate human-readable prediction explanations for each stock/date:

```bash
python -m src.explainability.explanation_report
```

> **Note:** TabPFN models are trained per rolling window. The permutation importance step fits 5 TabPFN models (one per ticker) and evaluates on the full rolling test set — allow enough time on a CPU-only machine.

### 11. Run the portfolio backtest

Rebalances every 5 trading days and compares Equal Weight, Markowitz, Random Forest, XGBoost, and TabPFN portfolios:

```bash
python -m src.portfolio.backtest
```

### 12. Launch the dashboard

```bash
cd dashboard
python app.py
```

Then open the dashboard in your browser:

```
http://127.0.0.1:5000
```

The dashboard has four pages:

- **Overview** - Model comparison table, best model by RMSE, window performance chart.
- **Portfolio** - Portfolio metrics, allocation table, interactive allocation pie chart, growth charts.
- **Explainability** - SHAP summaries for Random Forest / XGBoost, TabPFN permutation importance, and per-stock prediction explanations.
- **5-Day Predictions** - Latest rolling-window predictions with BUY / HOLD / SELL signals, an interactive chart, and today's investment recommendation.

---

## Output Files

After running the completed steps, the project generates:

```text
data/processed/        cleaned price data, engineered technical features, train/test splits
data/splits/           rolling train/validation/test splits for the 5-day evaluation
experiments/predictions/  rolling-window predictions from Random Forest, XGBoost, and TabPFN
experiments/shap/      SHAP values for Random Forest and XGBoost
experiments/portfolio_weights.csv   portfolio weights per rebalance period and strategy
results/tables/        model comparison, portfolio metrics, permutation importance, explanations
results/figures/       SHAP plots, feature importance, portfolio growth charts
dashboard/             Flask dashboard (app.py + templates + static assets)
```

These folders contain processed datasets, model predictions, evaluation metrics, portfolio results, and model explanations.

---
