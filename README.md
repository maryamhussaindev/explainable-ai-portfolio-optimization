### Install dependencies

```bash
pip install -r requirements.txt
```

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

### 10. Generate SHAP explanations *(current work in progress)*

```bash
python -m src.explainability.shap_analysis
```

> **Note:** SHAP generation for TabPFN is computationally expensive on CPU and may take considerable time.

---

## Output Files

After running the completed steps, the project generates:

```text
data/processed/
experiments/
results/figures/
results/tables/
```

These folders contain processed datasets, model predictions, evaluation metrics, portfolio results, and SHAP visualizations.

---
