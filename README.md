# Explainable AI-Assisted Portfolio Optimization Using Tabular Foundation Models

A framework for portfolio optimization that leverages tabular foundation models with explainable AI techniques to provide transparent and interpretable investment decisions.

## Project Structure

```
equity-ai-portfolio/
├── data/
│   ├── raw/            # Raw market data
│   └── processed/      # Processed/cleaned data
├── notebooks/          # Jupyter notebooks for exploration
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── features/       # Feature engineering
│   ├── models/         # Tabular foundation models
│   ├── portfolio/      # Portfolio optimization logic
│   ├── explainability/ # XAI techniques
│   ├── evaluation/     # Model and portfolio evaluation
│   └── utils/          # Utility functions
├── dashboard/          # Visualization dashboard
├── experiments/        # Experiment configurations
└── results/            # Output results and logs
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure parameters in `config.yaml`

3. Place raw data in `data/raw/`

## License

MIT
