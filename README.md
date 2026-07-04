# Corporación Favorita Sales Forecasting Pipeline

An end-to-end machine learning project designed to clean historical retail data, analyze store metrics, and evaluate predictive regression models to forecast sales volumes.

## 📁 Project Structure

```text
Database/
│
├── data/                      # Raw and processed datasets
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   ├── holidays_events.csv
│   ├── X_data.csv        # Preprocessed features
│   └── y_data.csv        # Target variable
│
├── src/                       # ETL & Data Preparation
│   └── clean.py               # Merges, encodes, and exports data
│
├── models/                    # Model evaluation scripts
│   ├── train_linear.py        # Baseline Linear Regression
│   ├── train_tree.py          # Decision Tree Regressor
│   └── train_knn.py           # K-Nearest Neighbors with tuning
│
├── main.py                    # Production-ready ensemble model execution
├── requirements.txt           # Python application dependencies
└── README.md                  # Project documentation
