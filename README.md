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
⚙️ Installation & SetupClone or download this repository to your local machine.Ensure you have your virtual environment activated:Bash# Windows
.venv\Scripts\Activate.ps1
Install the required libraries:Bashpip install -r requirements.txt
🚀 How to Run the ProjectStep 1: Clean and Prepare DataRun the cleaning script to merge data sources and prepare your feature matrices:Bashpython src/clean.py
Step 2: Run Experimental ModelsEvaluate how individual algorithms perform against the baseline:Bashpython models/train_linear.py
Step 3: Execute Production ModelRun the champion ensemble architecture:Bashpython main.py
