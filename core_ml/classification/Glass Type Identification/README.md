# Glass Type Identification

A machine learning classification project to identify the type of glass based on its physical and chemical properties.

> **Note:** This project is a work in progress. The structure and content below are subject to change as development continues.

## Project Overview

The goal of this project is to build a classification model that predicts glass type from features such as the concentration of various oxides (RI, Na, Mg, Al, Si, K, Ca, Ba, Fe, etc.).

## Project Structure

```
Glass Type Identification/
├── .vscode/
│   └── settings.json              # Editor settings
├── data/
│   ├── glass.csv                  # Raw dataset
│   └── glass_clean.csv            # Cleaned dataset
├── notebooks/
│   ├── data_understanding.ipynb   # Initial data exploration
│   └── eda.ipynb                  # Exploratory data analysis
├── src/
│   └── visualization.py           # Visualization utilities
├── utils/
│   └── eda.py                     # EDA helper functions
├── README.md                      # This file
├── requirements.txt               # Project dependencies
└── test.ipynb                     # Test notebook
```

### Directory Descriptions

| Directory / File | Description |
| --- | --- |
| `data/` | Contains the raw and cleaned datasets used for the project |
| `notebooks/` | Jupyter notebooks for data understanding and exploratory analysis |
| `src/` | Core source code, including visualization helpers |
| `utils/` | Reusable utility functions supporting the analysis |
| `test.ipynb` | Notebook used for testing/experimentation |

## Status

🚧 In progress — project structure is set up, with data exploration and EDA underway.
