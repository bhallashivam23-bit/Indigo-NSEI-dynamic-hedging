# Indigo-NSEI Dynamic Hedging

## Overview
This project analyzes hedging effectiveness between INDIGO.NS and NSEI using:

- Static OLS hedge ratio
- 60-day rolling beta
- Machine Learning–based beta prediction using Random Forest

The project compares hedge effectiveness under classical finance models and ML-based approaches while maintaining time-series integrity and avoiding future leakage.

---

## Objective
The main objective of this project is to evaluate whether machine learning can improve hedge ratio estimation compared to traditional OLS-based methods.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- yFinance

---

## Methodology

### 1. Static Hedge Ratio
- Calculated using covariance and variance:
  
\[
\beta = \frac{Cov(R_s, R_h)}{Var(R_h)}
\]

- Measured hedge effectiveness using variance reduction.

---

### 2. Dynamic Rolling Beta
- Used 60-day rolling windows
- Calculated time-varying hedge ratios
- Compared rolling hedge effectiveness against static hedge

---

### 3. Machine Learning Hedge Model
Features used:
- Rolling volatility
- Rolling correlation
- Lagged returns
- Volatility ratio
- Basis risk

Model used:
- Random Forest Regressor

Target:
- Future rolling beta

---

## Key Findings
- Static hedge effectiveness ≈ 15%
- Rolling hedge effectiveness ≈ 19%
- ML model did not outperform classical rolling hedge significantly
- Results suggest the hedge relationship is largely linear and already well captured by traditional finance models

---

## Project Structure

## Project Structure

```bash
Indigo-NSEI-dynamic-hedging/
│
├── README.md
├── requirements.txt
│
├── src/
│   └── FDMLPROJECT.py
│
├── reports/
│   └── Indigo NSEI Hedge Journal.docx
```

## Author
Shivam Bhalla
