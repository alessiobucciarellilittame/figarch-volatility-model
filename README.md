# FIGARCH Volatility Model

This project studies the presence of long-memory effects in hedge fund return volatility using FIGARCH models.

The analysis is based on a sample of HFRX hedge fund indices representing different investment strategies. The objective is to evaluate how fractional volatility dynamics affect risk measurement and portfolio management.

## Project Report

The full academic paper describing the theoretical framework, econometric methodology and empirical results can be found here:

[Project Report](PW_Econometric_Theory.pdf)

## Methodology

The project implements a FIGARCH(1,d,1) model estimated through Quasi-Maximum Likelihood Estimation (QMLE) in MATLAB.

The analysis includes:

- Estimation of FIGARCH volatility models for hedge fund returns
- Comparison with standard GARCH models
- Value-at-Risk (VaR) estimation under different distributions
- VaR backtesting using Christoffersen tests
- Portfolio optimization based on FIGARCH-based risk measures

## Data

The dataset consists of HFRX hedge fund indices covering several strategies such as:

- Equity Hedge
- Event Driven
- Relative Value
- Macro / CTA

## Output

The project produces:

- estimated FIGARCH parameters
- volatility forecasts
- VaR backtesting results
- portfolio allocation and performance metrics

## Files

Main scripts:

- Main_Part_1.m – econometric analysis and model estimation
- Main_Part_2.m – VaR backtesting and portfolio optimization
- stima_figarch_qmle.m – FIGARCH estimation procedure
- compute_portfolio_metrics.m – portfolio risk metrics
- run_multi_risk_figarch_portfolio.m – portfolio optimization under different VaR models

Data:

- HFRX2.xlsx

Results:

- Results_Portfolio_FIGARCH.xlsx

Full report:

- PW_Econometric_Theory.pdf

## Authors

Alessio Bucciarelli Littamè  
Alessandro Pedrini  
Francesco Pocci Sanguigni  
Luca Trainito  

MSc in Finance – Luiss Guido Carli University
