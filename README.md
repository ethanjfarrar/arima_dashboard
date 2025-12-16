# arima_dashboard

**OLS, ARIMA & GARCH Financial Modelling Dashboard**
Overview

This project is a Streamlit-based financial modelling dashboard that allows users to:
- Run CAPM OLS regressions with full residual diagnostics
- Estimate and diagnose ARIMA time-series models
-Fit and forecast volatility using GARCH(1,1) models
- Download market data directly from Yahoo Finance
- Visualise results interactively in a single application
The dashboard is designed to replicate a notebook-style econometric workflow while providing a clean, user-facing interface.

**Key Features**
1. Data Handling
Pulls daily close prices via yfinance
Supports multiple stocks simultaneously
Market index selectable from major benchmarks (S&P 500, Nasdaq, FTSE, etc.)
Uses ^TNX as a risk-free proxy
Fixed sample period (2020–2025) for reproducibility

2. OLS / CAPM Module
Computes log returns and excess returns
Runs CAPM regressions on excess returns

Full diagnostic suite:
Jarque–Bera (normality)
Breusch–Pagan & White (heteroskedasticity)
Durbin–Watson (autocorrelation)
Ljung–Box (serial correlation)
Summary statistics and distribution plots
Excess return time-series and market scatter plots

3. ARIMA Module
Stationarity testing:
ADF (levels and differences)
KPSS (levels and differences)
Automated ACF / PACF plotting
Grid search over ARMA(p,q) specifications (p,q ∈ {0,1,2})
Model selection via AIC

Residual diagnostics:
Time series of residuals
Histogram & Q–Q plots
Residual ACF
Out-of-sample price forecasts

4. GARCH Module
Mean equation estimated via OLS
ARCH effects tested using Engle LM tests
GARCH(1,1) with:
Constant mean
Student-t innovations
In-sample diagnostics
Out-of-sample conditional variance forecasts
Forecast paths visualised per asset

**Project Structure**
app3.py


All logic is contained in a single Streamlit application file for simplicity and portability.

**Requirements**
Install dependencies before running:
pip install streamlit yfinance pandas numpy statsmodels arch scipy matplotlib
Python 3.9+ recommended.

**Running the App**
From the project directory:
streamlit run app3.py
Then open the local URL provided by Streamlit in your browser.

**Usage Flow**
Enter stock tickers (comma or space separated)
Select a market index
Choose a model:
- OLS
- ARIMA
- GARCH
Click Run model
Review tables, diagnostics, and plots interactively

**Design Philosophy**
Econometrically transparent: mirrors academic workflows
Notebook-faithful: no hidden preprocessing
Modular: OLS, ARIMA, and GARCH can be run independently
Pedagogical: diagnostics shown explicitly, not abstracted away

**Notes & Assumptions**
Risk-free rate uses ^TNX (10-year Treasury yield, not converted to daily continuously-compounded form)
ARIMA models are selected via in-sample AIC minimisation
GARCH forecasts use a fixed in-sample / out-of-sample split

No transaction costs or dividends are included
