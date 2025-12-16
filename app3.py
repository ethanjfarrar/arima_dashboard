import streamlit as st
import re

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import jarque_bera, probplot
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox, het_arch
from statsmodels.stats.stattools import durbin_watson
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

def run_diagnostics(models_dict):
    rows = []

    for key, model in models_dict.items():
        resid = model.resid
        exog = model.model.exog

        jb_stat, jb_p = jarque_bera(resid)

        lm_stat, lm_p, f_stat, f_p = het_breuschpagan(resid, exog)

        w_lm_stat, w_lm_p, w_f_stat, w_f_p = het_white(resid, exog)

        dw = durbin_watson(resid)

        lb = acorr_ljungbox(resid, lags=[10], return_df=True)
        lb_stat = lb["lb_stat"].iloc[0]
        lb_p = lb["lb_pvalue"].iloc[0]

        rows.append({
            "asset": key.replace("ex_ret_", ""),
            "JB stat": jb_stat,
            "JB p": jb_p,
            "BP LM stat": lm_stat,
            "BP LM p": lm_p,
            "White LM stat": w_lm_stat,
            "White LM p": w_lm_p,
            "DW": dw,
            "LB(10) stat": lb_stat,
            "LB(10) p": lb_p
        })
    return pd.DataFrame(rows).set_index("asset")   

def compute_ex_rets(data, rf="risk_free"):
    df = data.copy()
    price_cols = [c for c in df.columns if c != rf]
    for col in price_cols:
        df[f"log_ret_{col}"] = 100 * np.log(df[col]).diff()

    for col in price_cols:
        df[f"ex_ret_{col}"] = df[f"log_ret_{col}"] - df["risk_free"]

    return df

def run_capm_ols(data, market_ex_ret="ex_ret_market"):
    results = {}
    ex_ret_cols = [
        c for c in data.columns
        if c.startswith("ex_ret_") and c != market_ex_ret
    ]
    for col in ex_ret_cols:
        formula = f"{col} ~ {market_ex_ret}"
        model = smf.ols(formula=formula, data=data).fit()
        results[col] = model

    return results

# Computing returns
def compute_rets(data, rf="risk_free"):
    df = data.copy()
    ret_cols = [c for c in df.columns if c != rf and not c.startswith("ex_ret_")]
    for col in ret_cols:
        df[f"ret_{col}"] = 100 * np.log(df[col]).diff()

    return df

def ARIMA_ols(data_2):
    results = {}
    ret_cols = [c for c in data_2.columns if c.startswith("ret_")]
    for col in ret_cols:
        model = smf.ols(f"{col} ~ 1", data=data_2).fit()
        results[col] = model

    return results
# Fit GARCH 11 model

def GARCH_modelling(data_2, in_sample):
    garch_results = {}
    ret_cols = [c for c in data_2.columns if c.startswith("ret_")]
    for col in ret_cols:
        y = in_sample[col].dropna()
    
        garch_11_model = arch_model(
            y,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="t"
        )
    
        garch_11_fit = garch_11_model.fit(disp="off")
        garch_results[col] = garch_11_fit

    return garch_results

# Checking validity of GARCH
# %matplotlib inline

def plot_garch_results(garch_results):
    for col, fit in garch_results.items():
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        axes[0].plot(fit.conditional_volatility)
        axes[0].set_title("Conditional Volatility")

        std_resid = fit.std_resid.dropna()
        axes[1].plot(std_resid)
        axes[1].axhline(0, linestyle="--")
        axes[1].set_title("Standardised Residuals")


        axes[2].plot(std_resid**2)
        axes[2].set_title("Squared Standardised Residuals")

        fig.suptitle(f"GARCH(1,1) Diagnostics — {col}", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        st.pyplot(fig)
        plt.close(fig)



def plot_garch_variance_forecasts(var_forecasted_df):
    for col in var_forecasted_df.columns:
        fig, ax = plt.subplots(figsize=(16, 6))

        ax.plot(
            var_forecasted_df.index,
            var_forecasted_df[col],
            linestyle='-',
            marker='o',
            linewidth=2
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Variance")
        ax.set_title(f"Forecasted Conditional Variance — {col}")
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ADF Testing

def ADF_testing(data, cols):
    adf_results = []
    for col in cols:
        stat, pval, _, _, crit, _ = adfuller(data[col].dropna())

        adf_results.append({
        "Series": col,
        "Stat": stat,
        "pval": pval,
        "Crit val": crit})

    return pd.DataFrame(adf_results).set_index("Series")

# Differencing the data

def difference_prices(data, stock_cols):
    df = data.copy()

    for col in stock_cols:
        df[f"{col}_diff"] = df[col].diff()

    return df

def kpss_test(df, df_diff, cols, diff_cols, regression="c"):
    kpss_results = []
    for col in cols:
        stat, pval, nlags, critval = kpss(df[col], regression=regression)
        kpss_results.append({
        "Series": col,
        "stat": stat,
        "Pval": pval,
        "nlags": nlags,
        "crit_val": critval
        })

    for col in diff_cols:
        stat_d, pval_d, nlags_d, critval_d = kpss(df_diff[col], regression=regression)
        kpss_results.append({
            "Series_diff": col,
            "stat_diff": stat_d,
            "Pval_diff": pval_d,
            "nlags_diff": nlags_d,
            "crit_val_diff": critval_d
            })

    return kpss_results

# PACF/ACF Plotting



def plotting_acfs(data_diff, diff_cols):
    fig, axes = plt.subplots(len(diff_cols), 2, figsize=(14, 4 * len(diff_cols)))

    for i, col in enumerate(diff_cols):
        plot_acf(data_diff[col].dropna(), ax=axes[i, 0], lags=30)
        axes[i, 0].set_title(f"ACF of {col}")

        plot_pacf(
            data_diff[col].dropna(),
            ax=axes[i, 1],
            lags=30,
            method="ywm"
        )
        axes[i, 1].set_title(f"PACF of {col}")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# Doing ARIMA

def arima_running(data_diff, diff_stock_cols):
    arima_results = []
    p_range = range(0,3)
    q_range= range(0,3)
    for col in diff_stock_cols:
        for p in p_range:
            for q in q_range:
                try:
                    arima_model = ARIMA(data_diff[col], order=(p,0,q))
                    arima_fitted = arima_model.fit()

                    arima_results.append({
                    "Series": col,
                    "AR(p)": p,
                    "MA(q)": q,
                    "I": 1,
                    "AIC": arima_fitted.aic,
                    "BIC": arima_fitted.bic
                    })
                except Exception:
                    arima_results.append({
                        "Series": col,
                        "AR(p)": p,
                        "MA(q)": q,
                        "I": 1,
                        "AIC": None,
                        "BIC": None
                    })

    arima_results_df = pd.DataFrame(arima_results)
    return arima_results_df


# ARIMA GRAPHING

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot
from statsmodels.graphics.tsaplots import plot_acf

def arima_residual_diagnostics_all(arima_fits, lags=30, bins=30):
    residuals_dict = {}

    for ticker, fitted in arima_fits.items():
        resid = fitted.resid

        if hasattr(resid, "index"):
            x = resid.index
            r = resid.dropna().values
            x = x[-len(r):] 
        else:
            r = np.asarray(resid)
            r = r[~np.isnan(r)]
            x = np.arange(len(r))

        residuals_dict[ticker] = r

        # 1) Residuals over time
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(x, r, label="Residuals")
        ax.axhline(0, linestyle="--")
        ax.set_title(f"{ticker} — Residuals")
        ax.set_xlabel("Date" if hasattr(resid, "index") else "Time")
        ax.set_ylabel("Residuals")
        ax.legend()

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)



        # 2) Histogram + Q-Q
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].hist(r, bins=bins, density=True, alpha=0.7, edgecolor="black")
        axes[0].set_title(f"{ticker} — Histogram of Residuals")
        axes[0].set_xlabel("Residual")
        axes[0].set_ylabel("Density")

        probplot(r, dist="norm", plot=axes[1])
        axes[1].set_title(f"{ticker} — Q-Q Plot of Residuals")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # 3) ACF of residuals
        fig, ax = plt.subplots(figsize=(10, 4))
        
        plot_acf(r, lags=lags, ax=ax)
        ax.set_title(f"{ticker} — ACF of Residuals")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


    return residuals_dict


@st.cache_data(show_spinner=False)
def _download_yf_close(tickers, market, risk_free, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    market_data = yf.download(market, start=start_date, end=end_date)["Close"]
    risk_free_data = yf.download(risk_free, start=start_date, end=end_date)["Close"]
    data["market"] = market_data
    data["risk_free"] = risk_free_data
    data.index = pd.to_datetime(data.index)
    return data

def _parse_tickers(raw: str):
    # Accept comma/space separated input; keep order; drop empties.
    if raw is None:
        return []
    parts = re.split(r"[\s,]+", raw.strip().upper())
    tickers = [p for p in parts if p]
    # de-duplicate while preserving order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def main():
    st.set_page_config(page_title="OLS + Diagnostics + (Optional) GARCH/ARIMA", layout="wide")
    st.title("OLS + Diagnostics Dashboard")

    with st.sidebar:
        st.header("Inputs")
        raw_tickers = st.text_input("Stocks (comma or space separated)", value="NVDA, GOOG, MS")
        tickers = _parse_tickers(raw_tickers)

        market_label_to_ticker = {
            "S&P 500 (^GSPC)": "^GSPC",
            "Nasdaq Composite (^IXIC)": "^IXIC",
            "Dow Jones (^DJI)": "^DJI",
            "FTSE 100 (^FTSE)": "^FTSE",
            "Euro Stoxx 50 (^STOXX50E)": "^STOXX50E",
        }
        market_choice = st.selectbox("Market index", options=list(market_label_to_ticker.keys()), index=0)
        market = market_label_to_ticker[market_choice]

        risk_free = "^TNX"  # kept as in the notebook
        st.caption(f"Risk-free proxy: {risk_free} (yfinance)")

        start_date = pd.Timestamp("2020-01-01").tz_localize("UTC")
        end_date = pd.Timestamp("2025-01-01").tz_localize("UTC")
        st.caption(f"Date range fixed to notebook defaults: {start_date.date()} to {end_date.date()}")


        st.markdown("---")
        model_choice = st.radio(
            "Select model", 
            ["OLS", "ARIMA", "GARCH"],
            index=0
        )
        run_btn = st.button("Run model", type="primary")

        if not run_btn:
            st.info("Select a model and click Run")
            st.stop()

    if len(tickers) == 0:
        st.error("Please enter at least one stock ticker.")
        return

    with st.spinner("Downloading data from yfinance..."):
        data = _download_yf_close(tickers, market, risk_free, start_date, end_date)

    if data is None or data.empty:
        st.error("No data returned from yfinance. Check tickers and try again.")
        return

    st.subheader("Downloaded Close Prices")
    st.dataframe(data.tail(10), use_container_width=True)

    # --- CAPM / OLS section (notebook logic) ---
    if model_choice == "OLS":
        st.header("CAPM OLS")
        # Compute excess returns
        data_1 = compute_ex_rets(data)
        data_1.dropna(inplace=True)

        reg_data = run_capm_ols(data_1)
        diogs = run_diagnostics(reg_data)
        
        # Display diagnostics and model outputs
        colA, colB = st.columns([1, 1])
        with colA:
            st.subheader("Diagnostics")
            
            diogs_fmt = diogs.copy()
            
            for col in diogs_fmt.columns:
                if col.lower().endswith("p"):
                    diogs_fmt[col] = diogs_fmt[col].apply(
                        lambda x: f"{x:.2e}" if isinstance(x, float) else x
                    )
                    
            st.dataframe(diogs_fmt, use_container_width=True)

        with colB:
            st.subheader("Model outputs")
            for k, model in reg_data.items():
                st.markdown(f"**{k.replace('ex_ret_', '')}**")
        st.subheader("Summary statistics")
        ex_rets_cols = [c for c in data_1.columns if c.startswith("ex_ret_")]
        summary_stats = data_1[ex_rets_cols].describe().T
        st.dataframe(summary_stats, use_container_width=True)
        
        st.subheader("Plots")

        fig, ax = plt.subplots(figsize=(10, 5))
        for col in ex_rets_cols:
            ax.plot(
                data_1.index,
                data_1[col],
                label=col,
                linewidth=0.7
        )
        
        ax.set_title("Daily Excess Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Excess return (%)")
        ax.legend()
        fig.tight_layout()
        
        st.pyplot(fig)
        plt.close(fig)
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        axes = axes.flatten()
        for ax, col in zip(axes, ex_rets_cols):
            ax.hist(data_1[col], bins=40)
            ax.set_title(col)
            ax.set_xlabel("Excess return (%)")
            ax.set_ylabel("Frequency")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Scatter plots vs market
        for col in [c for c in ex_rets_cols if c != "ex_ret_market"]:
            fig, ax = plt.subplots(figsize=(5, 4))

            ax.scatter(
                data_1["ex_ret_market"],
                data_1[col],
                s=5
            )
        ax.set_title(f"{col} vs Market Excess Return")
        ax.set_xlabel("Market excess return")
        ax.set_ylabel(col)
        
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        

# --- Optional GARCH ---
    elif model_choice == "GARCH":
        st.header("GARCH")
        data_2 = compute_rets(data)
        data_2.dropna(inplace=True)

        # Split into in-sample / out-of-sample exactly as in the notebook
        data_2 = data_2.copy()
        data_2.index = pd.to_datetime(data_2.index).tz_localize("UTC")
        split_date = start_date + (end_date - start_date) / 2
        in_sample = data_2.loc[:split_date]
        out_of_sample = data_2.loc[split_date:]
        st.write("In Sample", in_sample.shape)
        st.write("Out of sample", out_of_sample.shape)

        ARIMA_ols_res = ARIMA_ols(data_2)
        arch_name = ["Lagrange Multiplier Test Statistic", "lm p-value", "f-statistic", "f p-value"]
        arch_tests = {}
        for col, res in ARIMA_ols_res.items():
            lm_stat, lm_p, f_stat, f_p = het_arch(res.resid)
            arch_tests[col] = dict(zip(arch_name, [lm_stat, lm_p, f_stat, f_p]))
        st.subheader("ARCH tests (on ARIMA_ols residuals)")
        arch_df = pd.DataFrame(arch_tests).T
        for col in arch_df.columns:
            arch_df[col] = arch_df[col].apply(
                lambda x: f"{x:.2e}" if isinstance(x, float) else x
            )
        st.dataframe(arch_df, use_container_width=True)

        garch_results = GARCH_modelling(data_2, in_sample)
        st.subheader("GARCH(1,1) fit summaries")
        for col, fit in garch_results.items():
            st.markdown(f"**{col}**")
            st.code(str(fit.summary()))

        st.subheader("GARCH diagnostics plots")
        plot_garch_results(garch_results)

        H = len(out_of_sample)
        var_forecasted = {}
        for col, res in garch_results.items():
            fc = res.forecast(horizon=H, reindex=False)
            var_path = fc.variance.iloc[-1].to_numpy()
            var_forecasted[col] = pd.Series(var_path, index=out_of_sample.index)
        var_forecasted_df = pd.DataFrame(var_forecasted)
        st.subheader("Variance forecasts")
        st.dataframe(var_forecasted_df.head(10), use_container_width=True)
        plot_garch_variance_forecasts(var_forecasted_df)
 # --- Optional ARIMA ---
    elif model_choice == "ARIMA":
        st.header("ARIMA")
        stock_cols = [c for c in data.columns if c not in ("risk_free", "market")]
        st.subheader("ADF tests (prices)")
        adf_results = ADF_testing(data, stock_cols)
        st.dataframe(adf_results, use_container_width=True)

        data_diff = difference_prices(data, stock_cols)
        data_diff.dropna(inplace=True)
        diff_stock_cols = [c for c in data_diff.columns if c.endswith("_diff")]
        st.subheader("ADF tests (differenced prices)")
        adf_results_diff = ADF_testing(data_diff, diff_stock_cols)
        st.dataframe(adf_results_diff, use_container_width=True)

        st.subheader("KPSS tests")
        kpss_out = kpss_test(data, data_diff, stock_cols, diff_stock_cols)
        st.dataframe(pd.DataFrame(kpss_out), use_container_width=True)

        st.subheader("ACF / PACF")
        plotting_acfs(data_diff, diff_stock_cols)

        st.subheader("ARIMA grid search (AIC/BIC)")
        arima_grid = arima_running(data_diff, diff_stock_cols)
        st.dataframe(arima_grid, use_container_width=True)

        # Pick best (p,q) by minimum AIC per series (same model family as notebook grid)
        best = (
            arima_grid.dropna(subset=["AIC"]).sort_values("AIC").groupby("Series", as_index=False).first()
        )
        st.subheader("Selected orders (min AIC)")
        st.dataframe(best, use_container_width=True)

        # Build ARIMA specs for original price series
        arima_specs = {}
        for _, row in best.iterrows():
            series = row["Series"]
            base = series.replace("_diff", "")
            arima_specs[base] = (int(row["AR(p)"]), 1, int(row["MA(q)"]))
  # Ensure we have an out_of_sample horizon consistent with the notebook split
        data_2_for_h = compute_rets(data)
        data_2_for_h.dropna(inplace=True)
        data_2_for_h = data_2_for_h.copy()
        data_2_for_h.index = pd.to_datetime(data_2_for_h.index).tz_localize("UTC")
        split_date = start_date + (end_date - start_date) / 2
        out_of_sample = data_2_for_h.loc[split_date:]
        H = len(out_of_sample)

        st.subheader("ARIMA fits")
        arima_fits = {}
        for stock, order in arima_specs.items():
            try:
                model = ARIMA(data[stock], order=order)
                arima_fits[stock] = model.fit()
                st.markdown(f"**{stock} — order {order}**")
                st.code(arima_fits[stock].summary().as_text())
            except Exception as e:
                st.warning(f"ARIMA failed for {stock} with order {order}: {e}")

        if len(arima_fits) > 0:
            st.subheader("ARIMA residual diagnostics")
            arima_residual_diagnostics_all(arima_fits, lags=30, bins=30)

            st.subheader("ARIMA price forecasts (out-of-sample horizon)")
            price_forecasts = {}
            for stock, res in arima_fits.items():
                fc = res.get_forecast(steps=H)
                price_forecasts[stock] = fc.predicted_mean.to_numpy()
            price_forecasts_df = pd.DataFrame(price_forecasts, index=out_of_sample.index)
            st.dataframe(price_forecasts_df.head(10), use_container_width=True)



if __name__ == "__main__":
    main()

