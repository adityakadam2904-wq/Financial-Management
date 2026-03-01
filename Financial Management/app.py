import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="NSE Portfolio Optimizer", layout="wide")

st.title("NSE Portfolio Optimizer (MPT)")

# --- Sidebar inputs ---
st.sidebar.header("Settings")

default_tickers = "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, HINDUNILVR.NS"
tickers_str = st.sidebar.text_input("Tickers (NSE, comma-separated)", default_tickers)
tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start date", pd.to_datetime("2021-03-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("2026-03-01"))

risk_free = st.sidebar.number_input("Risk-free rate (annual, India)", value=0.07, step=0.01)
max_weight = st.sidebar.slider("Max weight per stock", 0.1, 1.0, 0.4, 0.05)
num_ports = st.sidebar.number_input("Number of random portfolios", 1000, 10000, 3000, 500)

st.sidebar.write("Click button to run:")
run = st.sidebar.button("Optimize")

if run:
    # --- Data download ---
    st.subheader("1. Price Data")

    raw_data = yf.download(tickers, start=start_date, end=end_date)
    if raw_data.empty:
        st.error("No data returned. Check tickers or dates.")
        st.stop()

    data = raw_data.xs(key='Close', axis=1, level='Price')
    data = data[tickers]

    st.write("Price data (head):")
    st.dataframe(data.head())

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    data.plot(ax=ax1)
    ax1.set_title("Close Prices")
    ax1.set_ylabel("Price")
    st.pyplot(fig1)

    # --- Returns & stats ---
    st.subheader("2. Returns & Statistics")

    log_returns = np.log(data / data.shift(1)).dropna()
    trading_days = 252

    mean_daily = log_returns.mean()
    cov_daily = log_returns.cov()

    annual_returns = mean_daily * trading_days
    annual_vol = log_returns.std() * np.sqrt(trading_days)

    stats_df = pd.DataFrame({
        "Annual Return": annual_returns,
        "Annual Volatility": annual_vol
    })
    st.write("Per-stock stats:")
    st.dataframe(stats_df.style.format("{:.2%}"))

    # --- Portfolio functions ---
    def portfolio_return(w):
        return np.dot(w, mean_daily) * trading_days

    def portfolio_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_daily * trading_days, w)))

    def portfolio_sharpe(w, rf=risk_free):
        ret = portfolio_return(w)
        vol = portfolio_vol(w)
        return (ret - rf) / vol

    # --- Monte Carlo simulation ---
    st.subheader("3. Random Portfolios (Monte Carlo)")

    results = np.zeros((3, num_ports))
    weights_record = []

    for i in range(num_ports):
        w = np.random.random(len(tickers))
        w = w / w.sum()
        weights_record.append(w)

        ret = portfolio_return(w)
        vol = portfolio_vol(w)
        sharpe = (ret - risk_free) / vol

        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = sharpe

    best_idx = np.argmax(results[2])
    best_w_random = weights_record[best_idx]

    st.write("Best random Sharpe:", float(results[2, best_idx]))

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sc = ax2.scatter(results[1], results[0], c=results[2], cmap='viridis', s=8)
    plt.colorbar(sc, ax=ax2, label="Sharpe")
    ax2.set_xlabel("Volatility")
    ax2.set_ylabel("Return")
    ax2.set_title("Random Portfolios")
    st.pyplot(fig2)

    # --- Optimization with constraints ---
    st.subheader("4. Optimized Portfolio (Max Sharpe)")

    bounds = tuple((0, max_weight) for _ in range(len(tickers)))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    init_guess = np.array([1/len(tickers)] * len(tickers))

    def neg_sharpe(w):
        return -portfolio_sharpe(w)

    opt_result = minimize(neg_sharpe,
                          init_guess,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)

    if not opt_result.success:
        st.error("Optimization failed: " + opt_result.message)
        st.stop()

    opt_w = opt_result.x
    opt_ret = portfolio_return(opt_w)
    opt_vol = portfolio_vol(opt_w)
    opt_sharpe = portfolio_sharpe(opt_w)

    opt_df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": opt_w
    })
    st.write("Optimal weights (max Sharpe with constraint):")
    st.dataframe(opt_df.style.format({"Weight": "{:.2%}"}))

    st.write(f"Optimized annual return: {opt_ret:.2%}")
    st.write(f"Optimized volatility: {opt_vol:.2%}")
    st.write(f"Optimized Sharpe: {opt_sharpe:.2f}")

    # Plot optimal vs random cloud
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sc2 = ax3.scatter(results[1], results[0], c=results[2], cmap='viridis', s=8)
    ax3.scatter(opt_vol, opt_ret, color='red', marker='*', s=200, label='Optimized')
    plt.colorbar(sc2, ax=ax3, label="Sharpe")
    ax3.set_xlabel("Volatility")
    ax3.set_ylabel("Return")
    ax3.set_title("Efficient Frontier Approximation")
    ax3.legend()
    st.pyplot(fig3)

    # --- Backtest ---
    st.subheader("5. Backtest Optimized Portfolio")

    initial_capital = 10000
    normed = data / data.iloc[0]
    alloc = normed * opt_w
    port_val = alloc.sum(axis=1) * initial_capital

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    port_val.plot(ax=ax4)
    ax4.set_title("Optimized Portfolio Value (₹10,000 start)")
    ax4.set_ylabel("Portfolio Value (₹)")
    st.pyplot(fig4)

    total_return = port_val.iloc[-1] / initial_capital - 1
    daily_port_ret = port_val.pct_change().dropna()
    ann_sharpe_bt = (daily_port_ret.mean() * 252) / (daily_port_ret.std() * np.sqrt(252))

    st.write(f"Total return over period: {total_return:.2%}")
    st.write(f"Backtested annual Sharpe: {ann_sharpe_bt:.2f}")
