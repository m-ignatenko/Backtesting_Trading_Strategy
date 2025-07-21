from strategy import Strategy
import warnings
import numpy as np
import streamlit as st
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
st.set_page_config(
        page_title="Markowitz Portfolio Model",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )
st.sidebar.header("Model parameters")
type = st.sidebar.selectbox('Model selection',("Logistic Regression","Gradient Boosting"))
n_est = st.sidebar.number_input("Enter number of estimators(for boosting)", min_value=1, step=10, value=1000, format="%d")

b_type = st.sidebar.selectbox('Base model (for boosting)',("Decision Tree","Random Forest"))

main_col, param_col = st.columns([3, 2])
with param_col:
    linkedin_url = "https://www.linkedin.com/in/mikhail-ignatenko-b79876243/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Mikhail Ignatenko`</a>', unsafe_allow_html=True)
    tg_link = "https://t.me/mikhail_lc"
    st.markdown(f'<a href="{tg_link}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/128/2111/2111646.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Mikhail I`</a>', unsafe_allow_html=True)
    github_link = "https://github.com/m-ignatenko"
    st.markdown(f'<a href="{github_link}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/128/14063/14063266.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Mikhail I`</a>', unsafe_allow_html=True)

    st.header("Asset Parameters")
    ticker = st.text_input("Ticker", value="AAPL")
    tc = st.number_input("Transaction cost", value=0.0)
    capital = st.number_input("Capital", value=10_000.0)
    t1 = st.text_input("Training Start Date", value='2020-01-01')
    t2 = st.text_input("Training End Date", value='2024-01-01')
    t3 = st.text_input("Test Start Date", value='2024-01-01')
    t4 = st.text_input("Test End Date", value='2025-06-01')
with main_col:
    strat = Strategy(ticker,t1,t2,t3,t4,tc, capital)
    strat.run(type, n_est, b_type)
    st.pyplot(plt)
    st.info(f"Strategy total return: {strat.df_test['strategy'].cumsum().apply(np.exp)[-1]:.4f} \n\n Profit by using strategy \
            : {strat.capital*strat.df_test['strategy'].cumsum().apply(np.exp)[-1] - strat.transaction_cost* sum(strat.df_test['prediction'].diff().fillna(0) !=0):.4f} by treating each transaction cost as {strat.transaction_cost} (total number of trades: {sum(strat.df_test['prediction'].diff().fillna(0) !=0)}) \n\n Profit by holding: {strat.capital*strat.df_test['return'].cumsum().apply(np.exp)[-1]:.4f}")