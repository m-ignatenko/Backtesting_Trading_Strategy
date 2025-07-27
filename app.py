from strategy import Strategy
from Technical import *
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
general = st.sidebar.selectbox('Model selection',("ML based", "Technical Indicators"))
if general == 'ML based':
    type = st.sidebar.selectbox('Model selection',("Logistic Regression","Gradient Boosting"))
    n_est = st.sidebar.number_input("Enter number of estimators(for boosting)", min_value=1, step=1, value=10, format="%d")
    b_type = st.sidebar.selectbox('Base model (for boosting)',("Decision Tree","Random Forest"))

    if (type == 'Gradient Boosting'):
        st.sidebar.text('Might take longer')
elif general =='Technical Indicators':
    type = st.sidebar.selectbox('Model selection',("Moving Averages","Momentum","Mean Reversed"))
    if type =='Moving Averages':
        ma1 = st.sidebar.number_input("Length of 1st Moving average", value=42)
        ma2 = st.sidebar.number_input("Length of 2nd Moving average", value=252)
    elif type == 'Momentum':
        ma = st.sidebar.number_input("Length of Moving average", value=50)
    elif type == "Mean Reversed":
        ma = st.sidebar.number_input("Length of Moving average", value=50)
        th = st.sidebar.number_input("Value of threshold", value=1.0,step=0.1)

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
    capital = st.number_input("Capital", value=10_000.0)
    tc = st.number_input("Transaction cost", value=0.0)
    if general == 'ML based':
        t1 = st.text_input("Training Start Date", value='2020-01-01')
        t2 = st.text_input("Training End Date", value='2024-01-01')
        t3 = st.text_input("Test Start Date", value='2024-01-01')
        t4 = st.text_input("Test End Date", value='2025-06-01')
    elif general =='Technical Indicators':
        t1 = st.text_input("Start Date", value='2020-01-01')
        t2 = st.text_input("End Date", value='2024-01-01')
with main_col:
    if general == "ML based":
        strat = Strategy(ticker,t1,t2,t3,t4,tc, capital)
        strat.run(type, n_est, b_type)
        st.pyplot(plt)
        st.info(f" \n\n Profit by using strategy \
                : {strat.capital*strat.df_test['strategy'].cumsum().apply(np.exp)[-1] - strat.transaction_cost* sum(strat.df_test['prediction'].diff().fillna(0) !=0):.4f} \
                      by treating each transaction cost as {strat.transaction_cost} (total number of trades: {sum(strat.df_test['prediction'].diff().fillna(0) !=0)})\n\nStrategy total return [%]: {strat.df_test['strategy'].cumsum().apply(np.exp)[-1]:.4f}")
        st.info(f"Profit by holding: \
                      {capital*(1+(
                          (strat.df_test['price'].iloc[-1] - strat.df_test['price'].iloc[0])/strat.df_test['price'][0])):.4f} \n\n Buy & Hold return [%]: {(strat.df_test['price'][-1] - strat.df_test['price'][0])/strat.df_test['price'][0]:.4f}")
    elif general == 'Technical Indicators':
        bt = Backtest(ticker,t1,t2,capital,tc,0,True)
        bt.get_data()
        n = bt.data.shape[0]
        if type == 'Moving Averages':
            bt.sma_crossover(ma1, ma2)
        elif type == 'Momentum':
            bt.momentum(ma)
        elif type =='Mean Reversed':
            bt.mean_reversed(ma,th)
        bt.plot()
        st.pyplot(plt)
        st.info(f"Profit by using strategy : {bt.get_wealth(n-1):.4f} by treating each transaction cost as {tc} (total number of trades: {bt.trades})\
                 \n\n Strategy total return [%]: { 100*(bt.get_wealth(n-1) - capital) /capital:.4f}")
        hold_profit = capital * (1 + (bt.data['price'].iloc[-1]-bt.data['price'].iloc[0])/bt.data['price'].iloc[0])
        st.info(f"Profit by holding : {hold_profit:.4f} \n \n Buy & Hold return [%]: {(bt.data['price'].iloc[-1]-bt.data['price'].iloc[0])/bt.data['price'].iloc[0]*100:.4f}")
        st.header("Trade journal")
        st.text(bt.msg)