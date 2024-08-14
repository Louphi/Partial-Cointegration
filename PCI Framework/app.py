import vectorbt as vbt
import streamlit as st
from PCIData import *
from PCIPipeline import *
from PCIStrat import *
import pickle
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def run_streamlit_app():
    st.title('Backtesting Interface : Single Pair')
    #def main(path: str, year: int) -> None:
    # Load adjPrice for all tickers
    year  = 2008
    path = "/Users/sebastiencaron/Desktop/PCI-Project/PCI Framework/data/"
    filepath1 = f"{path}adjPrice_nasdaq100.pkl"
    adjPrice = Definition.f_load_adjPrice(filepath1)


    # Load the CSV file : Compustat daily consituent
    filepath2 = f"{path}NasdaqConstituent.csv"
    object_nasdaq = PCIData.DataCollection(filepath2)
    # Liste des consituents pour une certaine année

    consituent2008 = object_nasdaq.list_nasdaq_constituents_by_year(year)

    # Get price series for list of tickers
    df_2008 = Definition.f_selectTicker(consituent2008, adjPrice)

    # Split In and Out sample
    df_2008_in_sample, df_2008_out_sample = Definition.f_insample_outsample(df_2008, year)

    # Get the In and Out sample 
    df_2008_inout = pd.concat([df_2008_in_sample, df_2008_out_sample])

    # Save in stock prices dictionnary
    stockPrices = {"inSample": df_2008_in_sample,
                "outSample": df_2008_out_sample}

    # Get the list of consituent of price that we have
    consituentList = df_2008_in_sample.columns.tolist()

    # Generate all possible permutations of pairs (including reversed pairs)
    pairData = Definition.pairDatabase(consituentList=consituentList)

    pairName = pairData["pair"]
    inSample = stockPrices["inSample"]
    pairTest  = pairName[0:1000]

    # Takes a lot of time to run
    #elligibilityData, estimationData = Computation.f_compute_estimate(pairTest, inSample, pairData)
    with open(f'{path}data.pickle', 'rb') as f:
        elligibilityData, estimationData = pickle.load(f)

    ### To add in an other category
    elligibility_sorted_data, estimatorData =  Computation.f_create_dataframe(elligibilityData, estimationData, pairTest)

    # Number of pair we keep
    n_keep = 10

    pairListSelected = Computation.f_pairSelected(elligibility_sorted_data, n_keep=n_keep)

    stockList = Computation.get_stock_list(pairData, pairListSelected)


    ### VECTOR BT PART

    window = st.slider('Window Size', min_value=1, max_value=100, value=30)
    entry_threshold = st.slider('Entry Threshold', min_value=0.0, max_value=1.0, value=0.1)
    exit_threshold = st.slider('Exit Threshold', min_value=-1.0, max_value=0.0, value=-0.1)
    type_back = st.selectbox("Select type:", ["inSample", "outSample"])
    pairBacktested = st.selectbox('Select Pair to Backtest:', pairListSelected)

    if st.button('Run Backtest'):
            
        #1) Sortir les informations : Mt, hedgeS1, price of inSample et outSample
        dict_data = vbt_strategyPCI.f_pre_vectorBT(pairListSelected, estimatorData, pairData, stockPrices["inSample"], stockPrices["outSample"])


        dict_pre_backtest = vbt_strategyPCI.f_pre_backtest(dict_output = dict_data, window  = window, entry_threshold = entry_threshold, exit_threshold = exit_threshold)


        #4) Get portfolio backtest
        
        hedgeS1_inSample = dict_data["hedgeS1"][type_back]
        price = dict_data["Price"][type_back]

        pf_dict= vbt_strategyPCI.f_backtest(dict_pre_backtest = dict_pre_backtest,type_back = type_back, hedgeS1 = hedgeS1_inSample, price = price, pairBacktested = pairBacktested, pairData = pairData)

        ## Streamlit app
        pf = pf_dict["pf"]
        entries = pf_dict["positions"]["entries"]
        exits = pf_dict["positions"]["exits"]
        prices1 = pf_dict["Price"]["X1"]
        prices2 = pf_dict["Price"]["X2"]
        

        # Display stats in a more readable format
        st.subheader("Backtest Statistics")
        stats = pf.stats()
        for key, value in stats.items():
            st.text(f"{key}: {value}")
        
        # Portfolio value over time
        st.subheader("Portfolio Value Over Time")
        pf_value_fig = pf.value().vbt.plot()
        st.plotly_chart(pf_value_fig, use_container_width=True)

        # Display log of trades
        st.subheader("Log of Trades")
        st.write(pf.trades.records_readable)
    
        Mt = dict_pre_backtest[type_back]["Mt"].z_score[pairBacktested]
        Mt.index = entries.index
    # Create a new Plotly figure for plotting
        fig = go.Figure()
        
        # Plot the Mt series
        fig.add_trace(go.Scatter(x=Mt.index, y=Mt, mode='lines', name='Spread_Mt'))
    
        print(Mt.loc[entries])
        # Plot entry points
        fig.add_trace(go.Scatter(x=entries.index[entries], y=Mt.loc[entries], mode='markers', 
                                marker=dict(symbol='triangle-up', color='red', size=10), name='Short'))

        # Plot exit points
        fig.add_trace(go.Scatter(x=exits.index[exits], y=Mt.loc[exits], mode='markers', 
                                marker=dict(symbol='triangle-down', color='Green', size=10), name='Exit'))
        
        # Update layout of the figure as needed
        fig.update_layout(title='Spread and Trades', xaxis_title='Date', yaxis_title='Spread Mt')
        
        # Display the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Prices Graph")
        fig = go.Figure()

        # Add prices1 as the first trace
        fig.add_trace(go.Scatter(x=prices1.index, y=prices1, mode='lines', name='Price 1'))
        
        # Plot entry points
        fig.add_trace(go.Scatter(x=entries.index[entries], y=prices1.loc[entries], mode='markers', 
                                marker=dict(symbol='triangle-up', color='Green', size=10), name='Long'))

        # Plot exit points
        fig.add_trace(go.Scatter(x=exits.index[exits], y=prices1.loc[exits], mode='markers', 
                                marker=dict(symbol='triangle-down', color='red', size=10), name='Exit'))
        
            # Add prices2 as the second trace
        fig.add_trace(go.Scatter(x=prices2.index, y=prices2, mode='lines', name='Price 2'))
        
        fig.add_trace(go.Scatter(x=entries.index[entries], y=prices2.loc[entries], mode='markers', 
                                marker=dict(symbol='triangle-up', color='red', size=10), name='Short'))

        # Plot exit points
        fig.add_trace(go.Scatter(x=exits.index[exits], y=prices2.loc[exits], mode='markers', 
                                marker=dict(symbol='triangle-down', color='Green', size=10), name='Exit'))



        # Update layout of the figure as needed
        fig.update_layout(title='Prices Comparison', xaxis_title='Date', yaxis_title='Price', legend_title="Prices")

        # Display the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)


# Streamlit run app.py

if __name__ == "__main__":
    
    run_streamlit_app()
    