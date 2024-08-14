import vectorbt as vbt
from PCIPipeline import *
from PCIData import *
import pandas as pd
import numpy as np


class vbt_strategyPCI:
        
    def f_pre_vectorBT(pairListSelected, estimatorData, pairData, inSample ,outSample) -> dict:
        
        Mt_inSample, hedgeS1_inSample = Computation.f_create_Mt(pairListSelected, estimatorData, pairData, inSample)
        Mt_outSample, hedgeS1_outSample = Computation.f_create_Mt(pairListSelected, estimatorData, pairData, outSample)
        
        stockList = Computation.get_stock_list(pairData, pairListSelected)
        
        dict_output = { "Mt":       {"inSample" : Mt_inSample,
                                    "outSample" : Mt_outSample},
                        "hedgeS1": {"inSample" : hedgeS1_inSample,
                                    "outSample": hedgeS1_outSample},
                        "Price": {"inSample": inSample[stockList],
                                "outSample": outSample[stockList]}}
        
        return dict_output


    def custom_indicator(Mt : np.array, window: int = 20):
        """
        La fonction qui sert comme indicateur sur mesure pour vectorBT 
        """
        
        Mt_df = pd.DataFrame(Mt)
        
        # Mt_zscore = Mt_df.apply(func=lambda x: Computation.calculate_zscores_historic(x, initial_in_sample_size=2), axis=0) # Normalisé entre -1 et 1
        Mt_zscore = Mt_df.apply(func=lambda x: Computation.rolling_z_score(x, window), axis=0) # Normalisé 
        return Mt_zscore


    def f_get_Mt_normalize(Mt : dict, window: int) -> vbt.indicators.factory:

        strategyBase = vbt.IndicatorFactory(
                                                class_name = "StrategyBase",
                                                short_name = "StratBase",
                                                input_names = ["Mt"],
                                                param_names= ["window"],
                                                output_names = ["z_score"],
                                                ).from_apply_func(
                                                                vbt_strategyPCI.custom_indicator,
                                                                window = window
                                                                )
        strategy = strategyBase.run(Mt)
        return strategy



    def f_get_signals(strategy: vbt.indicators.factory , entry_threshold: float, exit_threshold: float) -> tuple:
        
        # Generate entry signals (Z-score crosses above the entry threshold)
        upper_crossed = strategy.z_score.vbt.crossed_above(entry_threshold)

        # Generate exit signals (Z-score crosses below the exit threshold)
        lower_crossed = strategy.z_score.vbt.crossed_below(exit_threshold)

        clean_upper_crossed, clean_lower_crossed = upper_crossed.vbt.signals.clean(lower_crossed)  
        
        return (clean_upper_crossed, clean_lower_crossed)


    def f_backtest(dict_pre_backtest : dict, type_back : str, hedgeS1: pd.DataFrame, price: pd.DataFrame, pairBacktested: str,  pairData: pd.DataFrame) -> dict:

            
            clean_upper_crossed = dict_pre_backtest[type_back]["clean_upper_crossed"]
            clean_lower_crossed = dict_pre_backtest[type_back]["clean_lower_crossed"]
            
            positions_fill = price.copy()
            positions_fill.iloc[:, :] = False

            # Now, for sizing, long_entries, etc., we need independent copies of positions_fill
            sizing = positions_fill.copy()
            sizing.iloc[:, :] = 0
            long_entries = positions_fill.copy()
            short_entries = positions_fill.copy()
            short_exits = positions_fill.copy()
            long_exits = positions_fill.copy()

            # Directly access the first column name (pair)

            S1, S2 = Computation.get_stock1_stock2(pairBacktested, pairData)

            # Les prix
            X1, X2 = Computation.get_stock_price(price, [S1, S2])

            s1_sizing =  hedgeS1.loc[:, pairBacktested]

            s2_sizing = 1 / X2

            sizing.loc[clean_upper_crossed[pairBacktested].values, S2] =  s2_sizing.values[clean_upper_crossed[pairBacktested]]

            sizing.loc[clean_upper_crossed[pairBacktested].values, S1] = s1_sizing.values[clean_upper_crossed[pairBacktested].values]

            short_entries.loc[clean_upper_crossed[pairBacktested].values, S2] = True
            long_entries.loc[clean_upper_crossed[pairBacktested].values, S1] = True

            short_exits.loc[clean_lower_crossed[pairBacktested].values, S2] = True
            long_exits.loc[clean_lower_crossed[pairBacktested].values, S1] = True


            pf = vbt.Portfolio.from_signals(
                price,
                entries = long_entries,
                exits = long_exits,
                short_entries = short_entries,
                short_exits= short_exits,
                size = sizing,
                size_type= "amount",
                fees= 0.001,
                slippage=0.001,
                init_cash = 1)  # This line integrates your sizing logic

            return {"pf": pf, "Price": {"X1" : X1, "X2": X2}, "positions" : {"entries" : short_entries.loc[:, S2], "exits": short_exits.loc[:, S2]}, "sizing": sizing} 
        
        

    def f_pre_backtest(dict_output : dict, window : int, entry_threshold: float, exit_threshold: float) -> dict:
            
        # 2 Normaliser nos time series Mt pour chaque pair à l'aide vbt indicator factory
        
        Mt_inSample = dict_output["Mt"]["inSample"]
        strategy_inSample = vbt_strategyPCI.f_get_Mt_normalize(Mt = Mt_inSample, window = window)
        
        Mt_outSample = dict_output["Mt"]["outSample"]
        strategy_outSample = vbt_strategyPCI.f_get_Mt_normalize(Mt = Mt_outSample, window = window)
        
        
        clean_upper_crossed_inSample, clean_lower_crossed_inSample = vbt_strategyPCI.f_get_signals(strategy= strategy_inSample , entry_threshold = entry_threshold, exit_threshold = exit_threshold)
        
        clean_upper_crossed_outSample, clean_lower_crossed_outSample = vbt_strategyPCI.f_get_signals(strategy= strategy_outSample , entry_threshold = entry_threshold, exit_threshold = exit_threshold)

        return {"inSample": {"clean_upper_crossed" : clean_upper_crossed_inSample, "clean_lower_crossed" : clean_lower_crossed_inSample, "Mt": strategy_inSample},
                "outSample" : {"clean_upper_crossed" : clean_upper_crossed_outSample, "clean_lower_crossed" : clean_lower_crossed_outSample, "Mt": strategy_outSample}}