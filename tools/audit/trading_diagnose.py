"""Synthetic execution audit using the real vectorbt engine, without market data."""
import importlib.metadata
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import warnings
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'PCI Framework'))
import vectorbt as vbt
from PCIPipeline import Definition
from PCIStrat import vbt_strategyPCI

def run(beta):
    dates=pd.date_range('2023-01-01',periods=5)
    prices=pd.DataFrame({'A':[10.,11.,12.,13.,14.], 'B':[20.,22.,18.,17.,16.], 'UNTRADED':[30.,31.,32.,33.,34.]},index=dates)
    entry=pd.DataFrame({'A/B':[False,True,False,False,False]},index=dates)
    exit=pd.DataFrame({'A/B':[False,False,False,True,False]},index=dates)
    hedge=pd.DataFrame({'A/B':beta/prices['B']},index=dates)
    data={'outSample':{'clean_upper_crossed':entry,'clean_lower_crossed':exit}}
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        try:
            result=vbt_strategyPCI.f_backtest(data,'outSample',hedge,prices,'A/B',Definition.pairDatabase(['A','B']))
            pf=result['pf']
            return {'orders':pf.orders.records_readable.astype(str).to_dict('records'),
                    'group_lens':pf.wrapper.grouper.get_group_lens().tolist(),
                    'cash_sharing':bool(pf.cash_sharing),
                    'initial_cash':np.asarray(pf.init_cash).tolist(),
                    'last_assets':np.asarray(pf.assets().iloc[-1]).tolist(),
                    'warnings':list(dict.fromkeys(str(w.message) for w in ws))}
        except Exception as e:
            return {'error':type(e).__name__+': '+str(e),'warnings':list(dict.fromkeys(str(w.message) for w in ws))}
report={'versions':{p:importlib.metadata.version(p) for p in ['vectorbt','numba','numpy','pandas','scipy']},'positive_beta':run(1.),'negative_beta':run(-1.)}
(ROOT/'docs/audit/trading-diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
