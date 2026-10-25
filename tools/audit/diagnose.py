"""Read-only audit of inherited model; writes evidence under docs/audit.

Independent references are derived from the declared state equations, not copied
from the production filter. Run from repository root using the pinned audit env.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
sys.dont_write_bytecode = True
import time
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'PCI Framework'), str(ROOT / 'Simulation/Functions')]
from PCIPipeline import Computation, Definition
from PartialCointegration import partial_cointegration
from PCISimulation import pci_generator


def exact_filter(w, rho, sm, sr, covariance=None):
    """Conditional W[0]; known M0=0,R0=W0, unless pure AR R0=0.

    Optional posterior covariance allows an independent steady-state comparison.
    """
    f = np.diag([rho, 1.0])
    h = np.ones(2)
    q = np.diag([sm * sm, sr * sr])
    a = np.array([w[0], 0.0]) if sr == 0 else np.array([0.0, w[0]])
    p = np.zeros((2, 2)) if covariance is None else covariance.copy()
    ll = 0.0
    states = [a.copy()]
    innovations, variances, gains = [], [], []
    for y in w[1:]:
        a = f @ a
        p = f @ p @ f.T + q
        v = float(y - h @ a)
        variance = float(h @ p @ h)
        k = p @ h / variance
        a = a + k * v
        p = p - np.outer(k, k) * variance
        p = (p + p.T) / 2
        ll -= .5 * (np.log(2 * np.pi * variance) + v*v/variance)
        states.append(a.copy())
        innovations.append(v)
        variances.append(variance)
        gains.append(k)
    return dict(ll=float(ll), states=np.array(states), errors=np.array(innovations),
                variances=np.array(variances), gains=np.array(gains))


def dense_ll(w, rho, sm, sr):
    """Joint conditional Gaussian density via explicit shock loading matrix."""
    n = len(w)-1
    loading_m = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            loading_m[i, j] = sm * rho**(i-j)
    loading_r = sr * np.tril(np.ones((n, n)))
    cov = loading_m @ loading_m.T + loading_r @ loading_r.T
    mean = rho**np.arange(1, n+1)*w[0] if sr == 0 else np.full(n, w[0])
    residual = w[1:]-mean
    chol = np.linalg.cholesky(cov)
    z = np.linalg.solve(chol, residual)
    return float(-.5*(n*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+z@z))


def steady_p(rho, sm, sr):
    f = np.diag([rho, 1.0]); h = np.ones(2)
    q = np.diag([sm*sm, sr*sr]); p = np.zeros((2, 2))
    for _ in range(100000):
        prior = f @ p @ f.T + q
        variance = h @ prior @ h
        k = prior @ h / variance
        update = np.eye(2) - np.outer(k, h)
        new = update @ prior @ update.T
        # Exact observation M+R implies posterior errors sum to zero.
        new = new[0, 0] * np.array([[1., -1.], [-1., 1.]])
        if np.max(np.abs(new-p)) < 1e-12:
            return new, k, float(variance)
        p = new
    raise RuntimeError(f'Riccati iteration did not converge: {rho}, {sm}, {sr}, {p}, {new}')


def capture(call):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        try:
            value = call()
            return dict(value=np.asarray(value).tolist(), warnings=list(dict.fromkeys(str(w.message) for w in ws)))
        except Exception as e:
            return dict(error=type(e).__name__+': '+str(e), warnings=list(dict.fromkeys(str(w.message) for w in ws)))


def git(*args):
    return subprocess.check_output(['git', '-c', 'safe.directory='+str(ROOT).replace('\\','/'), *args], cwd=ROOT, text=True).strip()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke', action='store_true'); args=parser.parse_args()
    report = {'baseline':git('rev-parse','HEAD'), 'main':git('rev-parse','origin/main'),
              'python':platform.python_version(), 'platform':platform.platform(),
              'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','statsmodels','tqdm','xarray']},
              'source_sha256':{}, 'criteria':'validation-protocol.md'}
    for name in ['PCI Framework/PCIPipeline.py','Simulation/Functions/PartialCointegration.py','PCI Framework/PCIStrat.py','Simulation/Functions/ResultReplication.py']:
        report['source_sha256'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    w=np.array([0., .7, -.4, 1.3, -.8, .2, 1.1, -.1])
    model=Computation(w,w)
    comparisons=[]
    for rho,sm,sr in [(.6,1.,1.),(.9,1.,1.),(-.8,.7,1.2),(.99,1.,.1),(.6,0.,1.),(.6,1.,0.)]:
        ex=exact_filter(w,rho,sm,sr); dense=dense_ll(w,rho,sm,sr)
        assert np.isclose(ex['ll'],dense,atol=1e-9,rtol=1e-10)
        p,k,variance=steady_p(rho,sm,sr)
        ss=exact_filter(w,rho,sm,sr,p)
        m,r,e=model.kalman_estimate(w,rho,sm,sr)
        assert np.max(np.abs(m+r-w)) < 1e-10
        assert np.max(np.abs(ss['states']-np.column_stack([m,r]))) < 1e-9
        comparisons.append(dict(rho=rho,sm=sm,sr=sr,exact_ll=ex['ll'],dense_ll=dense,
            legacy_ll=float(model.calc_log_like(w,rho,sm,sr)),steady_correct_ll=ss['ll'],
            innovation_variance=variance,legacy_variance=sm*sm+sr*sr,steady_gain=k.tolist(),
            matrix_vs_legacy_state_max_error=float(np.max(np.abs(ss['states']-np.column_stack([m,r]))))))
    report['likelihood_comparisons']=comparisons
    report['alpha_profile']={str(alpha):float(model.calc_log_like(w-alpha,.6,1,1)) for alpha in [-100,0,100]}
    report['lagvar_constant']=capture(lambda:model.lagvar_estimate_par(np.ones(8)))
    report['lagvar_linear']=capture(lambda:model.lagvar_estimate_par(np.arange(8.)))
    report['zero_noise_ll']=capture(lambda:model.calc_log_like(w,.6,0,0))
    report['negative_noise_filter']=capture(lambda:model.kalman_estimate(w,.6,-1,1))
    with patch.object(Computation,'fit_mle',return_value=np.array([.6,1.,1.])), patch('PCIPipeline.opt.minimize',return_value=SimpleNamespace(success=False,x=np.array([0,1,.6,1,1]),fun=100.,message='forced failure')):
        report['failed_final_fit_accepted']=capture(lambda:model.fit_pci(LRT_mode=False))
    x=pd.Series([1.,2.,3.,4.], index=[3,2,1,0])
    y=x.sort_index()
    report['misaligned_input_accepted']=capture(lambda:Computation(x,y).fit_ols_on_diff())
    future=pd.DataFrame({'gvkey':[1], 'from':['2023-06-01'],'thru':[None],'co_tic':['FUTURE']})
    from PCIData import DataCollection
    obj=DataCollection.__new__(DataCollection); obj.data=future
    report['future_membership_selected']=obj.list_nasdaq_constituents_by_year(2023)
    import PCIData
    raw=pd.read_csv(ROOT/'PCI Framework/data/NasdaqConstituent.csv')
    report['constituent_data']={'rows':len(raw),'tickers':raw.co_tic.nunique(),'from_min':str(raw['from'].min()),'open_intervals':int(raw.thru.isna().sum()),'duplicates':int(raw.duplicated().sum())}
    a=np.arange(1.,9.); b=2*a+np.array([0,1,-1,2,-2,1,0,-1])
    full_m,_,_=Computation(a,b).kalman_estimate(b-2*a,.6,1,1)
    reset_m,_,_=Computation(a[4:],b[4:]).kalman_estimate((b-2*a)[4:],.6,1,1)
    report['filter_reset']={'continuous_test_m':full_m[4:].tolist(),'restarted_test_m':reset_m.tolist()}
    dates=pd.date_range('2023-01-01', periods=4)
    missing_prices=pd.DataFrame({'A':[10.,11.,12.,13.], 'B':[20.,np.nan,24.,26.]},index=dates)
    estimates=pd.DataFrame([[0.,2.,.6,1.,1.]],index=['A/B'])
    report['missing_test_price_rejected']=capture(lambda:Computation.f_create_Mt(
        ['A/B'],estimates,Definition.pairDatabase(['A','B']),missing_prices))
    pairs=Definition.pairDatabase(['A','B'])
    with contextlib.redirect_stdout(io.StringIO()):
        report['subset_pair_index']=capture(lambda:Computation.f_compute_estimate(pairs['pair'].iloc[1:],pd.DataFrame({'A':a,'B':b}),pairs))
    try:
        import ResultReplication as rep
        import xarray as xr
        # All estimates match the cell-specific truth, so correct MSE is zero.
        values=np.array([[[1,.9,.5,1,0]],[[1,.9,1.5,1,0]]])
        ds=xr.Dataset({'estimates_sample_size_100':xr.DataArray(values,dims=['var_param','simulation','estimates'],coords={'var_param':[.5,1.5],'simulation':[0],'estimates':['beta','rho','sigma_m','sigma_r','lrt']})})
        params=[0,1,.9,None,1,.0236]
        first=rep.f_compute_mse(ds,params)
        second=rep.f_compute_mse(ds,params)
        report['mse_repeat']={'first':first.to_array().values.tolist(),'second':second.to_array().values.tolist(),'mutated_params':params}
        # Failure handling: one rejection and one missing estimate become 50%.
        power_ds=xr.Dataset({'estimates_sample_size_100':xr.DataArray(
            np.array([[[-5.],[np.nan]]]), dims=['var_param','simulation','estimates'],
            coords={'var_param':[1.],'simulation':[0,1],'estimates':['lrt']})})
        crit=xr.Dataset({'estimates_sample_size_100':xr.DataArray(
            np.array([[-3.]]),dims=['var_param','quantile'],coords={'var_param':[0],'quantile':[.05]})})
        with patch.object(rep,'f_compute_critical_values_lrt',return_value=crit):
            report['missing_lrt_power']=rep.f_compute_powers_lrt(power_ds,[100],[.05]).to_dict(data=True)
        report['replication_import']='success'
        if args.smoke:
            np.random.seed(20261025)
            start=time.perf_counter()
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                generated=rep.f_generate_data([0,1,.9,None,1,.0236],[.5,1.5],[100,1000],2)
                fitted=rep.f_compute_estimates(generated,[100,1000])
                means=rep.f_compute_means(fitted)
                mse=rep.f_compute_mse(fitted,[0,1,.9,None,1,.0236])
            report['reduced_notebook_reproduction']={
                'seed':20261025,'n_replications_per_cell':2,
                'sample_sizes':[100,1000],'varying_sigma_m':[.5,1.5],
                'number_of_fits':8,'seconds':time.perf_counter()-start,
                'means':means.to_dict(data=True),'mse':mse.to_dict(data=True),
                'warnings':list(dict.fromkeys(str(w.message) for w in ws))}

    except Exception as e:
        report['replication_import_or_mse_error']=type(e).__name__+': '+str(e)
    if args.smoke:
        report['smoke']=[]
        for rho in [.6,.9]:
            for repno in range(3):
                seed=20261025+repno+int(100*rho)
                np.random.seed(seed)
                gen=pci_generator(100,0,1,rho,1,1,.0236)
                x,y,_,_,_=gen.f_generate_pci()
                estimator=partial_cointegration(x,y)
                calls=[]
                import scipy.optimize as opt
                original=opt.minimize
                def tracked(*a,**kw):
                    result=original(*a,**kw)
                    calls.append({'success':bool(result.success),'fun':float(result.fun),'message':str(result.message)})
                    return result
                start=time.perf_counter()
                with patch('PartialCointegration.opt.minimize',side_effect=tracked):
                    result=capture(lambda:estimator.fit_pci())
                result.update(seed=seed,true_rho=rho,n=100,seconds=time.perf_counter()-start,optimizer_calls=len(calls),optimizer_failures=sum(not c['success'] for c in calls),final_calls=calls[-2:])
                report['smoke'].append(result)
    def clean(value):
        if isinstance(value, dict):
            return {k:clean(v) for k,v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value
    report=clean(report)
    (ROOT/'docs/audit/diagnostic-results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=True))


if __name__=='__main__':
    main()
