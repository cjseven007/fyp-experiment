"""S005-informed synthetic benchmark. Run --help for local input paths."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from characterization.analysis import CharacterizationConfig, analyze_window
from ph_benchmark.core import (SCENARIOS, smooth, block_slopes, bocpd_scores,
                               slope_scores, alarms, event_metrics, generate)


def characterize(data, seconds):
    rows = []
    for start in range(0,len(data)-seconds+1,seconds):
        metric, _ = analyze_window(data.iloc[start:start+seconds], CharacterizationConfig())
        rows.append(dict(start_seconds=start, window_seconds=seconds, **metric))
    return pd.DataFrame(rows)


def calibrate(path, out):
    d = pd.read_csv(path)
    d['timestamp'] = pd.to_datetime(d['timestamp'],errors='raise')
    y = pd.to_numeric(d.ph_value,errors='raise').to_numpy()
    if len(y)<120 or not np.isfinite(y).all():
        raise ValueError('Need at least 120 finite pH observations')
    if not np.allclose(d.timestamp.diff().dt.total_seconds().dropna(),1,rtol=0,atol=.05):
        raise ValueError('This initial benchmark requires a continuous 1-Hz segment')
    if 'usable_ph' not in d or not d.usable_ph.astype(str).str.lower().eq('true').all():
        raise ValueError('Input must be prepared with every row usable_ph=True')
    if not np.allclose(y*100,np.round(y*100),atol=1e-7):
        raise ValueError('Observed 0.01-pH grid assumption fails; review calibration')
    s = characterize(d,60)
    s.to_csv(out/'source_window_statistics.csv',index=False)
    sensitivity = pd.concat([characterize(d,w) for w in (30,60,120)],ignore_index=True)
    sensitivity.to_csv(out/'window_sensitivity.csv',index=False)
    sensitivity.groupby('window_seconds').agg(n_windows=('ph_mean','size'),
        median_slope=('slope_ph_per_minute','median'),median_residual_sd=('residual_sd','median'),
        median_lag1=('residual_lag1_correlation','median'),
        median_unchanged=('unchanged_step_fraction','median')).to_csv(out/'window_sensitivity_summary.csv')
    neg = s.loc[s.slope_ph_per_minute < -1e-8,'slope_ph_per_minute']
    if neg.empty:
        raise ValueError('No declining windows; review scenario choice')
    profile = dict(source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        source_filename=Path(path).name, samples=len(d), start=str(d.timestamp.iloc[0]),end=str(d.timestamp.iloc[-1]),
        start_ph=float(y[0]),end_ph=float(y[-1]),observed_grid_ph=.01,
        decline_scale_ph_per_minute=float(-neg.median()),
        positive_residual_sd_median=float(s.loc[s.residual_sd>1e-10,'residual_sd'].median()),
        residual_lag1_median=float(s.residual_lag1_correlation.median()),
        unchanged_fraction_median=float(s.unchanged_step_fraction.median()),
        note='Residual scale and dependence are descriptive. Latent AR noise is assumed, not identified. Grid is observed recording precision, not certified sensor resolution.')
    # Only describe other channels; do not assume pH-only preparation validated them.
    context=[]
    fig,axes=plt.subplots(4,1,figsize=(11,9),sharex=True,layout='constrained')
    for ax,name in zip(axes,['ph','temperature','pressure','co2']):
        if name+'_value' not in d: continue
        v=pd.to_numeric(d[name+'_value'],errors='coerce')
        labels={'ph':'pH','temperature':'Temperature (°C)','pressure':'Pressure (barg)','co2':'CO₂ (ppm)'}
        ax.plot(np.arange(len(d))/60,v,lw=1);ax.set_ylabel(labels[name]);ax.grid(alpha=.2)
        context.append(dict(channel=name,start=v.iloc[0],end=v.iloc[-1],minimum=v.min(),maximum=v.max(),
                            unique_values=v.nunique(),nonfinite=int((~np.isfinite(v)).sum())))
    axes[-1].set_xlabel('Elapsed time (minutes)')
    axes[0].set_title('S005 context: pH-screened segment; other channels require their own quality checks')
    fig.savefig(out/'source_context.png',dpi=160);plt.close(fig)
    pd.DataFrame(context).to_csv(out/'channel_context.csv',index=False)
    (out/'calibration.json').write_text(json.dumps(profile,indent=2))
    return d,profile


def cases(profile, seeds):
    for sid,scenario in enumerate(SCENARIOS):
        for mult in (0.,1.,2.):
            for seed in seeds:
                # Same realization reused for EVERY candidate; seed namespaces disjoint by split.
                yield scenario,mult,seed,generate(profile,scenario,seed+sid*100000,mult)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--segment',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--dev-reps',type=int,default=8)
    p.add_argument('--test-reps',type=int,default=20)
    a=p.parse_args()
    if not (1<=a.dev_reps<=1000 and 1<=a.test_reps<=1000): raise ValueError('Repetitions must be 1..1000')
    out=Path(a.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError('Choose a new/empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    data,profile=calibrate(a.segment,out)
    config=dict(dev_seeds=list(range(1000,1000+a.dev_reps)),test_seeds=list(range(10000,10000+a.test_reps)),
        scenario_seconds=1200,block_seconds=30,metric_warmup_seconds=180,matching_tolerance_seconds=180,
        noise_multipliers=[0,1,2],hazard_per_block=.05,prior_mean_normalized=-.5,prior_sd_normalized=1.,
        assumptions='Synthetic endpoint-like transitions only; no validated real saturation label. Real replay has no accuracy metric.')
    (out/'benchmark_config.json').write_text(json.dumps(config,indent=2))
    dev=list(cases(profile,config['dev_seeds']))
    grid=[('raw',1)]+[('ema',w) for w in (5,15,31,61)]+[('wsg',w) for w in (15,31,61)]
    tuning=[]
    for family,size in grid:
        for scenario,mult,seed,(_,clean,y,_) in dev:
            error=smooth(y,family,size)[180:]-clean[180:]
            tuning.append(dict(family=family,size=size,scenario=scenario,noise_multiplier=mult,seed=seed,
                               rmse=float(np.sqrt(np.mean(error**2)))))
    pd.DataFrame(tuning).to_csv(out/'smoothing_tuning_trials.csv',index=False)
    means=pd.DataFrame(tuning).groupby(['family','size'],as_index=False).rmse.mean()
    means.to_csv(out/'smoothing_tuning_summary.csv',index=False)
    chosen={f:int(g.loc[g.rmse.idxmin(),'size']) for f,g in means.groupby('family')}
    scale=profile['decline_scale_ph_per_minute']
    settings={}; detector_trials=[]
    for family,size in chosen.items():
        # Known synthetic targets used ONLY to estimate model SD on DEVELOPMENT null cases.
        errors=[]
        for scenario,mult,seed,(_,clean,y,_) in dev:
            if scenario in ('stable','decline'):
                times,s=block_slopes(smooth(y,family,size)); _,truth=block_slopes(clean)
                errors.extend(((s-truth)/scale)[times>=180])
        sigma=max(.05,float(np.std(errors,ddof=1)))
        cached=[]
        for scenario,mult,seed,(_,clean,y,target) in dev:
            times,s=block_slopes(smooth(y,family,size)); z=s/scale
            cached.append((scenario,mult,seed,target,times,{'slope':slope_scores(z),'bocpd':bocpd_scores(z,sigma)}))
        settings[family]={'size':size,'normalized_observation_sd':sigma,'detectors':{}}
        for detector,thresholds in [('slope',(.25,.5,.75,1.0,1.5)),('bocpd',(.5,.7,.85,.95,.99))]:
            options=[]
            for threshold in thresholds:
                rows=[]
                for scenario,mult,seed,target,times,scores in cached:
                    met=event_metrics(alarms(times,scores[detector],threshold),target)
                    rows.append(met)
                r=pd.DataFrame(rows)
                missed=r.missed.sum()/r.target_count.sum()
                false=r.false_alarms.mean()
                delay=r.delay_from_onset_seconds.mean()
                # Explicit development objective, including failures and nuisance alarms.
                loss=2*missed+false+(.25*delay/180 if np.isfinite(delay) else .25)
                option=dict(family=family,detector=detector,threshold=threshold,loss=loss,
                            miss_rate=missed,false_alarms_per_trial=false,mean_delay_seconds=delay)
                options.append(option);detector_trials.append(option)
            winner=min(options,key=lambda x:x['loss'])
            settings[family]['detectors'][detector]=winner['threshold']
    pd.DataFrame(detector_trials).to_csv(out/'detector_tuning.csv',index=False)
    (out/'selected_settings.json').write_text(json.dumps(settings,indent=2))
    print('Development selection frozen; evaluating held-out realizations.',flush=True)
    smooth_rows=[]; detection=[]; diagnostic_rows=[]; traces=out/'traces';traces.mkdir()
    for scenario,mult,seed,(t,clean,y,target) in cases(profile,config['test_seeds']):
        trace=pd.DataFrame(dict(elapsed_seconds=t,clean_ph=clean,observed_ph=y))
        trace['target_onset_seconds']=np.nan if target is None else target[0]
        trace['target_end_seconds']=np.nan if target is None else target[1]
        if seed==10000:
            diagnostic=characterize(pd.DataFrame({'timestamp':pd.Timestamp('2000-01-01')+pd.to_timedelta(t,unit='s'),'ph_value':y}),60)
            diagnostic_rows.append(dict(scenario=scenario,noise_multiplier=mult,
                median_residual_sd=diagnostic.residual_sd.median(),
                median_lag1=(diagnostic.residual_lag1_correlation.dropna().median()
                             if diagnostic.residual_lag1_correlation.notna().any() else np.nan),
                median_unchanged=diagnostic.unchanged_step_fraction.median()))
        fig=None
        if seed==10000 and mult==1:
            fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
            axes[0].plot(t/60,y,color='gray',alpha=.6,label='Observed synthetic')
            axes[0].plot(t/60,clean,color='black',label='Known clean target')
        for family,setting in settings.items():
            filtered=smooth(y,family,setting['size']);trace[family]=filtered
            error=filtered-clean
            transition=np.zeros(len(t),bool) if target is None else (t>=target[0])&(t<=target[1]+180)
            transition &= t>=180
            smooth_rows.append(dict(scenario=scenario,noise_multiplier=mult,seed=seed,family=family,
                rmse=float(np.sqrt(np.mean(error[180:]**2))),
                transition_rmse=float(np.sqrt(np.mean(error[transition]**2))) if transition.any() else np.nan))
            times,s=block_slopes(filtered);z=s/scale
            scores={'slope':slope_scores(z),'bocpd':bocpd_scores(z,setting['normalized_observation_sd'])}
            for detector,score in scores.items():
                events=alarms(times,score,setting['detectors'][detector])
                met=event_metrics(events,target)
                detection.append(dict(scenario=scenario,noise_multiplier=mult,seed=seed,family=family,
                                      detector=detector,alarm_seconds=json.dumps(events),**met))
                trace.loc[times,f'{family}_{detector}_score']=score
                trace[f'{family}_{detector}_alarm']=np.isin(t,events)
                if fig is not None and detector=='bocpd':
                    axes[1].plot(times/60,score,label=family)
                    for ev in events: axes[0].axvline(ev/60,alpha=.2,ls=':')
            if fig is not None and family!='raw': axes[0].plot(t/60,filtered,label=family,lw=1)
        trace.to_csv(traces/f'{scenario}_noise{mult:g}_seed{seed}.csv',index=False)
        if fig is not None:
            if target is not None:
                for ax in axes: ax.axvspan(target[0]/60,max(target[1],target[0]+1)/60,color='green',alpha=.2)
            axes[0].set_title(f'{scenario}: synthetic assumptions, seed 10000; dotted lines = BOCPD alarms')
            axes[0].set_ylabel('pH');axes[1].set_ylabel('Recent-change probability');axes[1].set_xlabel('Elapsed minutes')
            for ax in axes: ax.legend();ax.grid(alpha=.2)
            fig.savefig(out/f'{scenario}.png',dpi=160);plt.close(fig)
    pd.DataFrame(diagnostic_rows).to_csv(out/'synthetic_observed_diagnostics.csv',index=False)
    sm=pd.DataFrame(smooth_rows);sm.to_csv(out/'test_smoothing_trials.csv',index=False)
    sm.groupby(['scenario','noise_multiplier','family']).agg(n=('rmse','size'),rmse_mean=('rmse','mean'),
        rmse_sd=('rmse','std'),transition_rmse_mean=('transition_rmse','mean')).to_csv(out/'test_smoothing_summary.csv')
    de=pd.DataFrame(detection);de.to_csv(out/'test_detection_trials.csv',index=False)
    summary=de.groupby(['scenario','noise_multiplier','family','detector']).agg(
        n=('seed','size'),targets=('target_count','sum'),detected=('detected','sum'),missed=('missed','sum'),
        false_alarms=('false_alarms','sum'),negative_hours=('negative_exposure_hours','sum'),
        median_delay_seconds=('delay_from_onset_seconds','median'),
        mean_delay_seconds=('delay_from_onset_seconds','mean'),
        median_delay_after_interval_seconds=('delay_after_interval_seconds','median')).reset_index()
    summary['recall']=summary.detected/summary.targets.replace(0,np.nan)
    summary['false_alarms_per_negative_hour']=summary.false_alarms/summary.negative_hours
    summary.to_csv(out/'test_detection_summary.csv',index=False)
    # Real replay: no physical endpoint truth, no accuracy/RMSE claims.
    real=pd.DataFrame({'timestamp':data.timestamp,'elapsed_seconds':np.arange(len(data)),'ph_value':data.ph_value})
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained');ax.plot(real.elapsed_seconds/60,real.ph_value,label='Recorded pH',color='gray')
    real_events=[]
    for family,setting in settings.items():
        filtered=smooth(data.ph_value.to_numpy(),family,setting['size']);real[family]=filtered
        times,s=block_slopes(filtered);z=s/scale
        for detector,score in [('slope',slope_scores(z)),('bocpd',bocpd_scores(z,setting['normalized_observation_sd']))]:
            events=alarms(times,score,setting['detectors'][detector])
            real.loc[times,f'{family}_{detector}_score']=score
            for ev in events: real_events.append(dict(family=family,detector=detector,elapsed_seconds=ev,
                timestamp=str(data.timestamp.iloc[int(ev)]),interpretation='Unvalidated statistical alarm'))
        if family!='raw': ax.plot(real.elapsed_seconds/60,filtered,label=family)
    real.to_csv(out/'real_replay.csv',index=False)
    pd.DataFrame(real_events,columns=['family','detector','elapsed_seconds','timestamp','interpretation']).to_csv(out/'real_alarms.csv',index=False)
    ax.legend();ax.set(xlabel='Elapsed minutes',ylabel='pH',title='S005 causal replay: no validated saturation label');ax.grid(alpha=.2)
    fig.savefig(out/'real_replay.png',dpi=160);plt.close(fig)
    print(f'Saved benchmark to {out.resolve()}')


if __name__=='__main__':
    main()
