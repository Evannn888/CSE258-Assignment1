"""Historical terminal-event validation for Task 6.

For a cutoff before the dataset end, use users having observed training events
after the cutoff. Their unknown test event is then guaranteed to be after it.
Hold out the final event before the cutoff, and predict its time using only
earlier history. This leaves exactly one event before the artificial cutoff,
matching the terminal-event task rather than leaving two unobserved events.
"""
from delta_research import *
from datetime import datetime, timezone


def prepare_cutoff(year, month=1):
    cut = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())
    cache = ROOT / f'experiments/delta_cutoff_{year}_{month}.npz'
    if cache.exists():
        return dict(np.load(cache))
    a, users, items = base.load(ROOT / 'assignment1_2026/task_6_train.csv')
    allhist = [[] for _ in users]
    for u, i, t, _ in a:
        allhist[u].append((int(i), int(t)))
    histories = [[row for row in h if row[1] <= cut] for h in allhist]
    # Need at least three known events up to cutoff plus one after it.
    qu = np.array([u for u in range(len(users)) if len(histories[u])>=3 and allhist[u][-1][1] > cut])
    qi = np.array([histories[u][-1][0] for u in qu])
    truth = np.array([histories[u][-1][1]-histories[u][-2][1] for u in qu])/DAY
    for u in qu:
        histories[u] = histories[u][:-1]
    original_cut = base.T_MAX_VALID
    base.T_MAX_VALID = cut
    ts = structures(histories, len(items))
    vf = base.extract_delta_features(histories, ts, qu, qi)
    x = extra_features(histories, ts, qu, qi)
    base.T_MAX_VALID = original_cut
    data = dict(x=x,y=truth,vf=np.asarray(vf),users=qu,cutoff=cut)
    np.savez_compressed(cache, **data)
    base.log(f'Saved cutoff {year}-{month}: {x.shape}')
    return data


def main():
    data = prepare()
    vf = (data['vf'][0].astype(bool), *data['vf'][1:])
    init=np.array([.13,.05,.08,.05,.12,.04,.05,.50,.65,.35,.26,.45,.35,.70])
    tr = np.random.default_rng(258).permutation(len(data['y']))[:int(.6*len(data['y']))]
    opt=minimize(lambda p: np.mean(abs(base.apply_delta_params(tuple(f[tr] for f in vf),p)/DAY-data['y'][tr])),init,method='Nelder-Mead',options={'maxiter':800})
    base.log('Local parameters: '+str(opt.x))
    records=[]
    for year, month in [(2016,1),(2017,1),(2017,7)]:
        d=prepare_cutoff(year,month)
        f=(d['vf'][0].astype(bool),*d['vf'][1:]); y=d['y']
        ix=np.random.default_rng(258).permutation(len(y)); tr,au=np.split(ix,[len(y)//2])
        fit=minimize(lambda p: np.mean(abs(base.apply_delta_params(tuple(z[tr] for z in f),p)/DAY-y[tr])),opt.x,method='Nelder-Mead',options={'maxiter':1000})
        for name,p in [('local',opt.x),('terminal',fit.x)]:
            pred=base.apply_delta_params(f,p)/DAY
            rec=dict(year=year,month=month,n=len(y),name=name,mae_seconds=float(np.mean(abs(pred[au]-y[au]))*DAY),params=p.tolist(),target_median_days=float(np.median(y)),prediction_median_days=float(np.median(pred)))
            records.append(rec);base.log(str(rec))
    (ROOT/'experiments/delta_cutoff_results.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
