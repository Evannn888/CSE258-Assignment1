"""Residual anchor-item ridge/EASE and held-out feature comparisons.

Uses only positive ratings from the designated fit split. Anchor targets have
an explicit zero-self-coefficient constraint, including self-input removal.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import assignment1 as base
import production_ratings as ratings

ROOT = Path(__file__).resolve().parents[1]


def fit_anchor_ridge(a_rated, nu, ni, n_anchors=2000, shrink=8., normalized=False):
    """Prepare one ridge Gram and sparse cross-covariance; lambda is deferred."""
    uc = np.bincount(a_rated[:, 0], minlength=nu)
    ic = np.bincount(a_rated[:, 1], minlength=ni)
    mu, bu, bi = base.als_biases_custom(a_rated, nu, ni, uc, ic, 8.,12.,n_iter=20)
    residual = a_rated[:,3] - mu - bu[a_rated[:,0]] - bi[a_rated[:,1]]
    prior = float(np.mean(residual**2))
    std = np.sqrt((np.bincount(a_rated[:,0],weights=residual**2,minlength=nu)+5*prior)/(uc+5))+.1
    if normalized:
        residual = residual/std[a_rated[:,0]]
    R = csr_matrix((residual.astype(np.float32),(a_rated[:,0],a_rated[:,1])),shape=(nu,ni))
    anchors = np.argsort(-ic,kind='stable')[:min(n_anchors,int((ic>0).sum()))]
    anchor_index=np.full(ni,-1,dtype=np.int32)
    anchor_index[anchors]=np.arange(len(anchors),dtype=np.int32)
    energy=np.asarray(R.power(2).sum(axis=0)).ravel()
    scale=1./np.sqrt(energy[anchors]+shrink)
    X=R[:,anchors].multiply(scale[None,:]).tocsr()
    G=(X.T@X).toarray().astype(np.float64)
    C=(X.T@R).tocsc()
    return dict(R=R,X=X,G=G,C=C,anchors=anchors,anchor_index=anchor_index,
                scale=scale,std=std,normalized=normalized,nu=nu,ni=ni)


@njit(cache=True)
def _anchor_predictions(qpos, qi, lat, cp, ci, cv, apos, scale, pdiag, self_value):
    out=np.zeros(len(qi),dtype=np.float64)
    ni=len(apos)
    for z in range(len(qi)):
        i=qi[z]
        if i>=ni or qpos[z]<0:continue
        row=qpos[z]; a=apos[i]
        if a>=0:
            out[z] = -lat[row,a]/(scale[a]*pdiag[a])+self_value[z]
        else:
            for j in range(cp[i],cp[i+1]):
                out[z]+=lat[row,ci[j]]*cv[j]
    return out


def anchor_ridge_features(state, qu, qi, lambdas=(.1,1.,10.)):
    X,R=state['X'],state['R']
    valid=qu<state['nu']
    unique=np.unique(qu[valid])
    qpos=np.full(len(qu),-1,dtype=np.int64)
    qpos[valid]=np.searchsorted(unique,qu[valid])
    selfval=np.zeros(len(qu))
    v=valid&(qi<state['ni'])
    selfval[v]=np.asarray(R[qu[v],qi[v]]).ravel()
    c=state['C']
    columns=[]
    for lam in lambdas:
        g=state['G'].copy()
        g.flat[::g.shape[0]+1]+=lam
        P=cho_solve(cho_factor(g,lower=True,overwrite_a=True,check_finite=False),np.eye(len(g)),check_finite=False)
        lat=np.asarray(X[unique]@P).astype(np.float32)
        pred=_anchor_predictions(qpos,qi,lat,c.indptr,c.indices,c.data,state['anchor_index'],state['scale'],np.diag(P),selfval)
        if state['normalized']:
            pred[valid]*=state['std'][qu[valid]]
        columns.append(pred)
    return np.column_stack(columns)


def residual_ridge_tune(task, features, tag):
    data=np.load(ROOT/f'experiments/v3_base_{task}.npz')
    y=data['y'].astype(float); X=data['X']; pred=data['pred']
    tr=np.arange(20000)[y[:20000]>0]
    tu=np.arange(20000,35000)[y[20000:35000]>0]
    config=ratings.RATING_CONFIGS[task]
    old=ratings.StandardizedRidge(config['best_l2']).fit(X[tr],y[tr])
    raw=old.predict_raw(X)
    old_extra=pred-raw
    baseline=float(np.mean((pred[tu]-y[tu])**2))
    records=[]; best=(baseline,pred,None)
    # Stack all base features and new features; the existing nonlinear residual
    # correction remains fixed. Tune new feature shrinkage by standardized ridge.
    for reg in [40.,200.,1000.,3500.]:
        xx=np.column_stack([X,features])
        model=ratings.StandardizedRidge(reg).fit(xx[tr],y[tr])
        for eta in [0.,.5,1.]:
            pp=np.clip(model.predict_raw(xx)+eta*old_extra,1,5)
            mse=float(np.mean((pp[tu]-y[tu])**2))
            rec=dict(reg=reg,eta=eta,tune_mse=mse)
            records.append(rec)
            if mse<best[0]:best=(mse,pp,rec)
    result=dict(task=task,tag=tag,baseline_tune=baseline,best_tune=best[0],selected=best[2],trials=records)
    np.savez_compressed(ROOT/f'experiments/v3_neighbors_pred_{task}_{tag}.npz',pred=best[1],features=features)
    (ROOT/f'experiments/v3/neighbors_{task}_{tag}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(result),flush=True)
    return result


def fit_star_distributions(a_rated, nu, ni):
    uc=np.zeros((nu+1,5));ic=np.zeros((ni+1,5))
    stars=a_rated[:,3].astype(np.int64)-1
    np.add.at(uc,(a_rated[:,0],stars),1.)
    np.add.at(ic,(a_rated[:,1],stars),1.)
    global_prob=np.bincount(stars,minlength=5).astype(float)
    global_prob/=global_prob.sum()
    return dict(uc=uc,ic=ic,global_prob=global_prob)


def star_distribution_features(state, qu, qi):
    uc,ic,g=state['uc'],state['ic'],state['global_prob']
    uq=uc[qu];iq=ic[qi]
    cols=[]
    vals=np.arange(1,6,dtype=float)
    for su in [3.,10.,30.]:
        pu=(uq+su*g)/(uq.sum(axis=1)[:,None]+su)
        for si in [3.,10.,30.]:
            pi=(iq+si*g)/(iq.sum(axis=1)[:,None]+si)
            post=pu*pi/g
            post/=post.sum(axis=1,keepdims=True)
            mean=post@vals
            cols.extend([mean,post@(vals**2)-mean**2])
    pu=(uq+10*g)/(uq.sum(axis=1)[:,None]+10)
    pi=(iq+10*g)/(iq.sum(axis=1)[:,None]+10)
    cols.extend([pu,pi,pu*pi,pu/pi])
    return np.column_stack(cols)


@njit(cache=True)
def _summarize_neighbors(sim, values, adjusted):
    # Three similarities, signed/positive, three k values, two denominators.
    result=np.zeros(42)
    col=0
    for variant in range(3):
        for positive in range(2):
            w=sim[variant].copy()
            if positive:
                for j in range(len(w)):
                    if w[j]<0:w[j]=0.
            order=np.argsort(-np.abs(w))
            for k_index in range(3):
                k=(5,20,len(w))[k_index]
                numerator=0.;denominator=0.;adjust=0.
                for j in order[:min(k,len(order))]:
                    numerator+=w[j]*values[j]
                    denominator+=abs(w[j])
                    adjust+=w[j]*adjusted[j]
                result[col]=numerator/(denominator+.1);col+=1
                result[col]=numerator/(denominator+1.);col+=1
                if variant==1:
                    result[36+positive*3+k_index]=adjust/(denominator+1.)
    return result


@njit(cache=True)
def topk_residual_features(qu,qi,up,ui,ur,ip,iu,ir,ue,ie):
    nu=len(up)-1;ni=len(ip)-1
    out=np.zeros((len(qu),84))
    present=np.zeros(nu,np.bool_); target=np.zeros(nu)
    dot=np.zeros(nu);sx=np.zeros(nu);sy=np.zeros(nu);xx=np.zeros(nu);yy=np.zeros(nu);count=np.zeros(nu)
    for z in range(len(qu)):
        u=qu[z];i=qi[z]
        if u>=nu or i>=ni:continue
        us,uep=up[u],up[u+1];is_,iep=ip[i],ip[i+1]
        if us==uep or is_==iep:continue
        for p in range(is_,iep):
            v=iu[p];present[v]=True;target[v]=ir[p]
        n_h=uep-us
        sims=np.zeros((3,n_h));values=np.zeros(n_h);adjusted=np.zeros(n_h)
        for n,p in enumerate(range(us,uep)):
            j=ui[p];ru=ur[p];values[n]=ru
            if j==i:continue
            ds=0.;xs=0.;ys=0.;x2=0.;y2=0.;nc=0.
            for pos in range(ip[j],ip[j+1]):
                v=iu[pos]
                if present[v] and v!=u:
                    a=target[v];b=ir[pos]
                    ds+=a*b;xs+=a;ys+=b;x2+=a*a;y2+=b*b;nc+=1
                    dot[v]+=ru*b;sx[v]+=ru;sy[v]+=b;xx[v]+=ru*ru;yy[v]+=b*b;count[v]+=1
            if nc>=3:
                sims[0,n]=ds/(np.sqrt(ie[i]*ie[j])+1e-8)*nc/(nc+5)
                covariance=ds-xs*ys/nc
                variance=max(0.,x2-xs*xs/nc)*max(0.,y2-ys*ys/nc)
                sims[1,n]=covariance/(np.sqrt(variance)+1e-8)*nc/(nc+10)
                sims[2,n]=np.exp(-(x2+y2-2*ds)/(nc*.5))*nc/(nc+10)
                adjusted[n]=ru+(xs-ys)/(nc+5)
        out[z,:42]=_summarize_neighbors(sims,values,adjusted)
        nr=iep-is_
        sims=np.zeros((3,nr));values=np.zeros(nr);adjusted=np.zeros(nr)
        for n,p in enumerate(range(is_,iep)):
            v=iu[p];values[n]=target[v];nc=count[v]
            if nc>=3 and v!=u:
                ds=dot[v];xs=sx[v];ys=sy[v];x2=xx[v];y2=yy[v]
                sims[0,n]=ds/(np.sqrt(ue[u]*ue[v])+1e-8)*nc/(nc+5)
                covariance=ds-xs*ys/nc
                variance=max(0.,x2-xs*xs/nc)*max(0.,y2-ys*ys/nc)
                sims[1,n]=covariance/(np.sqrt(variance)+1e-8)*nc/(nc+10)
                sims[2,n]=np.exp(-(x2+y2-2*ds)/(nc*.5))*nc/(nc+10)
                adjusted[n]=target[v]+(xs-ys)/(nc+5)
            present[v]=False;target[v]=0.;dot[v]=0.;sx[v]=0.;sy[v]=0.;xx[v]=0.;yy[v]=0.;count[v]=0.
        out[z,42:]=_summarize_neighbors(sims,values,adjusted)
    return out


def extract_topk_features(state, qu, qi):
    a=state['a_rated'];r=state['res']
    ue=np.bincount(a[:,0],weights=r*r,minlength=state['nu'])
    ie=np.bincount(a[:,1],weights=r*r,minlength=state['ni'])
    return topk_residual_features(qu,qi,state['u_pos_ptr'],state['u_pos_items'],state['u_pos_res'],
                                 state['i_pos_ptr'],state['i_pos_users'],state['i_pos_res'],ue,ie)


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,default=1);p.add_argument('--anchors',type=int,default=2000)
    p.add_argument('--normalized',action='store_true');args=p.parse_args()
    task=args.task
    path=ROOT/f'experiments/v3_base_{task}.npz'
    while not path.exists():time.sleep(1)
    d=np.load(path);q=d['valid']
    filename='tasks_1_2_train.csv' if task==1 else 'tasks_3_4_train.csv'
    a,u,i=base.load(ROOT/'assignment1_2026'/filename)
    order=np.random.default_rng(258).permutation(len(a));fit=a[order[50000:]]
    fit=fit[fit[:,3]>0]
    started=time.time()
    state=fit_anchor_ridge(fit,len(u),len(i),args.anchors,normalized=args.normalized)
    print('PREPARED',task,'anchors',args.anchors,'normalized',args.normalized,'seconds',time.time()-started,flush=True)
    f=anchor_ridge_features(state,q[:,0],q[:,1])
    tag=f'anchor{args.anchors}_norm{int(args.normalized)}'
    np.savez_compressed(ROOT/f'experiments/v3_neighbors_features_{task}_{tag}.npz',features=f)
    print('FEATURES',task,'std',f.std(axis=0).tolist(),'seconds',time.time()-started,flush=True)
    residual_ridge_tune(task,f,tag)


if __name__=='__main__':main()
