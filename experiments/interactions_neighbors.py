"""Candidate-symmetric user neighborhood features using designated data only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import assignment1 as a1
import numpy as np
from numba import njit

@njit(cache=True)
def neighborhood(qu,qi1,qi2,qt,uptr,ui,ut,iptr,iu,ipos,it,norm,task):
    out=np.zeros((len(qu),26),np.float32)
    symmetric=np.zeros((len(qu),26),np.float32)
    nusers=len(uptr)-1
    stamps=np.full(nusers,-1,np.int64)
    ov=np.zeros((nusers,5),np.float32)
    lastpos=np.zeros(nusers,np.int32)
    nitems=len(iptr)-1
    for row in range(len(qu)):
        u=qu[row]; i1=qi1[row];i2=qi2[row]
        if u>=nusers: continue
        start=uptr[u];end=uptr[u+1];L=end-start
        if L==0: continue
        for pos in range(start,end):
            j=ui[pos]
            if j==i1 or j==i2: continue
            cj=iptr[j+1]-iptr[j]
            idf=np.log(1+nusers/(cj+5))
            rec=1./np.sqrt(end-pos)
            if task==4: rec=np.exp(-abs(float(ut[pos]-qt[row]))/(86400*365))
            for ptr in range(iptr[j],iptr[j+1]):
                v=iu[ptr]
                if v==u: continue
                if stamps[v]!=row:
                    stamps[v]=row;ov[v,:]=0;lastpos[v]=-1
                ov[v,0]+=1
                ov[v,1]+=idf
                ov[v,2]+=idf*rec
                ov[v,3]+=1./np.sqrt(cj+5)
                ov[v,4]+=rec
                if ipos[ptr]>lastpos[v]:lastpos[v]=ipos[ptr]
        fs=np.zeros((2,26),np.float64)
        for side in range(2):
            item=i1 if side==0 else i2
            if item>=nitems:continue
            count=iptr[item+1]-iptr[item]
            best=np.zeros(10,np.float64)
            nmatch=0
            for ptr in range(iptr[item],iptr[item+1]):
                v=iu[ptr]
                if stamps[v]!=row:continue
                nmatch+=1
                nv=uptr[v+1]-uptr[v]
                cosine=ov[v,0]/np.sqrt((L+3.)*(nv+3.))
                idfcos=ov[v,1]/np.sqrt((norm[u]+8.)*(norm[v]+8.))
                recent=ov[v,2]/np.sqrt((norm[u]+8.)*(norm[v]+8.))
                rare=ov[v,3]/np.sqrt((L+3.)*(nv+3.))
                for k,sim in enumerate((cosine,idfcos,recent,rare)):
                    fs[side,k*4]+=sim
                    fs[side,k*4+1]+=sim*sim
                    fs[side,k*4+2]+=sim*sim*sim
                    if sim>fs[side,k*4+3]:fs[side,k*4+3]=sim
                # Candidate-relative local sequence signal; time kernel for Task4.
                if task==5:
                    dist=ipos[ptr]-lastpos[v]
                    if dist>0:
                        fs[side,16]+=idfcos*idfcos/(dist+1)
                        if dist<=3:fs[side,17]+=idfcos*idfcos
                    elif dist>=-3:
                        fs[side,18]+=idfcos*idfcos
                elif task==4:
                    dt=abs(float(it[ptr]-qt[row]))/86400
                    fs[side,16]+=idfcos*idfcos*np.exp(-dt/7)
                    fs[side,17]+=idfcos*idfcos*np.exp(-dt/90)
                    fs[side,18]+=idfcos*idfcos*np.exp(-dt/365)
                else:
                    fs[side,16]+=idfcos*idfcos/(nv+5.)
                    fs[side,17]+=idfcos*idfcos*ov[v,0]
                    fs[side,18]+=idfcos*idfcos*ov[v,1]
                if idfcos>best[0]:
                    best[0]=idfcos;best.sort()
            fs[side,19]=np.sum(best)
            fs[side,20]=np.sum(best[5:])
            fs[side,21]=np.sum(best[7:])
            fs[side,22]=nmatch/(count+2.)
            fs[side,23]=np.sum(best**2)
            fs[side,24]=np.sum(best**3)
            fs[side,25]=nmatch
            # Log density lifts; max/top-neighbor evidence kept without popularity adjustment.
            for k in range(19):
                if k%4==3 and k<16:
                    fs[side,k]=np.log(fs[side,k]+.001)
                else:
                    fs[side,k]=np.log((fs[side,k]+.001)/(count+3.))
            fs[side,25]=np.log((nmatch+.1)/(count+3.))
        for k in range(26):
            out[row,k]=fs[0,k]-fs[1,k]
            symmetric[row,k]=fs[0,k]+fs[1,k]
    return out,symmetric

def build(a,nu,ni,qu,qi1,qi2,qt,task):
    uc=np.bincount(a[:,0],minlength=nu);ic=np.bincount(a[:,1],minlength=ni)
    uorder=np.argsort(a[:,0],kind='stable') if task==5 else np.lexsort((a[:,2],a[:,0]))
    uptr=np.r_[0,uc.cumsum()];ui=a[uorder,1];ut=a[uorder,2]
    position=np.empty(len(a),np.int64)
    position[uorder]=np.arange(len(a))-np.repeat(uptr[:-1],uc)
    iorder=np.argsort(a[:,1],kind='stable'); iptr=np.r_[0,ic.cumsum()]
    norm=np.bincount(a[:,0],weights=np.log(1+nu/(ic[a[:,1]]+5)),minlength=nu)
    return neighborhood(qu,qi1,qi2,qt,uptr,ui,ut,iptr,a[iorder,0],position[iorder],a[iorder,2],norm,task)

def cache(task):
    path=Path('experiments')/f'interactions_neighbor_cache_{task}.npz'
    if path.exists():return dict(np.load(path))
    fname={2:'tasks_1_2_train.csv',4:'tasks_3_4_train.csv',5:'task_5_train.csv'}[task]
    a,u,i=a1.load(Path('assignment1_2026')/fname);nu=len(u);ni=len(i)
    tr,va=a1.make_interaction_split(a,nu,ni,sequential=task==5)
    x,c=build(tr,nu,ni,va[:,0],va[:,1],va[:,2],va[:,3],task)
    tests=a1.read_test(Path('assignment1_2026')/f'task_{task}_test.csv')
    qu=np.array([u.get(r['user'],nu) for r in tests]);qi1=np.array([i.get(r['item1'],ni) for r in tests]);qi2=np.array([i.get(r['item2'],ni) for r in tests]);qt=np.array([int(r['timestamp']) if task==4 else 0 for r in tests])
    xt,ct=build(a,nu,ni,qu,qi1,qi2,qt,task)
    d=dict(x=x,c=c,xt=xt,ct=ct);np.savez_compressed(path,**d)
    print('NEIGHBOR DONE',task,flush=True)
    return d

if __name__=='__main__':
    for task in map(int,sys.argv[1:] or [2,4,5]):cache(task)
