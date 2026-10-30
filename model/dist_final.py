"""Final distribution choices + a check of whether a post-hoc calibration learned on
2024 (out of sample) improves 2025."""
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize
import dist as d
P = d.P

def nb_fit(mu, y):
    def nll(z):
        th=np.exp(z[0]+z[1]*np.log(mu)); return -np.sum(stats.nbinom.logpmf(y,th,th/(th+mu)))
    z=minimize(nll,[2.0,0.0],method="Nelder-Mead").x
    return {"type":"nb","z":z.tolist()}
def dnorm_fit(mu, y):
    def nll(z):
        sd=np.exp(z[0])*mu**z[1]; return -np.sum(np.log(stats.norm.cdf(y+0.5,mu,sd)-stats.norm.cdf(y-0.5,mu,sd)+1e-12))
    z=minimize(nll,[0.5,0.5],method="Nelder-Mead").x
    return {"type":"dnorm","z":z.tolist()}
def zig_fit(mu, y):
    pos=y>0
    def nll(z):
        a,b,k0,k1=z; pi=1/(1+np.exp(-(a+b*np.log(mu)))); k=np.exp(k0+k1*np.log(mu)); mean=mu/(1-pi)
        return -np.sum(np.where(pos,np.log(1-pi)+stats.gamma.logpdf(np.maximum(y,1e-9),k,scale=mean/k),np.log(pi)))
    z=minimize(nll,[0,-1,0.5,0],method="Nelder-Mead",options={"maxiter":6000}).x
    return {"type":"zig","z":z.tolist()}

def p_over(spec, m, L):
    z=spec["z"]; m=np.asarray(m,float); L=np.asarray(L,float)
    if spec["type"]=="nb":
        th=np.exp(z[0]+z[1]*np.log(m)); return stats.nbinom.sf(np.floor(L),th,th/(th+m))
    if spec["type"]=="dnorm":
        sd=np.exp(z[0])*m**z[1]; return stats.norm.sf(np.floor(L)+0.5,m,sd)
    a,b,k0,k1=z; pi=1/(1+np.exp(-(a+b*np.log(m)))); k=np.exp(k0+k1*np.log(m)); mean=m/(1-pi)
    return np.where(L<0,1.0,(1-pi)*stats.gamma.sf(np.maximum(L,0),k,scale=mean/k))

CHOICE={"receptions":(nb_fit,0.6),"rec_yards":(zig_fit,0.6),"carries":(dnorm_fit,1.0),"rush_yards":(zig_fit,0.6)}

def fold_preds(st, fold):
    mc,yc,rc,cut,isc=d.STATS[st]; fit,trcut=CHOICE[st]
    D=P[(P.fold==fold)&P[mc].notna()&(P[mc]>0.05)]
    tr=D[(D.split=="train")&(D[rc]>=trcut*cut)]; te=D[(D.split=="test")&(D[rc]>=cut)]
    spec=fit(tr[mc].to_numpy(),tr[yc].to_numpy().astype(float))
    L=d.lines_for(te[mc].to_numpy(),isc); hit=(te[yc].to_numpy()[:,None]>L).astype(float)
    p=np.clip(p_over(spec,te[mc].to_numpy()[:,None]*np.ones_like(L),L),1e-4,1-1e-4)
    return p.ravel(), hit.ravel(), spec

if __name__=="__main__":
    rows=[]; calrows=[]
    for st in CHOICE:
        pA,hA,_=fold_preds(st,"A"); pB,hB,spec=fold_preds(st,"B")
        # Platt calibration learned on 2024 out-of-sample, applied to 2025
        lg=lambda p: np.log(p/(1-p))
        z=minimize(lambda z: -np.mean(hA*np.log(1/(1+np.exp(-(z[0]+z[1]*lg(pA)))))+(1-hA)*np.log(1-1/(1+np.exp(-(z[0]+z[1]*lg(pA)))))),[0,1]).x
        pB2=1/(1+np.exp(-(z[0]+z[1]*lg(pB))))
        for lab,p,h in [("2024 raw",pA,hA),("2025 raw",pB,hB),("2025 + calib from 2024",pB2,hB)]:
            mid=(p>0.35)&(p<0.65)
            rows.append(dict(stat=st,set=lab,brier=np.mean((p-h)**2),logloss=-np.mean(h*np.log(p)+(1-h)*np.log(1-p)),
                             mean_p=p.mean(),hit=h.mean(),mean_p_35_65=p[mid].mean(),hit_35_65=h[mid].mean()))
        for lo in np.arange(0,1,0.1):
            for lab,p,h in [("pooled raw",np.r_[pA,pB],np.r_[hA,hB])]:
                m=(p>=lo)&(p<lo+0.1)
                if m.sum()>=30: calrows.append(dict(stat=st,bin=f"{lo:.1f}-{lo+.1:.1f}",pred=p[m].mean(),actual=h[m].mean(),n=int(m.sum())))
        print(st, "calib from 2024:", np.round(z,3), "| spec:", spec)
    pd.set_option("display.width",220)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    C=pd.DataFrame(calrows); print(C.round(3).to_string(index=False))
    C.to_csv("dist_final_calib.csv",index=False)
