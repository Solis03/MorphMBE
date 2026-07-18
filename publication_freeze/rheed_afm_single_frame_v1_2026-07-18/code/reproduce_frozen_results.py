#!/usr/bin/env python3
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr,kendalltau
r=Path(__file__).resolve().parents[1]; p=pd.read_csv(r/"results/strict_oof/predictions.csv"); y=p.true_target_nm.to_numpy(float); q=p.predicted_target_nm.to_numpy(float)
def pc(y,q):
 n=ok=0
 for i in range(len(y)):
  for j in range(i+1,len(y)):
   if y[i]!=y[j] and q[i]!=q[j]: n+=1; ok+=int((y[i]>y[j])==(q[i]>q[j]))
 return ok/n
out={"N":len(y),"MAE":float(np.mean(abs(q-y))),"median_AE":float(np.median(abs(q-y))),"RMSE":float(np.sqrt(np.mean((q-y)**2))),"R2":float(1-np.sum((q-y)**2)/np.sum((y-np.mean(y))**2)),"Spearman":float(spearmanr(y,q).statistic),"Kendall":float(kendalltau(y,q).statistic),"pairwise_concordance":float(pc(y,q))}
print(json.dumps(out,indent=2,sort_keys=True))
