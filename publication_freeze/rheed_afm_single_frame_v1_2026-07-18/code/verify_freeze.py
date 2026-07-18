#!/usr/bin/env python3
import hashlib,json,shutil
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]; IDS=[6022,6028,6029,6033,6047,6048,6056,6057,6062,6063,6070,6072,6078,6080,6081,6082,6084,6085,6090,6094,6095,6099,6101]
EXP={"N":23,"MAE":1.2600983407909774,"median_AE":1.1205674310532974,"RMSE":1.8393101333714263,"R2":0.2939669042578328,"Spearman":0.42885375494071143,"Kendall":0.28063241106719367,"pairwise_concordance":0.6403162055335968,"low_high_balanced_accuracy":0.625,"high_rq_sensitivity":0.625,"high_rq_specificity":0.7333333333333333}
def sha(p):
 h=hashlib.sha256();
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
def check(c,n,d=""): return {"name":n,"pass":bool(c),"details":d}
def manifest():
 rows=[]
 for p in sorted(R.rglob("*")):
  if p.is_file() and p.relative_to(R).as_posix()!="provenance/MANIFEST.sha256": rows.append(f"{sha(p)}  {p.relative_to(R).as_posix()}")
 (R/"provenance/MANIFEST.sha256").write_text("\n".join(rows)+"\n"); return len(rows)
def main():
 cs=[]; c=pd.read_csv(R/"data_snapshot/canonical_sample_index.csv"); cs.append(check(c.sample_id.astype(int).tolist()==IDS,"active_sample_ids_exact"))
 p=pd.read_csv(R/"results/strict_oof/predictions.csv").sort_values("sample_id"); cs.append(check(p.sample_id.astype(int).tolist()==IDS,"prediction_sample_ids_exact")); cs.append(check(not np.allclose(p.true_target_nm,p.predicted_target_nm),"true_and_predicted_rq_not_identical"))
 m=json.loads((R/"results/strict_oof/metrics.json").read_text())
 for k,v in EXP.items(): cs.append(check(abs(float(m[k])-float(v))<1e-9,f"metric_{k}_matches",{"observed":m[k],"expected":v}))
 rt=pd.read_csv(R/"results/strict_oof/retrieval_results.csv"); cs.append(check(len(rt)==23 and set(rt.method_id)=={"A3"} and set(rt.family)=={"retrieval"},"retrieval_a3_only_n23")); cs.append(check(float(rt.heldout_source_contribution.max())==0.0,"retrieval_no_heldout_source_contribution")); cs.append(check((rt.uses_heldout_true_afm_for_selection==False).all(),"retrieval_no_true_afm_selection")); cs.append(check((rt.uses_heldout_true_descriptors_for_selection==False).all(),"retrieval_no_true_descriptor_selection"))
 a=pd.read_csv(R/"results/strict_oof/retrieval_source_audit.csv"); cs.append(check((a.candidate_group_count==22).all(),"strict_candidate_group_count_22")); cs.append(check((a.strict_identity_pass==True).all(),"strict_identity_pass_all")); cs.append(check((R/"BLOCKER_PROSPECTIVE_DEPLOYMENT.md").exists(),"prospective_deployment_blocker_exists")); cs.append(check(not (R/"code/predict_unseen.py").exists(),"no_noncanonical_predict_unseen_py")); large=[p.relative_to(R).as_posix() for p in R.rglob("*") if p.is_file() and p.stat().st_size>100*1024*1024]; cs.append(check(not large,"no_file_over_100mb",large)); cs.append(check(all((R/p).exists() for p in ["figures/main/Figure1_overall_pipeline.png","figures/main/Figure2_strict_oof_rq_scatter.png","figures/main/Figure3_strict_a3_q50_atlas_all23.png","figures/main/Figure4_q10_q50_q90_amplitude_example.png"]),"main_figures_exist"))
 n=manifest(); rep={"freeze_id":"rheed_afm_single_frame_v1_2026-07-18","all_checks_passed":all(x["pass"] for x in cs),"checks":cs,"manifest_entry_count":n}; (R/"provenance/verification_report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n"); n=manifest(); rep["manifest_entry_count"]=n; (R/"provenance/verification_report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n"); manifest(); print(json.dumps(rep,indent=2,sort_keys=True)); raise SystemExit(0 if rep["all_checks_passed"] else 1)
if __name__=="__main__": main()
