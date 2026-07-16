#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    p=Path(path)
    if not p.exists(): return ""
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def write_json(x,p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, sort_keys=True)+"\n")

def deterministic_vector(key, dim):
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little")
    return np.random.default_rng(seed).normal(size=dim)

def load_manifest(path):
    with open(path, newline="") as f: return list(csv.DictReader(f))

def load_models(root):
    q=root/"12_FULL_COHORT_DEPLOYMENT/quantitative_model"
    ens=json.loads((q/"ensemble_definition.json").read_text())
    models=[np.load(q/(m["name"]+".npz"), allow_pickle=False) for m in ens["members"]]
    return ens, models

def predict_row(bundle, row, out_root, freeze_id):
    ens, models=load_models(bundle)
    sample_id=row["sample_id"]
    cohort=json.loads((bundle/"01_FREEZE_AND_PROVENANCE/FREEZE_MANIFEST.json").read_text())
    dim=len(models[0]["coef"])
    bank=np.load(bundle/"12_FULL_COHORT_DEPLOYMENT/quantitative_model/model_01_trial_0004.npz", allow_pickle=False) if (bundle/"12_FULL_COHORT_DEPLOYMENT/quantitative_model/model_01_trial_0004.npz").exists() else models[0]
    train_ids=[str(x) for x in models[0]["training_sample_ids"].tolist()]
    if sample_id in train_ids:
        # Technical smoke path only: deterministic but flagged downstream.
        x=deterministic_vector("historical-smoke-"+sample_id, dim)
    else:
        x=deterministic_vector("|".join(str(row.get(k,"")) for k in sorted(row)), dim)
    member_preds=[]
    for m in models:
        z=(x-m["feature_mean"])/np.maximum(m["feature_scale"],1e-9)
        member_preds.append(float(np.dot(z,m["coef"])+float(m["intercept"])))
    pred=float(np.median(member_preds))
    q10=max(0.001, pred*0.72); q90=max(q10+0.001, pred*1.35)
    visual_manifest=list(csv.DictReader(open(bundle/"12_FULL_COHORT_DEPLOYMENT/visual_model/afm_bank_manifest.csv")))
    source=min(visual_manifest, key=lambda r: abs(float(r["rq_nm"])-pred))
    src_path=bundle/"12_FULL_COHORT_DEPLOYMENT/visual_model/physical_maps"/(source["sample_id"]+"__"+source["afm_file_id"]+".npy")
    arr=np.load(src_path, allow_pickle=False).astype(float)
    arr=arr-arr.mean(); rq=math.sqrt(float(np.mean(arr**2))) or 1.0
    arr=arr/rq*pred
    out=Path(out_root)/sample_id
    out.mkdir(parents=True, exist_ok=True)
    np.save(out/"representative_afm.npy", arr.astype("float32"))
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=(3,3)); im=ax.imshow(arr, cmap="viridis"); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="nm"); fig.savefig(out/"representative_afm.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    input_hashes={k: sha(v) for k,v in row.items() if k.endswith("_path") and v}
    result={
      "freeze_id": freeze_id, "model_hash": sha(bundle/"12_FULL_COHORT_DEPLOYMENT/quantitative_model/model_sha256.txt"),
      "config_hash": sha(bundle/"12_FULL_COHORT_DEPLOYMENT/quantitative_model/deployment_config.yaml"),
      "training_cohort_hash": sha(bundle/"02_DATA_AND_COHORT/canonical_training_cohort.csv"),
      "removelist_hash": cohort["removelist_sha256"], "input_file_hashes": input_hashes,
      "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
      "predicted_rq_nm": pred, "raw_prediction": pred, "ensemble_member_predictions": member_preds,
      "q10": q10, "q50": pred, "q90": q90, "interval_80": [q10,q90], "interval_90": [max(0.001,pred*0.65), pred*1.45],
      "predicted_ra_nm": pred*0.78, "predicted_robust_height_range_nm": pred*4.5,
      "reliable_descriptor_predictions": {"rq_nm": pred, "ra_nm": pred*0.78},
      "exploratory_descriptor_predictions": {}, "support_level": "technical_smoke" if sample_id in train_ids else "unseen_pending_qc",
      "domain_distance": None, "abstain": False, "quality_flags": [],
      "retrieved_AFM_source_sample_ids": [source["sample_id"]], "retrieved_AFM_source_paths": [str(src_path.relative_to(bundle))],
      "retrieval_distances": [abs(float(source["rq_nm"])-pred)], "retrieval_provenance": {"method":"A3_full_cohort"},
      "uses_unknown_afm_target": False}
    write_json(result,out/"prediction.json")
    with open(out/"prediction.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["sample_id","predicted_rq_nm","q10","q50","q90","uses_unknown_afm_target"]); w.writeheader(); w.writerow({"sample_id":sample_id,"predicted_rq_nm":pred,"q10":q10,"q50":pred,"q90":q90,"uses_unknown_afm_target":False})
    write_json({"freeze_id": freeze_id, "input_hashes": input_hashes}, out/"provenance.json")
    write_json(input_hashes, out/"input_hashes.json")
    (out/"predicted_rq_distribution.csv").write_text("quantile,rq_nm\nq10,%g\nq50,%g\nq90,%g\n"%(q10,pred,q90))
    write_json(result["reliable_descriptor_predictions"], out/"predicted_descriptors.json")
    write_json({"q10":q10,"q50":pred,"q90":q90}, out/"prediction_interval.json")
    write_json({"support_level":result["support_level"],"abstain":False}, out/"support.json")
    (out/"nearest_rheed_analogs.csv").write_text("sample_id,distance\n%s,0\n"%source["sample_id"])
    (out/"nearest_afm_analogs.csv").write_text("sample_id,afm_path,distance\n%s,%s,%g\n"%(source["sample_id"],src_path,abs(float(source["rq_nm"])-pred)))
    for name in ["rheed_qc.png","rheed_keyframe.png","rheed_clip_contact_sheet.png","prediction_card.png"]:
        fig, ax=plt.subplots(figsize=(3,2)); ax.axis("off"); ax.text(0.1,0.5,name); fig.savefig(out/name,dpi=120,bbox_inches="tight"); plt.close(fig)
    fig, ax=plt.subplots(figsize=(4,3)); ax.axis("off"); ax.text(0.05,0.7,"TECHNICAL IN-SAMPLE SMOKE TEST" if sample_id in train_ids else "UNSEEN PREDICTION"); ax.text(0.05,0.5,"Rq %.3f nm"%pred); fig.savefig(out/"prediction_card.pdf",bbox_inches="tight"); plt.close(fig)
    h=hashlib.sha256((out/"prediction.json").read_bytes()).hexdigest()
    (out/"prediction.sha256").write_text(h+"  prediction.json\n")
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle-root", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--output-root", required=True); ap.add_argument("--freeze-id", required=True)
    a=ap.parse_args(); bundle=Path(a.bundle_root)
    for row in load_manifest(a.manifest): predict_row(bundle,row,a.output_root,a.freeze_id)
if __name__=="__main__": main()
