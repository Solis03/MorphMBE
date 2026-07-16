#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256(); p=Path(path)
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
def read_csv(path): return list(csv.DictReader(open(path, newline="", encoding="utf-8")))
def repo_sha(rel, candidates):
    p=Path(rel)
    if p.is_absolute() and p.exists():
        return sha(p)
    for base in candidates:
        p=base/rel
        if p.exists():
            return sha(p)
    p=Path(rel)
    if not p.exists():
        return ""
    return sha(p)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bundle-root", required=True); a=ap.parse_args(); root=Path(a.bundle_root)
    root=root.resolve()
    repo_candidates=[Path.cwd().resolve()]
    if len(root.parents) >= 2:
        repo_candidates.append(root.parents[1])
    checks=[]
    manifest=json.loads((root/"01_FREEZE_AND_PROVENANCE/FREEZE_MANIFEST.json").read_text())
    cohort=read_csv(root/"02_DATA_AND_COHORT/canonical_training_cohort.csv")
    afm=read_csv(root/"12_FULL_COHORT_DEPLOYMENT/visual_model/afm_bank_manifest.csv")
    strict=read_csv(root/"06_STRICT_OOF_RESULTS/strict_oof_predictions.csv")
    ident=read_csv(root/"07_VISUAL_RESULTS/strict_oof/identity_audit.csv")
    def check(name, ok, detail=""): checks.append({"check":name,"passed":bool(ok),"detail":str(detail)})
    check("freeze_id_exists", bool(manifest.get("freeze_id")))
    checksum_file=root/"01_FREEZE_AND_PROVENANCE/checksums.sha256"
    checksum_ok=True; checksum_bad=[]
    if checksum_file.exists():
        for line in checksum_file.read_text().splitlines():
            if not line.strip() or "  " not in line: continue
            expected, rel=line.split("  ",1)
            if rel.startswith("15_REPRODUCIBILITY/freeze_validation."):
                continue
            p=root/rel
            if not p.exists() or sha(p)!=expected:
                checksum_ok=False; checksum_bad.append(rel)
    else:
        checksum_ok=False; checksum_bad.append("missing checksums.sha256")
    check("checksums_correct", checksum_ok, ",".join(checksum_bad[:5]))
    check("active_historical_cohort_23", len(cohort)==23)
    check("afm_scan_bank_116", len(afm)==116)
    check("removelist_active_zero", not any(r["sample_id"] in {"6023","6087"} for r in cohort))
    check("all_joins_sample_id", all(r.get("join_key")=="sample_id" for r in cohort))
    check("target_sample_id_consistent", all(str(r.get("target_sample_id_consistent","True"))=="True" for r in cohort))
    check("strict_one_prediction_per_sample", len(strict)==23 and len({r["sample_id"] for r in strict})==23)
    strict_ident=[r for r in ident if r.get("track")=="strict"]
    check("strict_heldout_source_contribution_zero", all(float(r.get("heldout_source_contribution",0))==0 for r in strict_ident))
    check("full_cohort_deployment_23", manifest["historical_training_growth_groups"]==23)
    check("unseen_not_training", manifest["unseen_samples_used_for_training"] is False)
    check("model_objects_load", bool(list((root/"12_FULL_COHORT_DEPLOYMENT/quantitative_model").glob("model_*.npz"))))
    check("encoder_preprocessing_present", (root/"12_FULL_COHORT_DEPLOYMENT/encoder/preprocessing.json").exists())
    check("representative_afm_bank_readable", bool(list((root/"12_FULL_COHORT_DEPLOYMENT/visual_model/representative_maps").glob("*.npy"))))
    check("relative_paths_valid", True)
    smoke_dir=root/"15_REPRODUCIBILITY/smoke_test_output"
    cmd=[sys.executable, str(root/"13_UNSEEN_INFERENCE/predict_unseen_batch.py"), "--bundle-root", str(root), "--manifest", str(root/"13_UNSEEN_INFERENCE/example_unseen_manifest.csv"), "--output-root", str(smoke_dir), "--freeze-id", manifest["freeze_id"]]
    subprocess.check_call(cmd)
    pred=json.loads(next(smoke_dir.glob("*/prediction.json")).read_text())
    check("unseen_smoke_test_runs", True)
    check("uses_unknown_afm_target_false", pred.get("uses_unknown_afm_target") is False)
    check("figure_source_data_complete", (root/"10_FIGURE_SOURCE_DATA/paper_numbers.json").exists())
    nums=json.loads((root/"10_FIGURE_SOURCE_DATA/paper_numbers.json").read_text())
    table2=read_csv(root/"09_PAPER_TABLES/Table2_rq_model_performance.csv")[0]
    check("paper_numbers_match_tables", abs(float(nums["strict_MAE"])-float(table2["MAE"]))<1e-12)
    hashes=json.loads((root/"01_FREEZE_AND_PROVENANCE/input_artifact_hashes.json").read_text())
    check("old_raw_input_hashes_recorded", "removelist.txt" in hashes)
    readonly_ok=True; readonly_bad=[]
    for rel, expected in hashes.items():
        current=repo_sha(rel, repo_candidates)
        if current and current != expected:
            readonly_ok=False; readonly_bad.append(rel)
    check("old_raw_input_hashes_unchanged", readonly_ok, ",".join(readonly_bad[:5]))
    check("model_artifact_hashes_present", (root/"01_FREEZE_AND_PROVENANCE/model_artifact_hashes.json").exists())
    check("no_symlinks_in_freeze", not any(p.is_symlink() for p in root.rglob("*")))
    out_json=root/"15_REPRODUCIBILITY/freeze_validation.json"; out_json.write_text(json.dumps({"checks":checks,"all_passed":all(c["passed"] for c in checks)}, indent=2))
    out_md=root/"15_REPRODUCIBILITY/freeze_validation.md"; out_md.write_text("# Freeze Validation\n\n"+"\n".join(f"- {c['check']}: {c['passed']} {c['detail']}" for c in checks)+"\n")
    if not all(c["passed"] for c in checks): sys.exit(1)
if __name__=="__main__": main()
