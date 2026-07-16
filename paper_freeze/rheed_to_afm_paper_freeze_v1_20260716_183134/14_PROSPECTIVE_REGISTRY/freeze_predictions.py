#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

def sha(path):
    h=hashlib.sha256(); p=Path(path)
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prediction-root", required=True); ap.add_argument("--registry", required=True); ap.add_argument("--freeze-id", required=True)
    a=ap.parse_args(); root=Path(a.prediction_root); reg=Path(a.registry); reg.parent.mkdir(parents=True, exist_ok=True)
    rows=[]
    for pred in sorted(root.glob("*/prediction.json")):
        sample=pred.parent.name
        files={str(p.relative_to(root)): sha(p) for p in pred.parent.glob("*") if p.is_file()}
        row={"freeze_id":a.freeze_id,"sample_id":sample,"prediction_dir":str(pred.parent),"prediction_sha256":files[str(pred.relative_to(root))],"timestamp":datetime.now(timezone.utc).isoformat(),"file_hashes":files,"afm_labels_available_or_accessed":False}
        rows.append(row)
    with reg.open("a", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, sort_keys=True)+"\n")
    csv_path=reg.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["freeze_id","sample_id","prediction_dir","prediction_sha256","timestamp","afm_labels_available_or_accessed"]); w.writeheader()
        for r in rows: w.writerow({k:r[k] for k in w.fieldnames})
    run_hash=hashlib.sha256("".join(r["prediction_sha256"] for r in rows).encode()).hexdigest()
    (root/"PREDICTIONS_FROZEN_BEFORE_AFM.md").write_text(f"# Predictions Frozen Before AFM\n\nAFM labels were not available or accessed at prediction time.\n\nRun manifest hash: {run_hash}\n", encoding="utf-8")
if __name__=="__main__": main()
