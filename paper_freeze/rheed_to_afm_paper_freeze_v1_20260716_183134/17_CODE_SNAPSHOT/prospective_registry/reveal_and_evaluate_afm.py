#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
def sha(path):
    h=hashlib.sha256(); p=Path(path)
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--registry", required=True); ap.add_argument("--afm-file", required=True); ap.add_argument("--sample-id", required=True); ap.add_argument("--output-root", required=True)
    a=ap.parse_args(); out=Path(a.output_root)/"revealed_results"/a.sample_id; out.mkdir(parents=True, exist_ok=True)
    arr=np.load(a.afm_file, allow_pickle=False).astype(float); arr=arr-arr.mean(); rq=float(np.sqrt(np.mean(arr**2)))
    (out/"afm_manifest.csv").write_text("sample_id,afm_file\n%s,%s\n"%(a.sample_id,a.afm_file))
    (out/"measured_targets.json").write_text(json.dumps({"sample_id":a.sample_id,"true_rq_nm":rq}, indent=2))
    (out/"prospective_evaluation.json").write_text(json.dumps({"sample_id":a.sample_id,"true_rq_nm":rq,"note":"prediction files were not modified"}, indent=2))
    (out/"reveal_timestamp.txt").write_text(datetime.now(timezone.utc).isoformat())
    (out/"reveal_hashes.json").write_text(json.dumps({"afm_file_sha256":sha(a.afm_file)}, indent=2))
if __name__=="__main__": main()
