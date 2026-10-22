#!/bin/bash
# M9c replan recipe of this dir: the shipped pr9 graph re-planned on the new Shadow (CPU, ~100 s), then the shipped
# runtime= / configs= lines restored (make_plan.sh writes its own short runtime line). Result == shipped (0 diffs).
set -euo pipefail
cd /data02/users/mithrillo/projects/perseus-core-wt/m9c
source tools/agent.sh
A=python/perseus_core/assets/bootstrap_placements
CUDA_VISIBLE_DEVICES= PYTHONPATH=python:python/tests .venv/bin/python -c "
from pathlib import Path
import tempfile, test_paper_plans as T
out, wall = T.replan('llama31_8b_prefill_python_n32_pr9', Path(tempfile.mkdtemp()) / 'replan.log')
res = T.compare('llama31_8b_prefill_python_n32_pr9', out, Path(tempfile.mkdtemp()))
bad = {k: v for k, v in res.items() if isinstance(v, str) or any(v.values())}
assert not bad, bad
print(out, wall)"
N=llama31_8b_prefill_python_n32_pr9_m9c
rm -rf $A/$N.new && mv $A/_regen_llama31_8b_prefill_python_n32_pr9 $A/$N.new
cp $A/llama31_8b_prefill_python_n32_pr9/pcmm_calib.json $A/$N.new/
{ grep -v "^runtime=" $A/$N.new/PLAN_CMD.txt; grep "^runtime=\|^configs=" $A/llama31_8b_prefill_python_n32_pr9/PLAN_CMD.txt
  echo "replanned=M9c on the new Shadow"; } > $A/$N.new/PLAN_CMD.tmp && mv $A/$N.new/PLAN_CMD.tmp $A/$N.new/PLAN_CMD.txt
