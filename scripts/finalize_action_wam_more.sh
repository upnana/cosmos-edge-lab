#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL="$LAB_ROOT/outputs/eval_action_wam_more"
DOCS="$LAB_ROOT/docs/assets/action_wam_more"
METRICS="$EVAL/metrics.json"
[[ -f "$METRICS" ]] || { echo "missing $METRICS"; exit 1; }

mkdir -p "$DOCS"
cp -f "$METRICS" "$DOCS/metrics.json"
cp -f "$EVAL/previews"/ep*_gt_wam_fd.mp4 "$DOCS/" 2>/dev/null || true

# compact gifs
for mp4 in "$DOCS"/ep*_gt_wam_fd.mp4; do
  [[ -f "$mp4" ]] || continue
  gif="${mp4%.mp4}.gif"
  ffmpeg -y -i "$mp4" -vf "fps=8,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse" \
    -loop 0 "$gif" </dev/null 2>/dev/null
done

# summary table via python
"$LAB_ROOT/../cosmos-framework/.venv/bin/python" - <<'PY' || python3 - <<'PY'
import json
from pathlib import Path
lab=Path('/home/july/cosmos-edge-lab')
m=json.loads((lab/'outputs/eval_action_wam_more/metrics.json').read_text())
rows=[]
for ep in m['episodes']:
    a=ep['action']['raw']['l1']
    an=ep['action']['normalized_meanstd']['l1']
    w=ep['vision']['wam']['psnr_mean_vs_gt']
    f=ep['vision']['forward_dynamics']['psnr_mean_vs_gt']
    sf=ep['action'].get('start_frame')
    rows.append((ep['episode'], sf, a, an, w, f))
rows.sort()
lines=['# Action-Policy WAM/FD — more held-out (chunk=32)\n',
       f"> job `stack3cam_action_policy_edge_2000` / `iter_000002000`  \n> episodes: {', '.join(str(r[0]) for r in rows)}  \n",
       '\n| Ep | start | Action L1 (raw°) | Action L1 (norm) | WAM PSNR | FD PSNR |\n|----|-------|------------------|------------------|----------|---------|']
for ep,sf,a,an,w,f in rows:
    lines.append(f'| {ep} | {sf} | {a:.2f} | {an:.3f} | {w:.1f} | {f:.1f} |')
    lines.append('')
    lines.append(f'### ep{ep:03d} GT | WAM | FD\n')
    lines.append(f'![ep{ep:03d}](assets/action_wam_more/ep{ep:03d}_gt_wam_fd.gif)\n')
    lines.append(f'<video src="assets/action_wam_more/ep{ep:03d}_gt_wam_fd.mp4" controls width="960" preload="metadata"></video>\n')
mean_a=sum(r[2] for r in rows)/len(rows)
mean_w=sum(r[4] for r in rows)/len(rows)
mean_f=sum(r[5] for r in rows)/len(rows)
lines.insert(4, f'\n**Mean over {len(rows)} eps:** Action L1 raw={mean_a:.2f}°, WAM PSNR={mean_w:.1f}, FD PSNR={mean_f:.1f}\n')
out=lab/'docs/ACTION_POLICY_WAM_MORE.md'
out.write_text('\n'.join(lines)+'\n')
print('wrote', out)
print('mean L1', mean_a, 'WAM', mean_w, 'FD', mean_f)
PY

# bundle
STAGING=/tmp/action_wam_more_bundle
rm -rf "$STAGING"; mkdir -p "$STAGING"
cp -a "$DOCS"/. "$STAGING/"
cp -f "$METRICS" "$STAGING/"
tar -czf "$LAB_ROOT/outputs/bundles/action_wam_more.tar.gz" -C /tmp action_wam_more_bundle
ls -lh "$LAB_ROOT/outputs/bundles/action_wam_more.tar.gz"

# append note
NOTE="$LAB_ROOT/notes/action_policy_2000_showcase.md"
if ! grep -q 'eval_action_wam_more' "$NOTE" 2>/dev/null; then
  cat >> "$NOTE" <<'N'

### 更多 held-out（chunk=32，2026-08-12）

episodes：3 / 22 / 47 / 69 / 99（max-motion 窗口）  
产物：`outputs/eval_action_wam_more/`；文档：`docs/ACTION_POLICY_WAM_MORE.md`；打包：`outputs/bundles/action_wam_more.tar.gz`
N
fi
echo DONE finalize
