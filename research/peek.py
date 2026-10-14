import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "yeno-bot"))
from model import FairValue
from recorded import load_rows, build
src = sys.argv[1]; which = int(sys.argv[2])
ev, mk = build(load_rows(), ref_src=src)
slugs = sorted(mk); slug = slugs[which]; m = mk[slug]
fv = FairValue(); last = -1
for e in ev:
    if e[0] == "btc": fv.observe_mid(e[1], e[2]); continue
    if e[0] != "book" or e[3] != "YES" or e[2] != slug: continue
    fv.learn_basis(m["start"], m["open"])
    t = e[1]; S = m["end"] - t
    if S > 150 or int(S/3) == last: continue
    last = int(S/3)
    p, mean, sd = fv.prob_yes(t, m["end"], m["open"])
    b = e[4][0] if e[4] else None; a = e[5][0] if e[5] else None
    print(f"S={S:6.1f} model={p:.3f} gap={mean-m['open']:+7.1f} sd={sd:5.1f} yes bid={b} ask={a}")
print("open", m["open"], "close", m["close"], "basis", round(fv.basis,2))
