"""Compare question answering with a fixed view, fixed multi-view, and the active agent.

    python agent/evaluate.py --out data/alma_v2/agent_eval

Grading is a deterministic keyword match against each question's `accept`
list (agent/qa.json), so results are reproducible and easy to audit; every
answer and every render is saved under --out for inspection.
"""

import argparse
import json
from pathlib import Path

import cv2

from agent import Agent
from cameras import OrbitFrame
from query import LanguageField
from render import SplatRenderer
from scene import load_views
from vlm import VLM

ROOT = Path(__file__).resolve().parents[1]


def correct(answer, accept):
    a = (answer or "").lower()
    return any(k in a for k in accept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/alma_v2/agent_eval")
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--only", nargs="*", help="question ids to run")
    args = ap.parse_args()

    qa = json.loads((ROOT / "agent/qa.json").read_text())
    if args.only:
        qa = [x for x in qa if x["id"] in args.only]
    views = load_views(ROOT / "data/alma_v2/undistorted/sparse")
    frame = OrbitFrame.from_views(views)
    field = LanguageField.load(ROOT / "data/alma_v2/language_field.npz")
    vlm = VLM(args.model)
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    with SplatRenderer(ROOT / "data/alma_v2/brush_30k_masked/splat_30000.ply") as renderer:
        agent = Agent(vlm, renderer, field, frame, max_steps=args.max_steps, out_dir=args.out / "renders")
        fixed = {side: agent._render("statue", side, "level", "medium")[0]
                 for side in ("front", "left", "back", "right")}
        for side, img in fixed.items():
            cv2.imwrite(str(args.out / f"fixed_{side}.jpg"), img)

        for i, item in enumerate(qa):
            row = {"id": item["id"], "q": item["q"]}
            row["front_only"] = vlm.ask(item["q"] + " Answer briefly.", [fixed["front"]])
            row["four_views"] = vlm.ask(item["q"] + " The four images show the statue from her front, "
                                        "left, back and right. Answer briefly.", list(fixed.values()))
            ep = agent.run(item["q"], ep_idx=i)
            row["agent"], row["agent_steps"], row["agent_renders"], row["agent_seconds"] = \
                ep.answer, [s.action for s in ep.steps], ep.renders(), round(ep.seconds, 1)
            for mode in ("front_only", "four_views", "agent"):
                row[f"{mode}_ok"] = correct(row[mode], item["accept"])
            results.append(row)
            print(f"[{item['id']}] front={row['front_only_ok']} four={row['four_views_ok']} "
                  f"agent={row['agent_ok']} ({row['agent_renders']} renders, {row['agent_seconds']}s)\n"
                  f"    agent: {str(row['agent'])[:120]!r}", flush=True)

    summary = {m: sum(r[f"{m}_ok"] for r in results) for m in ("front_only", "four_views", "agent")}
    summary["n"] = len(results)
    summary["agent_mean_renders"] = round(sum(r["agent_renders"] for r in results) / len(results), 2)
    (args.out / "results.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
