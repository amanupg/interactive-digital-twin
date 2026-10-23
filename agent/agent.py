"""Active-perception agent: a VLM that answers questions about the statue by
deciding where to look in the 3DGS twin.

Each step the VLM sees the question, a log of its earlier actions, and its most
recent renders, and replies with one JSON action:
    {"action": "locate", "query": "<thing>"}
    {"action": "look", "target": "<statue|located thing>", "side": "<front|back|left|right|...>",
     "height": "<low|level|high>", "distance": "<close|medium|far>"}
    {"action": "answer", "answer": "<text>"}
`locate` uses the language field (semantics/query.py); `look` renders a novel
view from the splat (agent/render.py). Every step is logged with its camera
pose and image so the trajectory can be replayed in the web demo.
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cameras import OrbitFrame, describe_azimuth, orbit_view
from query import LanguageField

SIDES = {"front": 0, "front-left": 45, "left": 90, "back-left": 135, "back": 180,
         "back-right": 225, "right": 270, "front-right": 315}
HEIGHTS = {"low": -15, "level": 0, "high": 15}
DISTANCES = {"close": 2.5, "medium": 5.0, "far": 9.0}  # multiples of the target's radius

SYSTEM = """You are inspecting a 3D digital twin of a seated bronze statue on a stone pedestal
(Columbia University's "Alma Mater"), set on library steps in a campus plaza. You can move a
virtual camera to look at it from any side. "left"/"right" are the STATUE'S own left/right.

Reply with exactly one JSON object and nothing else, choosing one action:
{"action": "locate", "query": "<short noun phrase>"}        find a part in 3D so you can look at it
{"action": "look", "target": "<statue or a located query>", "side": "<front|front-left|left|back-left|back|back-right|right|front-right>", "height": "<low|level|high>", "distance": "<close|medium|far>"}
{"action": "answer", "answer": "<short answer>"}

Look before you answer: inspect the relevant part closely, from the side where it is visible.
You have at most {max_steps} actions; answer before you run out."""


@dataclass
class Step:
    action: dict
    note: str
    camera: dict | None = None
    image_path: str | None = None


@dataclass
class Episode:
    question: str
    steps: list = field(default_factory=list)
    answer: str | None = None
    seconds: float = 0.0

    def renders(self):
        return sum(1 for s in self.steps if s.image_path)


def parse_action(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        a = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return a if isinstance(a, dict) and a.get("action") in ("locate", "look", "answer") else None


class Agent:
    def __init__(self, vlm, renderer, field_: LanguageField, frame: OrbitFrame,
                 max_steps=6, keep_images=2, width=1024, height=768, out_dir=None):
        self.vlm, self.renderer, self.field, self.frame = vlm, renderer, field_, frame
        self.max_steps, self.keep_images = max_steps, keep_images
        self.width, self.height = width, height
        self.out_dir = Path(out_dir) if out_dir else None
        statue = field_.locate("bronze statue")
        self.targets = {"statue": (statue.centre, self._radius(statue), statue.extent)}

    @staticmethod
    def _radius(hit):
        return float(max(np.linalg.norm(hit.extent[1] - hit.extent[0]) / 2, 0.15))

    def _render(self, target_name, side, height, distance):
        centre, radius, extent = self.targets[target_name]
        az, el = SIDES[side], HEIGHTS[height]
        d = max(DISTANCES[distance] * radius, 0.5)
        # Back off until the target's whole extent is in frame. The height clamp
        # often pushes the camera below a high part (the scepter top), which would
        # otherwise crop exactly the region the question is about.
        corners = np.array([[x, y, z] for x in extent[:, 0] for y in extent[:, 1] for z in extent[:, 2]])
        for _ in range(12):
            view = orbit_view(self.frame, centre, d, az, el, width=self.width, height=self.height)
            uv, z = view.project(corners)
            m = 0.08
            if (z > 0).all() and (uv[:, 0] > m * self.width).all() and (uv[:, 0] < (1 - m) * self.width).all() \
                    and (uv[:, 1] > m * self.height).all() and (uv[:, 1] < (1 - m) * self.height).all():
                break
            d *= 1.2
        return self.renderer.render(view), view

    def _save(self, ep_idx, step_idx, img):
        if not self.out_dir:
            return None
        p = self.out_dir / f"ep{ep_idx:03d}_step{step_idx}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img)
        return str(p)

    def run(self, question, ep_idx=0):
        t0 = time.time()
        ep = Episode(question)
        # Start from a default overview so the first decision is grounded in an image.
        img, view = self._render("statue", "front", "level", "medium")
        images = [img]
        ep.steps.append(Step({"action": "look", "target": "statue", "side": "front", "height": "level",
                              "distance": "medium"}, "initial overview",
                             {"R": view.R.tolist(), "t": view.t.tolist()}, self._save(ep_idx, 0, img)))
        system = SYSTEM.replace("{max_steps}", str(self.max_steps))

        for i in range(1, self.max_steps + 1):
            log = "\n".join(f"{k}. {json.dumps(s.action)} -> {s.note}" for k, s in enumerate(ep.steps))
            last = i == self.max_steps
            prompt = (f"Question: {question}\n\nYour actions so far:\n{log}\n\n"
                      f"Known targets: {', '.join(self.targets)}.\n"
                      f"The image(s) are your most recent view(s), newest last."
                      + ("\nThis is your last action: you must answer now." if last else ""))
            reply = self.vlm.chat([{"role": "system", "content": system},
                                   {"role": "user", "content": prompt, "images": images[-self.keep_images:]}],
)
            a = parse_action(reply)
            if a is None:
                ep.steps.append(Step({"action": "invalid", "raw": reply[:200]}, "could not parse; reply with JSON"))
                continue
            if a["action"] == "answer" or last:
                ep.answer = a.get("answer") if a["action"] == "answer" else self._force_answer(question, images)
                ep.steps.append(Step(a, "final answer"))
                break
            if a["action"] == "locate":
                q = str(a.get("query", "")).strip()
                hit = self.field.locate(q) if q else None
                if hit is None or not len(hit.indices) or hit.score < 0.52:
                    ep.steps.append(Step(a, f"'{q}' not found in the 3D map"))
                    continue
                self.targets[q] = (hit.centre, self._radius(hit), hit.extent)
                az, el = self.frame.angles_of(hit.centre, self.targets["statue"][0])
                h = (hit.centre - self.targets["statue"][0]) @ self.frame.up
                where = "upper part" if h > 0.3 else "lower part" if h < -0.3 else "middle"
                ep.steps.append(Step(a, f"found '{q}' (confidence {hit.score:.2f}) on the {where} of the statue, "
                                        f"towards its {describe_azimuth(az)}; now a target you can look at"))
                continue
            # look
            tgt = a.get("target", "statue")
            tgt = tgt if tgt in self.targets else "statue"
            side = a.get("side", "front") if a.get("side") in SIDES else "front"
            height = a.get("height", "level") if a.get("height") in HEIGHTS else "level"
            dist = a.get("distance", "medium") if a.get("distance") in DISTANCES else "medium"
            img, view = self._render(tgt, side, height, dist)
            images.append(img)
            ep.steps.append(Step({"action": "look", "target": tgt, "side": side, "height": height, "distance": dist},
                                 "rendered; see newest image", {"R": view.R.tolist(), "t": view.t.tolist()},
                                 self._save(ep_idx, i, img)))
        if ep.answer is None:
            ep.answer = self._force_answer(question, images)
        ep.seconds = time.time() - t0
        return ep

    def _force_answer(self, question, images):
        return self.vlm.ask(f"{question}\nAnswer briefly based on the images.", images[-self.keep_images:])
