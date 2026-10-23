"""Thin client for a local VLM served by Ollama (persistent, unlike v1's per-query reload)."""

import base64
import json
import urllib.request

import cv2

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen3-vl:8b"


def encode_image(img_bgr, max_side=1024, quality=90):
    h, w = img_bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


class VLM:
    def __init__(self, model=DEFAULT_MODEL, temperature=0.0, timeout=300, max_tokens=2048):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens  # reasoning included: caps runaway thinking (~400 is typical)

    def chat(self, messages):
        """messages: [{"role", "content", "images": [BGR arrays]}] -> reply text.

        No Ollama `format` constraint: qwen3-vl reasons before replying even with
        think=False, and with format="json" or a schema the reply comes back empty.
        Callers extract the JSON object from plain text instead (agent.parse_action).
        """
        payload = {
            "model": self.model,
            "messages": [{"role": m["role"], "content": m["content"],
                          **({"images": [encode_image(i) for i in m["images"]]} if m.get("images") else {})}
                         for m in messages],
            "stream": False,
            "think": False,
            "options": {"temperature": self.temperature, "num_ctx": 16384, "num_predict": self.max_tokens},
        }
        req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())["message"]["content"]
        except TimeoutError:
            return ""  # treated by callers as an unusable reply

    def ask(self, question, images, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": question, "images": images}]
        return self.chat(msgs)
