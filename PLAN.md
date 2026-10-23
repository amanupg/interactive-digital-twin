# Alma Mater Twin v2: plan

**Goal:** turn the 2025 course project into a *spatial agent over a semantic digital twin*. That means a
metric 3D Gaussian Splatting reconstruction of Columbia's Alma Mater statue that you can query in natural
language ("where is the owl?"). A vision-language agent explores it by choosing its own viewpoints, and
everything runs in the browser.

The v2 version takes on four problems the industry is currently investing in:

| Capability | Where it shows up in industry |
|---|---|
| High-fidelity, metric 3DGS from casual phone video | Digital twins for robotics simulation, AR, heritage and real estate |
| Language-grounded 3D (per-Gaussian semantic features) | Open-vocabulary 3D perception, robot manipulation targets |
| VLM agent that plans camera moves (active perception) | Embodied AI, inspection drones, spatial reasoning benchmarks |
| Compressed splats streamed to a web viewer | Product delivery of 3D: e-commerce, maps, sharing |

---

## Data

Captured 2025-11-12 on an iPhone 13 Pro Max (4K HEVC, 30 fps, overcast). Each video is a full orbit at a different height.

| File | Duration | Frames | Start (UTC) | Ring |
|---|---|---|---|---|
| `IMG_5816.MOV` | 4:03 | 7,299 | 12:48 | r0, middle orbit |
| `IMG_5818.MOV` | 5:14 | 9,425 | 12:56 | r1, outer orbit |
| `IMG_5820.MOV` | 3:56 | 7,086 | 13:02 | r2, inner orbit |

The raw videos live outside the repo (`~/Desktop/DL4CV_Data`) and are gitignored. Derived data goes in `data/` and is also gitignored.

## Baseline: v1 (Dec 2025)

The v1 system is the code currently at the repo root: `camera_control.py`, `brain_service.py`, and splatfacto trained with nerfstudio.

Known weaknesses, based on reviewing `final_hd_v3.mp4` and the code:
- Floaters and "glass shard" artifacts on the steps and plaza, smeared ground texture, and noisy sky.
- Pedestrians walking through the frame are baked into the scene.
- The agent maps keywords to one of 4 *training camera indices*. It cannot render novel views, so the "navigation" is a lookup table.
- The VLM is reloaded from disk for every question, through a subprocess with a hardcoded `/home/au2327` path.
- There are no quantitative metrics.

All v2 results are reported against this baseline.

---

## Phase 1: Reconstruction (in progress)

**Output:** a clean, metric, gravity-aligned splat of the statue and plaza, with a benchmark table.

1. **Frame selection** (`recon/extract_frames.py`). Decode each video with VideoToolbox. For every
   time window, keep the sharpest frame (variance of the Laplacian), which drops motion-blurred frames.
   Target about 250–350 frames per ring, downscaled to 1920 px on the long side.
2. **Pose estimation** (`recon/run_sfm.sh`). Run COLMAP on the three rings together: SIFT features,
   sequential matching with loop detection, plus exhaustive matching between rings so the heights get
   registered to each other. Then run a global mapper (GLOMAP-style) if available, otherwise incremental.
   A feed-forward pose prior (VGGT / MASt3R-family) is a fallback if COLMAP fails to bridge the rings.
3. **Transient masking.** Segment people and other moving objects (a detector plus SAM masks) and remove
   them from the photometric loss.
4. **Scene alignment.** Make the z-axis gravity-up using the plaza plane, and set metric scale from a measured
   reference, either the pedestal or the step riser height (the iPhone GPS altitude is too noisy).
5. **Training** (gsplat on a GCP L4). Use MCMC densification for a bounded Gaussian budget, anti-aliased
   rasterization, per-image appearance embeddings for exposure changes over 15 min of capture, and
   rolling-shutter-aware or 3DGUT-style camera handling if artifacts persist. Use Brush or another
   Metal/WebGPU trainer on the M5 for quick local previews.
6. **Evaluation.**
   - *Interpolation:* hold out every 8th frame and report PSNR, SSIM and LPIPS.
   - *Cross-ring extrapolation:* train on two rings and test on the third. This is a harder test of novel views and gives a strong headline result.
     SfM shows the rings differ mostly in orbit radius rather than height, so the default split is train r0 + r2, test the outer ring r1.
   - Report Gaussian count, file size and render FPS, and compare with v1 under the same protocol.

**Compute:** everything runs locally on the M5 Mac (32 GB) for now. GCP is on hold because of credit limits.
Training uses [Brush](https://github.com/ArthurBrussee/brush) v0.3.0 (WebGPU/Metal), installed at
`~/.local/share/brush-v0.3.0/`. gsplat-specific features such as MCMC densification and appearance
embeddings wait until there's CUDA access. If they turn out to matter, the plan will look at Metal-capable alternatives.

## Phase 2: Language field

**Output:** open-vocabulary 3D queries such as "owl" or "scepter" that return a highlighted Gaussian set and a 3D centroid or box.

- Extract multi-scale SAM masks per training view and encode them with CLIP/SigLIP. Distill the features
  into compact per-Gaussian embeddings, following the LangSplat / feature-splatting family, with an autoencoder to keep memory small.
- Query by text, take the relevancy map, then 3D clustering to get a part.

**As built (no CUDA, so the lifting is training-free rather than a distilled feature rasterizer):**
- `semantics/extract_features.py` runs SAM 2.1-B "everything" mode twice per view, once on the whole frame and once on a
  **statue zoom crop**. The crop is the bronze figure and throne projected from the splat (radius-1.1 cylinder above the pedestal).
  - Each mask gets an OpenCLIP ViT-B/16 (LAION-2B) embedding of its masked crop.
  - Each pixel then gets two labels: the smallest covering mask (*part*) and the largest covering mask under 25% of the frame (*object*).
  - The zoom pass is what separates the crown, book and scepter, which are too small at frame scale.
- `semantics/lift.py` projects about 1.16M Gaussians (opacity > 0.1, within 25 units) into each of 300 views.
  - A coarse z-buffer over the opaque Gaussians decides visibility.
  - All 33k mask embeddings are quantized into a 12k-entry k-means codebook.
  - Each visible Gaussian casts a vote for the code under it at both levels, weighted by 1/√(mask area).
  - Each Gaussian keeps its **top 6 codes** (multi-label) instead of one averaged embedding.
  - The field file is 35 MB.
- `semantics/query.py` exposes `LanguageField.locate(text)`, which the Phase 3 agent will call.
  - A Gaussian's score is its best code with at least 3 supporting observations, using LERF relevancy against generic negatives.
  - It takes the top 0.2% of Gaussians and links them into connected components (radius 0.25).
  - It returns the component with the **highest peak relevancy** (mean of its top 20 scores, at least 20 members),
    as a centroid, an extent and the Gaussian ids.

**What it took to make small parts work** (each step was diagnosed on the actual crown and book Gaussians):

| Change | Why |
|---|---|
| Statue zoom pass in 2D | Frame-scale SAM never isolates the crown, book or scepter. |
| Multi-label codes instead of averaging | Crown Gaussians are visible in a median of 180 views but isolated as "crown" in only a few. Averaging dropped them from 0.65 (2D) to 0.52. |
| 12k codebook instead of 4k | The book is only weakly matched (0.55 in 2D), and 4k k-means merged it into a generic cluster scoring 0.51. An exact codebook breaks support counting, because each mask exists in only one view. |
| Rank components by peak, with a minimum size | Total-mass ranking picked a big lamp post over the small scepter. Without a size floor, single noisy Gaussians won. |

**Results** (`data/alma_v2/queries_full_{a,b}.jpg`, checked in views from all three rings, including from behind):

| Query | Result |
|---|---|
| laurel crown | ✅ centroid on the crown from front, side and back |
| scepter | ✅ centroid on the scepter head from all rings |
| open book | ✅ centroid on the book in her lap |
| stone inscription | ✅ the "ALMA MATER" granite block (SAM treats the whole block as one mask) |
| bronze statue | ✅ figure and throne |
| column | ✅ centroid on a Low Library column, with heat bleeding onto the steps and pedestal |
| lamp | ❌ misses the green lamps flanking the statue |
| owl | ❌ lands on her face; the hidden owl in the robe folds is never isolated by SAM |

**Open issues:**
- **No quantitative benchmark yet.** Results are checked visually. The next step is ground-truth 3D positions for each part:
  click a part in 2 views, triangulate it, then report localization error and hit rate.
- **Heat bleed:** the max-over-codes rule trades precision for recall, as the "column" query shows.
- **Owl:** needs a targeted approach, for example a much tighter zoom on the robe or prompted SAM points.
- **Cost on the M5:** 2D extraction takes about 21 s per view (1h45m for 300 views). Lifting takes about 3 min.
- **Evaluation:** hand-label a small part set (owl hidden in the robe folds, scepter, open book, laurel crown,
  throne-arm lamps, "ALMA MATER" inscription) in about 20 held-out views and report mIoU and localization accuracy.

## Phase 3: VLM agent

**Output:** an agent that answers questions about the statue by *deciding where to look*.

- Keep a persistent VLM server, either a current Qwen-VL generation through vLLM or an API model, instead of reloading per query.
- The agent gets these tools:
  - `locate(text)` returns a 3D position from the language field.
  - `look_at(target, distance, elevation, azimuth)` renders a **novel view** from the splat.
  - `orbit(target, n)` renders multiple views around a target.
  - `crop/zoom` takes a high-resolution render of a region.
- The loop is plan, then render, then inspect, then either answer or move again. The camera trajectory is logged for the demo.
- **Evaluation:** a small QA set of about 30 questions with ground truth (e.g. "what animal is hidden on the statue?",
  "what is she holding in her right hand?", "what is written on the pedestal?"). Compare three setups:
  front view only, fixed multi-view, and the active agent. Report accuracy and the number of renders.

**As built (all local on the M5):**
- **Renderer** (`web/render.html`, `agent/render.py`): Spark 2.3.1 (three.js) runs in headless Chromium through Playwright.
  - The splat stays in COLMAP coordinates. Cameras are set from OpenCV extrinsics, and the projection matrix comes
    straight from the intrinsics.
  - The 3M splats load in 1.2 s, and a frame takes about 0.2 s.
  - Each render waits for Spark's worker sort to finish for *this* camera. A fixed frame count left stale sort order
    after large camera jumps, which dropped one view to 14 PSNR.
  - **Known gap:** on held-out views, Spark scores 24.4 PSNR and Brush scores 28.8 on the same PLY (they agree with each
    other at 26.1). The error is concentrated in fine foliage, with slightly darker highlights. Spark's encoding and
    kernel settings don't change this, so it's probably an anti-aliasing or opacity-model difference from what Brush trained with.
    The statue and inscription render legibly, which is what the agent needs.
- **Cameras** (`agent/cameras.py`): a gravity-aligned orbit frame.
  - Azimuth 0 is her front, 90 her left side and 270 her right side, verified on renders.
  - Camera height is **clamped to the capture band**, because the eye-level capture leaves views from above full of floaters.
  - The agent backs the camera off until the target's 3D extent fits in frame. Without this, the clamp cropped the scepter's crown.
- **VLM** (`agent/vlm.py`): Qwen3-VL 8B (Q4_K_M) served persistently by Ollama.
  - It's a reasoning model, so Ollama's JSON/schema modes return empty replies. Actions are parsed from plain text instead.
  - Output is capped at 2048 tokens, because one owl question reasoned until the 10-minute timeout.
- **Agent** (`agent/agent.py`): it starts from a frontal overview, then takes up to 6 JSON actions:
  `locate(query)` against the Phase 2 field, `look(target, side, height, distance)` to render a novel view, and `answer`.
  Every step's camera pose and render is saved for replay in the web demo.
- **Eval** (`agent/evaluate.py`, `agent/qa.json`): 15 questions graded by keyword match. It compares front-only (1 render),
  four fixed views (4 renders at once) and the active agent.

**First results** (`data/alma_v2/agent_eval/results.json`, Qwen3-VL 8B, temperature 0):

| Setup | Correct | Renders per question |
|---|---|---|
| Front view only | 11 / 15 | 1 |
| Four fixed views (front, left, back, right) | 12 / 15 | 4 |
| **Active agent** | **12 / 15** | **1.27** (median 44 s per question) |

- **Where the agent wins:** "what sits at the top of the scepter?" The agent located the scepter, chose a close,
  high, front-right view and answered **crown**. Both fixed setups said "sphere", because the crown is too small in an overview.
- **Where it loses:** the plaza question ("standing behind the statue…"). The agent answered from the frontal overview
  without looking behind her. The four-view baseline got it because its back view happened to be included.
- **All three setups fail** on the hidden owl (answers: mouse or dove) and on the green lamp (answers: urn).
- **Honest read:** on this small set the agent ties four views while rendering about 3× less. Its main weakness is
  **under-exploration**: the 8B model often answers from the overview even when the question names a viewpoint.
  - Next steps: a policy rule or few-shot examples that force a look when the question names a side; a stronger planner VLM; more questions for statistical power.
  - 15 questions is too few to separate the setups with confidence (a 1-question difference is noise).

## Phase 4: Web demo

**Output:** a public link where anyone can orbit the statue and chat with the agent while watching its camera move.

- Compress and export the splat (SPZ or similar) and serve it in a WebGPU/three.js splat viewer.
- Add a chat panel that calls the agent backend. When the agent moves, the viewer camera moves with it, and the language-field highlights are overlaid.
- Include the benchmark tables and a short write-up. That page is the artifact for the portfolio and resume.

## Stretch

- Mesh extraction (2DGS/PGSR-style) and a USD export to drop the statue into a robotics simulator.
- Relighting or day/night appearance editing.
- A second campus landmark captured with the same pipeline, to show it generalizes.

---

## Repo layout (target)

```
PLAN.md
recon/        # phase 1: frame extraction, SfM, masking, alignment, training configs, eval
semantics/    # phase 2: language field
agent/        # phase 3: VLM tools + loop + QA eval
web/          # phase 4: viewer + chat
legacy/       # v1 code (moved once v2 replaces it)
data/         # gitignored: frames, colmap, checkpoints
```

## Status log

- 2026-10-20: plan written. Repo cloned and the raw videos reviewed. Phase 1 started on the M5.
  - The videos are **HLG HDR** (10-bit BT.2020). v1 decoded them naively, which gives washed-out color.
    v2 tone maps to SDR BT.709 through VideoToolbox `scale_vt`.
  - `recon/extract_frames.py` selected 300 sharpest-in-window frames per ring (900 total at 1920×1080) in about 1 min.
    The sharpness floor is about 1.4k Laplacian variance (median about 2.7k).
  - COLMAP 4.2.1 (CPU, Homebrew):
    - SIFT extraction took 10.6 min, about 11.7k features per frame.
    - Ring-aware pairs from `recon/make_pairs.py` give 39.2k pairs (7.2k within-ring including loop closure, 32k cross-ring by orbit phase).
    - Global mapper: **1 model, 899/900 images registered, 224k points, mean track length 21.8,
      mean reprojection error 0.83 px.** 39.1k of 39.2k pairs verified, with about 1.6k inliers per pair on average.
    - Shared OPENCV camera: f ≈ 1570 px at 1920×1080 (about 63° HFOV, since EIS crops the frame).
    - Undistorted PINHOLE images are in `data/alma_v2/undistorted/`, ready for gsplat.
  - Camera layout (`recon/plot_cameras.py`, `data/alma_v2/cameras.png`):
    - All three orbits close cleanly.
    - Gravity estimated from the camera up vectors is very consistent (|mean| = 0.994).
    - The rings separate mainly by **distance**:

      | Ring | Video | Orbit radius (SfM units) |
      |---|---|---|
      | r2 | `IMG_5820` | 3.2 |
      | r0 | `IMG_5816` | 3.6 |
      | r1 | `IMG_5818` | 4.7 |

    - Mean camera height differs by only about 0.1 units between rings, which is smaller than the ±0.3 variation from walking the steps.
      Pitch is +4–6° for all three rings.
    - Because of this, the "train on two rings, test on the third" evaluation is better framed as **cross-distance**:
      train on r0 + r2 and test on the outer ring r1.
  - **First local training preview** (Brush, 5k steps, max 1280 px, every 8th frame held out for 113 eval views)
    took about 6 min on the M5 and produced 730k splats in a 172 MB PLY.
    Scored with `recon/eval_renders.py`:

    | Step | PSNR | SSIM | r0 | r1 (outer) | r2 (inner) |
    |---|---|---|---|---|---|
    | 2,500 | 24.65 | 0.816 | 25.28 | 24.00 | 24.65 |
    | 5,000 | 25.74 | 0.855 | 25.97 | 25.45 | 25.80 |

    - The statue and the "ALMA MATER" inscription are already sharp, and the steps don't show v1's glass-shard artifacts.
    - The worst views (PSNR 21–23) are mostly views with **pedestrians in the ground truth**, which the model correctly leaves out.
      Transient masking matters for evaluation as well as training.
    - Distant buildings are blurry. That's expected, because they're only seen from the plaza.
  - **Transient masks** (`recon/make_masks.py`): YOLO11l-seg on MPS takes about 2 min for 899 frames.
    - The statue is a human figure, so person detections on it have to be kept. A detection counts as static only if it:
      - falls mostly inside the statue region (SfM points within a radius-1.3 cylinder around the orbit axis, projected into each view),
      - covers at least 1% of the frame, and
      - has triangulated-keypoint support.
    - Keypoint density alone wasn't enough, because shadowed bronze in distant frontal views looks "untracked".
    - Results:
      - 747 of 748 large person detections (the statue) are kept.
      - 2,982 detections are masked.
      - 604 of 899 frames have some masked pixels, averaging 0.8% of the frame (max 4.4%).
    - One known false positive: the open book on the statue's lap sometimes gets masked as a bag.
  - **Full run** (`data/alma_v2/brush_30k_masked/`): Brush, 30k steps, native 1892×1063, masks on, 3M splat cap
    (reached before step 20k). It took **1 h 51 min** on the M5 and produced a 708 MB PLY.
    All scores use masked metrics on the same 113 held-out views:

    | Run | Step | PSNR @1280 | SSIM @1280 | PSNR native | SSIM native | r0 / r1 / r2 PSNR @1280 |
    |---|---|---|---|---|---|---|
    | preview, no masks, trained at 1280 | 5k | 26.32 | 0.858 | — | — | 26.11 / 25.83 / 27.00 |
    | masked, native | 10k | 27.13 | 0.882 | 25.62 | 0.828 | 26.75 / 26.57 / 28.05 |
    | masked, native | 20k | 28.17 | 0.908 | 26.56 | 0.863 | 27.87 / 27.81 / 28.83 |
    | masked, native | **30k** | **28.65** | **0.917** | **26.99** | **0.875** | 28.23 / 28.41 / 29.31 |

    - The worst views (about 23 PSNR) look nearly identical to the ground truth. The remaining error is in wind-blown
      foliage and the flag, which are transient too. The "ALMA MATER" inscription and the statue's drapery are crisp.
    - **Caveats:**
      - The masking gain isn't isolated yet, because the full run also differs from the preview in step count and resolution.
        An unmasked 30k run is needed as an ablation.
      - The v1 baseline hasn't been scored under this protocol yet, because its checkpoint was on the GCP VM.
        The fair version is to retrain v1's recipe (single video, naive HDR decode, splatfacto defaults) through the same eval.
      - 708 MB is too big for the web, so Phase 4 needs compression (SPZ, pruning).
