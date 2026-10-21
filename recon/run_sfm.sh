#!/usr/bin/env bash
# Camera poses for the three orbit rings with COLMAP 4 (CPU build on macOS).
#   features -> custom ring-aware pairs -> global mapper -> undistort to PINHOLE
# Usage: bash recon/run_sfm.sh data/alma_v2
set -euo pipefail

SCENE=${1:-data/alma_v2}
PY=${PY:-.venv/bin/python}
DB=$SCENE/database.db

mkdir -p "$SCENE/sparse" "$SCENE/logs"
rm -f "$DB"

# One shared camera: same phone, same 1x lens, same resolution for all rings.
colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$SCENE/images" \
    --ImageReader.camera_model OPENCV \
    --ImageReader.single_camera 1 \
    --FeatureExtraction.use_gpu 0 \
    --SiftExtraction.max_num_features 8192 \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1 \
    2>&1 | tee "$SCENE/logs/features.log" | grep -E "Processed file|Error" | tail -3

$PY recon/make_pairs.py "$SCENE/frames.csv" "$SCENE/pairs.txt"

colmap matches_importer \
    --database_path "$DB" \
    --match_list_path "$SCENE/pairs.txt" \
    --match_type pairs \
    --FeatureMatching.use_gpu 0 \
    2>&1 | tee "$SCENE/logs/matching.log" | grep -E "Error" || true

colmap global_mapper \
    --database_path "$DB" \
    --image_path "$SCENE/images" \
    --output_path "$SCENE/sparse" \
    2>&1 | tee "$SCENE/logs/mapper.log" | grep -E "Error|registered" || true

# Report each model so a split reconstruction is obvious.
for m in "$SCENE"/sparse/*/; do
    echo "== $m"
    colmap model_analyzer --path "$m" 2>&1 | grep -E "Cameras|Images|Registered|Points|Mean reprojection|Mean track"
done

# gsplat / nerfstudio want undistorted PINHOLE images.
colmap image_undistorter \
    --image_path "$SCENE/images" \
    --input_path "$SCENE/sparse/0" \
    --output_path "$SCENE/undistorted" \
    --output_type COLMAP \
    2>&1 | tail -1
