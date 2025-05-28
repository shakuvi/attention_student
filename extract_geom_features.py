import os
import re
import csv
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ───────────── CONFIG ─────────────
# Path to your raw Columbia Gaze Data Set (folders 0001,0002,…)
SRC_ROOT   = r'..\Columbia Gaze Data Set'

# Where to write the geometry+iris CSV
OUT_CSV    = 'geom_features.csv'

# How many degrees from center counts as “looking”
VERT_THRESH  = 10
HORIZ_THRESH = 10

# Regex to extract (headPose)P_(gazeVert)V_(gazeHoriz)H
#   e.g. "0001_2m_-15P_-10V_-10H.jpg"
PATT = re.compile(
    r'\d+_\d+m_([+-]?\d+)P_([+-]?\d+)V_([+-]?\d+)H',
    re.IGNORECASE
)

# 3D model points (mm) for solvePnP: nose tip, chin, eye corners, mouth corners
PNP_IDS  = [1, 152, 33, 263, 61, 291]
MODEL_3D = np.array([
    (  0.0,   0.0,    0.0),
    (  0.0, -63.6,  -12.5),
    (-43.3,  32.7,  -26.0),
    ( 43.3,  32.7,  -26.0),
    (-28.9, -28.9,  -24.1),
    ( 28.9, -28.9,  -24.1),
], dtype=np.float64)
# ────────────────────────────────────

def compute_head_pose(landmarks, w, h):
    pts2d = np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in PNP_IDS
    ], dtype=np.float64)
    cam = np.array([[w,0, w/2],
                    [0, w, h/2],
                    [0, 0,    1]], dtype=np.float64)
    dist = np.zeros((4,1))
    ok, rvec, _ = cv2.solvePnP(
        MODEL_3D, pts2d, cam, dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    pitch = float(np.degrees(np.arctan2(-R[2,0], sy)))
    yaw   = float(np.degrees(np.arctan2( R[1,0], R[0,0])))
    return yaw, pitch

def compute_iris_ratios(landmarks, w, h, iris_ids):
    # Separate left/right by splitting the sorted iris_ids in two
    ids = sorted(iris_ids)
    half = len(ids)//2
    left_ids, right_ids = ids[:half], ids[half:]
    def ratios(idxs):
        xs = np.array([landmarks[i].x * w for i in idxs])
        ys = np.array([landmarks[i].y * h for i in idxs])
        cx, cy = xs.mean(), ys.mean()
        xL, xR = xs.min(), xs.max()
        yT, yB = ys.min(), ys.max()
        # horizontal ratio in [0,1] across the eye corners
        h_ratio = (cx - xL) / (xR - xL + 1e-6)
        # vertical ratio in [0,1] top→bottom of iris region
        v_ratio = (cy - yT) / (yB - yT + 1e-6)
        return float(h_ratio), float(v_ratio)
    l_h, l_v = ratios(left_ids)
    r_h, r_v = ratios(right_ids)
    return l_h, l_v, r_h, r_v

def main():
    # Prepare CSV
    with open(OUT_CSV, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow([
            'subject','filename',
            'yaw','pitch',
            'l_h_ratio','l_v_ratio',
            'r_h_ratio','r_v_ratio',
            'label'
        ])

        # Setup MediaPipe FaceMesh
        mp_face = mp.solutions.face_mesh
        with mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        ) as face_mesh:

            # Build the unique iris landmark IDs from the built-in FACEMESH_IRISES edges
            iris_id_set = set()
            for edge in mp_face.FACEMESH_IRISES:
                iris_id_set.update(edge)
            iris_ids = sorted(iris_id_set)

            # Walk every subject folder
            for subj in sorted(os.listdir(SRC_ROOT)):
                subj_dir = os.path.join(SRC_ROOT, subj)
                if not os.path.isdir(subj_dir):
                    continue

                for fn in tqdm(os.listdir(subj_dir), desc=f"Subject {subj}"):
                    if not fn.lower().endswith(('.jpg','jpeg','png')):
                        continue

                    # Parse gaze labels from filename
                    m = PATT.match(fn)
                    if not m:
                        continue
                    gv = int(m.group(2))
                    gh = int(m.group(3))
                    label = 1 if abs(gv) <= VERT_THRESH and abs(gh) <= HORIZ_THRESH else 0

                    img = cv2.imread(os.path.join(subj_dir, fn))
                    if img is None:
                        continue
                    h, w = img.shape[:2]

                    # FaceMesh inference
                    res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if not res.multi_face_landmarks:
                        continue
                    lm = res.multi_face_landmarks[0].landmark

                    # Compute head-pose
                    yaw, pitch = compute_head_pose(lm, w, h)
                    if yaw is None:
                        continue

                    # Compute iris ratios
                    l_h, l_v, r_h, r_v = compute_iris_ratios(lm, w, h, iris_ids)

                    # Write row
                    writer.writerow([
                        subj, fn,
                        f"{yaw:.3f}", f"{pitch:.3f}",
                        f"{l_h:.3f}", f"{l_v:.3f}",
                        f"{r_h:.3f}", f"{r_v:.3f}",
                        label
                    ])

    # Quick sanity check
    import pandas as pd
    df = pd.read_csv(OUT_CSV)
    print(f"✅ Wrote {len(df)} rows to {OUT_CSV}")
    print(df.head())

if __name__ == '__main__':
    main()
