"""Turn each demo clip into a 3D gripper trajectory and an open/close signal.

Pipeline per clip (all positions in metres, in the ArUco marker's frame:
origin = marker centre, z = up from the floor):

  1. Camera: detect the marker (5x5 dict, id 68, 100 mm) and the A4 sheet it is
     printed on; estimate the focal length from both, then the camera pose.
  2. Bottle: find it automatically from the first-vs-last frame difference (it
     is the only thing that moved from its start spot), giving its start position,
     height and drop position. A template tracker detects the moment of lift-off.
  3. Carry: the hand's thumb-index midpoint is intersected with the vertical
     plane through the bottle's start and drop positions (the carry stays close
     to that plane), which gives 3D without trusting monocular depth.
  4. Hand: MediaPipe finds the hand landmarks; the working hand is the one that
     moves (the resting hand, or a foot mistaken for a hand, stays put).
  5. Reach / retreat: PnP on MediaPipe's metric hand model gives depth, rescaled
     so it meets the carry path exactly at lift-off and release.
     Gripper closes just before lift-off and opens at release.

How much to trust each part: the carry (lift-off to release) is geometric and
reliable; the reach and retreat depend on monocular hand depth and are
approximate (good for direction and timing, a few cm to ~10 cm off in depth).

Outputs, per episode, in --out:
  ep_XXX.npz        t, pos (N,3), gripper (N,) 1=closed, segment (N,) 0 reach /
                    1 carry / 2 retreat, plus raw tracks
  ep_XXX.json       events, bottle start/drop, focal length, quality stats
  ep_XXX_debug.png  path drawn on the first frame + x/y/z/gripper over time
  cache/ep_XXX.v2.pkl raw hand/bottle tracks, reused on reruns (delete to recompute)
and summary.csv across all episodes.

Usage:
    python src/extract_trajectory.py                       # every clip in data/clips
    python src/extract_trajectory.py --episodes ep_003 ep_029
    python src/extract_trajectory.py --focal 2660          # fix focal (px at 3840 wide)
"""

import argparse
import csv
import json
import os
import pickle
import urllib.request
from collections import deque
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")       # quiet MediaPipe's C++ log lines
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
from scipy.signal import medfilt, savgol_filter

MARKER_ID = 68
MARKER_SIZE = 0.100            # m, measured black square
SHEET = (0.297, 0.210)         # A4, m
WORK_W = 1920                  # processing width (px); everything is rescaled to this
DEFAULT_FOCAL_4K = 2660        # px at 3840 wide; median over clips where the A4 sheet was found
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")


# ----------------------------------------------------------------------------- camera

def detect_marker(frames_gray):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000),
                                  cv2.aruco.DetectorParameters())
    found = []
    for g in frames_gray:
        c, ids, _ = det.detectMarkers(g)
        if ids is not None and MARKER_ID in ids.ravel():
            found.append(c[list(ids.ravel()).index(MARKER_ID)].reshape(4, 2))
    if not found:
        raise RuntimeError(f"marker {MARKER_ID} not found in the first frames")
    return np.median(found, axis=0).astype(np.float32)


def detect_sheet(frame_bgr, marker_px):
    """Corners of the white A4 sheet that contains the marker (full-res pixels).

    "White" is measured on the paper itself (a ring just outside the marker), so the
    threshold follows the phone's auto-exposure instead of being a fixed number.
    """
    s = frame_bgr.shape[1] / WORK_W
    small = cv2.resize(frame_bgr, (WORK_W, round(frame_bgr.shape[0] / s)))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mk = (marker_px / s).astype(np.float32)
    centre = mk.mean(0)
    inner = np.zeros(hsv.shape[:2], np.uint8)
    outer = np.zeros(hsv.shape[:2], np.uint8)
    cv2.fillConvexPoly(inner, np.round(centre + 1.25 * (mk - centre)).astype(np.int32), 1)
    cv2.fillConvexPoly(outer, np.round(centre + 1.6 * (mk - centre)).astype(np.int32), 1)
    ring = (outer > 0) & (inner == 0)
    paper_v = float(np.percentile(hsv[..., 2][ring], 60))
    paper_s = float(np.percentile(hsv[..., 1][ring], 60))
    for frac in (0.85, 0.78, 0.70):
        mask = ((hsv[..., 1] < max(40, paper_s + 25)) & (hsv[..., 2] > frac * paper_v)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            quad = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True).reshape(-1, 2)
            if len(quad) == 4 and cv2.pointPolygonTest(quad.astype(np.float32),
                                                        tuple(float(v) for v in centre), False) > 0:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                pts = (quad * s).astype(np.float32).reshape(-1, 1, 2)
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
                return cv2.cornerSubPix(gray, pts, (7, 7), (-1, -1), crit).reshape(4, 2)
    return None


def marker_obj_points():
    h = MARKER_SIZE / 2
    return np.float32([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])


def intrinsics(f, w, h):
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def marker_pose(marker_px, K):
    ok, rvec, tvec = cv2.solvePnP(marker_obj_points(), marker_px, K, None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise RuntimeError("marker pose failed")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()


def sheet_error(f, w, h, marker_px, sheet_px):
    """How far the sheet, back-projected onto the floor, is from a 297x210 rectangle."""
    K = intrinsics(f, w, h)
    R, t = marker_pose(marker_px, K)
    n, d = R[:, 2], R[:, 2] @ t
    Kinv = np.linalg.inv(K)
    P = []
    for p in sheet_px:
        r = Kinv @ np.array([p[0], p[1], 1.0])
        P.append(r * (d / (n @ r)))
    P = np.array(P)
    edges = [P[(i + 1) % 4] - P[i] for i in range(4)]
    sides = sorted(np.linalg.norm(e) for e in edges)
    cos = [abs(edges[i] @ edges[(i + 1) % 4]) / (np.linalg.norm(edges[i]) * np.linalg.norm(edges[(i + 1) % 4]))
           for i in range(4)]
    target = [SHEET[1], SHEET[1], SHEET[0], SHEET[0]]
    return sum((a - b) ** 2 for a, b in zip(sides, target)) + 0.05 * sum(c * c for c in cos), sides


def estimate_focal(w, h, marker_px, sheet_px):
    """Focal length (px at this resolution) that makes the sheet a true A4 rectangle."""
    grid = np.arange(1500, 4500, 20) * w / 3840
    errs = [sheet_error(f, w, h, marker_px, sheet_px)[0] for f in grid]
    best = grid[int(np.argmin(errs))]
    fine = np.arange(best - 25 * w / 3840, best + 25 * w / 3840, 1.0)
    errs = [sheet_error(f, w, h, marker_px, sheet_px)[0] for f in fine]
    f = float(fine[int(np.argmin(errs))])
    return f, sheet_error(f, w, h, marker_px, sheet_px)[1]


class Camera:
    """Pinhole camera at WORK_W resolution, posed in the marker frame."""

    def __init__(self, K, R, t):
        self.K, self.Kinv = K, np.linalg.inv(K)
        self.R, self.t = R, t                      # marker -> camera
        self.origin = -R.T @ t                     # camera centre in marker frame

    def ray(self, px):
        d = self.R.T @ (self.Kinv @ np.array([px[0], px[1], 1.0]))
        return d / np.linalg.norm(d)

    def floor_point(self, px, z=0.0):
        d = self.ray(px)
        s = (z - self.origin[2]) / d[2]
        return self.origin + s * d

    def to_marker(self, p_cam):
        return self.R.T @ (np.asarray(p_cam) - self.t)

    def project(self, P):
        P = np.atleast_2d(P)
        pc = (self.R @ P.T).T + self.t
        uv = (self.K @ pc.T).T
        return uv[:, :2] / uv[:, 2:3]


def closest_on_vertical(cam, px, base_xy):
    """Height where the pixel's ray passes closest to the vertical line above base_xy."""
    d, o = cam.ray(px), cam.origin
    # minimise |o + s d - (bx, by, z)| over s, z
    A = np.array([[d[0], 0], [d[1], 0], [d[2], -1]])
    b = np.array([base_xy[0] - o[0], base_xy[1] - o[1], -o[2]])
    s, z = np.linalg.lstsq(A, b, rcond=None)[0]
    return z


def ray_plane(cam, px, p0, normal):
    d = cam.ray(px)
    s = ((p0 - cam.origin) @ normal) / (d @ normal)
    return cam.origin + s * d


# ----------------------------------------------------------------------------- video

def work_size(w, h):
    return WORK_W, round(h * WORK_W / w)


def count_stale(frames, from_end=False):
    """Frames at a clip edge that belong to a different shot (left over from a cut).

    They differ strongly from the settled scene next to them, whereas a normal
    frame differs only by sensor noise and slight movement.
    """
    small = [cv2.resize(f, (240, 135), interpolation=cv2.INTER_AREA).astype(np.float32) for f in frames]
    if from_end:
        small = small[::-1]
    ref = np.median(np.stack(small[4:14]), axis=0)
    k = 0
    while k < 3 and float(np.abs(small[k] - ref).mean()) > 2.5:
        k += 1
    return k


def ends_of_clip(path, n_head=10, n_tail=5):
    """First/last frames of a clip, with leftover frames from editing cuts dropped.

    Returns full-res head frames, resized head frames, resized tail frames, fps and
    (skip_head, skip_tail): how many stale frames to ignore at each end.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    head = []
    for _ in range(n_head + 14):
        ok, f = cap.read()
        if ok:
            head.append(f)
    if len(head) < 15:
        raise RuntimeError(f"could not read {path}")
    size = work_size(head[0].shape[1], head[0].shape[0])
    tail = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - n_tail - 20))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        tail.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA))
    cap.release()
    skip_head = count_stale(head)
    skip_tail = count_stale(tail, from_end=True) if len(tail) >= 15 else 0
    head = head[skip_head:skip_head + n_head]
    tail = tail[:len(tail) - skip_tail][-n_tail:]
    small_head = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in head]
    return head, small_head, tail, fps, (skip_head, skip_tail)


def stream(path, skip_head=0, skip_tail=0):
    """Resized frames, leaving out skip_head frames at the start and skip_tail at the end."""
    cap = cv2.VideoCapture(str(path))
    size, i, buf = None, 0, deque()
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        if i <= skip_head:
            continue
        if size is None:
            size = work_size(f.shape[1], f.shape[0])
        buf.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA))
        if len(buf) > skip_tail:
            yield buf.popleft()
    cap.release()


# ----------------------------------------------------------------------------- bottle

def moved_blobs(first, last):
    d = cv2.absdiff(cv2.GaussianBlur(first, (9, 9), 0), cv2.GaussianBlur(last, (9, 9), 0)).max(2)
    m = (d > 40).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    return [tuple(int(v) for v in stats[i, :5]) for i in range(1, n)]   # x, y, w, h, area


def find_bottle(head, tail, cam):
    """Start box of the bottle and its drop position, from what moved during the clip."""
    first = np.median(np.stack(head[:5]), axis=0).astype(np.uint8)
    last = np.median(np.stack(tail), axis=0).astype(np.uint8)
    blobs = [b for b in moved_blobs(first, last) if b[4] > 1500 and b[1] > 20]  # skip body at top edge
    upright = [b for b in blobs if 1.6 < b[3] / b[2] < 4.0]
    if not upright:
        raise RuntimeError("could not find the bottle's start position")
    start = max(upright, key=lambda b: b[4])
    x, y, w, h, _ = start
    base = cam.floor_point((x + w / 2, y + h))
    height = closest_on_vertical(cam, (x + w / 2, y), base[:2])
    # the drop: the biggest other moved blob where the box can be. The box is 30-80 cm
    # from the start, roughly along +y (to the side of the bottle, as in the sim); this
    # rules out my body or arm, which also moved between the first and last frame.
    others = []
    for b in blobs:
        if b is start:
            continue
        p = cam.floor_point((b[0] + b[2] / 2, b[1] + b[3] * 0.7))
        d = p[:2] - base[:2]
        if 0.30 < np.linalg.norm(d) < 0.80 and d[1] > 0.25 and abs(d[0]) < 0.25:
            others.append((b[4], p, b))
    if not others:
        raise RuntimeError("could not find where the bottle ended up")
    _, drop, drop_box = max(others, key=lambda o: o[0])
    return (x, y, w, h), base, float(height), drop, drop_box


class BottleTracker:
    """Multi-scale normalised template matching, anchored to the first appearance."""

    def __init__(self, first_frame, box):
        x, y, w, h = box
        g0 = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.w, self.h = w, h
        self.tmpl0 = g0[y:y + h, x:x + w].astype(np.float32)
        self.tmpl = self.tmpl0.copy()
        self.cx, self.cy, self.scale = x + w / 2, y + h / 2, 1.0
        self.started = False

    def update(self, frame):
        if not self.started:          # first frame: the box we were given
            self.started = True
            return (self.cx, self.cy, self.w, self.h, 1.0)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        H, W = g.shape
        w, h = self.w, self.h
        best = (-1, self.cx, self.cy, self.scale)
        for s in (self.scale * 0.96, self.scale, self.scale * 1.04):
            tw, th = max(8, int(w * s)), max(8, int(h * s))
            t = cv2.resize(self.tmpl, (tw, th))
            rx, ry = int(0.6 * tw), int(0.45 * th)   # a bottle can't jump to its neighbour
            x0, y0 = int(max(0, self.cx - tw / 2 - rx)), int(max(0, self.cy - th / 2 - ry))
            x1, y1 = int(min(W, self.cx + tw / 2 + rx)), int(min(H, self.cy + th / 2 + ry))
            if x1 - x0 <= tw or y1 - y0 <= th:
                continue
            res = cv2.matchTemplate(g[y0:y1, x0:x1], t, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score > best[0]:
                best = (score, x0 + loc[0] + tw / 2, y0 + loc[1] + th / 2, s)
        score, ncx, ncy, ns = best
        if score > 0.35:
            self.cx, self.cy, self.scale = ncx, ncy, float(np.clip(ns, 0.6, 1.6))
            tw, th = int(w * self.scale), int(h * self.scale)
            ya, xa = int(self.cy - th / 2), int(self.cx - tw / 2)
            patch = g[max(0, ya):ya + th, max(0, xa):xa + tw]
            if patch.shape == (th, tw) and score > 0.6:
                self.tmpl = 0.85 * self.tmpl0 + 0.15 * cv2.resize(patch, (w, h))
        return (self.cx, self.cy, w * self.scale, h * self.scale, float(score))


# ----------------------------------------------------------------------------- hand

def ensure_hand_model(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading MediaPipe hand model -> {path}")
        urllib.request.urlretrieve(HAND_MODEL_URL, path)
    return str(path)


class HandDetector:
    def __init__(self, model_path, roi, fps):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp, self.roi, self.fps, self.i = mp, roi, fps, 0
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3)
        self.landmarker = vision.HandLandmarker.create_from_options(opts)

    def __call__(self, frame):
        x0, y0, x1, y1 = self.roi
        crop = np.ascontiguousarray(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        r = self.landmarker.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=crop), int(self.i * 1000 / self.fps))
        self.i += 1
        hands = []
        for lm, wl in zip(r.hand_landmarks, r.hand_world_landmarks):
            img = np.array([[p.x * (x1 - x0) + x0, p.y * (y1 - y0) + y0] for p in lm])
            world = np.array([[p.x, p.y, p.z] for p in wl])
            hands.append((img, world))
        return hands

    def close(self):
        self.landmarker.close()


def tracklets(per_frame):
    """Link detections frame to frame into short tracks: list of [(frame, det_index), ...]."""
    tracks, open_ = [], []                      # open_: indices into tracks still being extended
    for i, cands in enumerate(per_frame):
        pairs = []
        for ti in open_:
            fi, di = tracks[ti][-1]
            last = per_frame[fi][di][0][0]
            gap = i - fi - 1
            for dj, c in enumerate(cands):
                d = np.linalg.norm(c[0][0] - last)
                if d < 60 + 25 * gap:
                    pairs.append((d, ti, dj))
        used_t, used_d = set(), set()
        for d, ti, dj in sorted(pairs):
            if ti in used_t or dj in used_d:
                continue
            tracks[ti].append((i, dj))
            used_t.add(ti)
            used_d.add(dj)
        for dj in range(len(cands)):
            if dj not in used_d:
                tracks.append([(i, dj)])
                used_t.add(len(tracks) - 1)
        open_ = [ti for ti in used_t if i - tracks[ti][-1][0] <= 6] + \
                [ti for ti in open_ if ti not in used_t and i - tracks[ti][-1][0] <= 6]
    return tracks


def follow_hand(per_frame, track, lift):
    """Pick the hand that carries the bottle, one detection per frame.

    Things that stay put for a long time are not the working hand: the other hand
    resting on the floor, or a bare foot that MediaPipe mistakes for a hand. Those
    tracks are discarded; the working hand is then followed from lift-off, where
    it is the detection whose fingertips are at the bottle.
    """
    n = len(per_frame)
    static = set()
    for tr in tracklets(per_frame):
        pts = np.array([per_frame[f][d][0][0] for f, d in tr])
        span = np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)
        if len(tr) >= 30 and span.max() < 60:
            static.update(tr)
    cand = [[c for d, c in enumerate(per_frame[i]) if (i, d) not in static] for i in range(n)]

    def grip_px(c):                       # thumb-index midpoint in pixels
        return (c[0][4] + c[0][8]) / 2

    # The hand can be hidden behind the bottle while gripping, so look up to ~1.5 s
    # either side of lift-off, preferring detections close to the bottle and to lift-off.
    best = None
    bh = track[0, 3]
    for i in range(max(0, lift - 45), min(n, lift + 45)):
        for c in cand[i]:
            d = np.linalg.norm(grip_px(c) - track[i, :2]) / bh
            if d < 1.0:
                score = d + 0.02 * abs(i - lift)
                if best is None or score < best[0]:
                    best = (score, i, c)
    if best is None:
        raise RuntimeError("no hand found at the bottle around lift-off")
    _, seed, seed_c = best
    chosen = [None] * n
    chosen[seed] = seed_c

    def walk(order, prev):
        gap = 0
        for i in order:
            if cand[i]:
                j = int(np.argmin([np.linalg.norm(c[0][0] - prev) for c in cand[i]]))
                if np.linalg.norm(cand[i][j][0][0] - prev) < 120 + 40 * gap:
                    chosen[i], prev, gap = cand[i][j], cand[i][j][0][0], 0
                    continue
            gap += 1

    walk(range(seed + 1, n), seed_c[0][0])
    walk(range(seed - 1, -1, -1), seed_c[0][0])
    return chosen


def hand_point(hand, cam):
    """3D thumb-index midpoint (marker frame) via PnP on MediaPipe's metric hand model."""
    img, world = hand
    ok, rvec, tvec = cv2.solvePnP(world.astype(np.float64), img.astype(np.float64),
                                  cam.K, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok or tvec[2, 0] <= 0:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    grasp_world = (world[4] + world[8]) / 2
    p_cam = R @ grasp_world + tvec.ravel()
    aperture = np.linalg.norm(world[4] - world[8])
    return cam.to_marker(p_cam), aperture


# ----------------------------------------------------------------------------- main logic

def fill_gaps(t, X, valid):
    X = X.copy()
    if valid.sum() < 2:
        return X
    for k in range(X.shape[1]):
        X[:, k] = np.interp(t, t[valid], X[valid, k])
    return X


def process(clip, out_dir, model_path, focal_4k=None):
    name = clip.stem
    head, head_small, tail_small, fps, skips = ends_of_clip(clip)
    full = head[0]
    H0, W0 = full.shape[:2]
    s = WORK_W / W0
    h_work = head_small[0].shape[0]

    # 1. camera
    marker_full = detect_marker([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in head])
    sheet_full = detect_sheet(full, marker_full)
    focal_note = None
    if focal_4k:
        f_full, sheet_sides, focal_note = focal_4k * W0 / 3840, None, "focal given on command line"
    else:
        f_full, sheet_sides = (estimate_focal(W0, H0, marker_full, sheet_full)
                               if sheet_full is not None else (None, None))
        # The focal length belongs to the phone's lens, so it is the same in every clip.
        # If the sheet was not found, or the estimate is implausible, use the typical value.
        if f_full is None or abs(f_full * 3840 / W0 - DEFAULT_FOCAL_4K) > 150:
            focal_note = ("A4 sheet not found" if f_full is None else
                          f"sheet gave unlikely focal {f_full * 3840 / W0:.0f}px") + \
                f"; used default {DEFAULT_FOCAL_4K}px"
            f_full, sheet_sides = DEFAULT_FOCAL_4K * W0 / 3840, None
    K = intrinsics(f_full * s, WORK_W, h_work)
    R, tv = marker_pose(marker_full * s, K)
    cam = Camera(K, R, tv)

    # 2. bottle start/drop, then one streaming pass for bottle tracking + hand landmarks
    box, base, bottle_h, drop, drop_box = find_bottle(head_small, tail_small, cam)
    bx, by, bw, bh = box
    roi_x0 = max(0, min(bx, drop_box[0]) - 260)
    roi_x1 = min(WORK_W, max(bx + bw, drop_box[0] + drop_box[2]) + 260)
    roi_y1 = min(h_work, max(by + bh, drop_box[1] + drop_box[3]) + 60)
    roi = (int(roi_x0), 0, int(roi_x1), int(roi_y1))
    cache = out_dir / "cache" / f"{name}.v2.pkl"
    if cache.exists():                       # tracking is the slow part; reuse it
        with open(cache, "rb") as fh:
            track, per_frame = pickle.load(fh)
    else:
        tracker = BottleTracker(head_small[0], box)
        detector = HandDetector(model_path, roi, fps)
        track, per_frame = [], []
        try:
            for frame in stream(clip, *skips):
                track.append(tracker.update(frame))
                per_frame.append(detector(frame))
        finally:
            detector.close()
        track = np.array(track)
        cache.parent.mkdir(exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump((track, per_frame), fh)
    n = len(track)
    t = np.arange(n) / fps
    first_small = head_small[0]

    # lift-off: bottle centre moved by >12% of its height (and stays moved)
    disp = np.hypot(track[:, 0] - track[0, 0], track[:, 1] - track[0, 1]) / bh
    rise = (track[0, 1] - track[:, 1]) / bh
    moved = (disp > 0.12) & (rise > 0.06)
    lift = next((i for i in range(n - 3) if moved[i:i + 4].all()), None)
    if lift is None:
        raise RuntimeError("bottle never lifted")

    # 3. hand: 3D grasp point per frame, following the hand that grabs the bottle
    chosen = follow_hand(per_frame, track, lift)
    hand = np.full((n, 3), np.nan)             # PnP thumb-index midpoint, marker frame
    grip_px = np.full((n, 2), np.nan)          # same point in the image
    aperture = np.full(n, np.nan)
    for i, hnd in enumerate(chosen):
        if hnd is not None:
            grip_px[i] = (hnd[0][4] + hnd[0][8]) / 2
            p, a = hand_point(hnd, cam)
            if p is not None and np.linalg.norm(p - cam.origin) < 3.0:
                hand[i], aperture[i] = p, a
    seen = ~np.isnan(grip_px[:, 0])
    if seen.mean() < 0.5:
        raise RuntimeError(f"hand tracked in only {seen.mean():.0%} of frames")
    grip_f = fill_gaps(t, grip_px, seen)
    grip_f = savgol_filter(grip_f, 9 if n > 9 else (n - 1) | 1, 2, axis=0)

    # 4. carry: the hand moves from the bottle's start to the box, so it stays close
    #    to the vertical plane through those two points. Intersecting the hand's
    #    camera ray with that plane gives its 3D position without needing depth.
    p0 = np.array([base[0], base[1], 0.0])
    along = np.array([drop[0] - base[0], drop[1] - base[1], 0.0])
    span = float(np.linalg.norm(along))
    along /= span
    normal = np.cross(along, [0.0, 0.0, 1.0])
    on_plane = np.array([ray_plane(cam, g, p0, normal) for g in grip_f])
    prog = (on_plane - p0) @ along / span

    # release: first time the hand is >80% of the way to the drop, at its lowest point there
    z_s = savgol_filter(on_plane[:, 2], 15 if n > 15 else (n - 1) | 1, 2)
    over = [i for i in range(lift + 5, n) if prog[i] > 0.8]
    if over:
        block = [over[0]]
        for i in over[1:]:
            if i - block[-1] > 5:
                break
            block.append(i)
        release = min(block, key=lambda i: z_s[i])
    else:
        release = int(lift + 5 + np.argmax(prog[lift + 5:]))

    # 5. reach and retreat: MediaPipe's hand-size depth, rescaled along each camera ray so
    #    it meets the carry path exactly at lift-off and at release (an average hand model
    #    gets distance wrong by a roughly constant factor within a motion).
    def rescaled(anchor_frame, frames):
        near = []
        for half in (8, 30, 60):              # widen if the hand was hidden at that moment
            near = [i for i in range(max(0, anchor_frame - half), min(n, anchor_frame + half + 1))
                    if not np.isnan(hand[i, 0])]
            if near:
                break
        if not near:
            return None, np.nan
        target = np.linalg.norm(on_plane[anchor_frame] - cam.origin)
        k = float(np.median([target / np.linalg.norm(hand[i] - cam.origin) for i in near]))
        dist = np.linalg.norm(hand - cam.origin, axis=1) * k
        idx = [i for i in frames if not np.isnan(dist[i])]
        if len(idx) < 2:
            return None, k
        d = np.interp(frames, idx, dist[idx])
        d = medfilt(d, 15) if len(d) >= 15 else d
        d = savgol_filter(d, 31, 2) if len(d) > 31 else d
        rays = np.array([cam.ray(grip_f[i]) for i in frames])
        return cam.origin + rays * d[:, None], k

    pos = on_plane.copy()
    reach, k_in = rescaled(lift, list(range(0, lift)))
    if reach is not None:
        pos[:lift] = reach + (on_plane[lift] - reach[-1]) * (np.arange(lift) / max(lift - 1, 1))[:, None] ** 2 \
            if lift > 1 else reach
    after = list(range(release + 1, n))
    k_out = np.nan
    if after:
        retreat, k_out = rescaled(release, after)
        if retreat is not None:
            fade = np.clip(1 - (t[after] - t[release]) / 1.0, 0, 1)[:, None]
            pos[after] = retreat + fade * (on_plane[release] - retreat[0])
    pos[:, 2] = np.maximum(pos[:, 2], 0.01)
    pos = savgol_filter(pos, 15 if n > 15 else (n - 1) | 1, 2, axis=0)

    close = max(0, lift - int(0.25 * fps))
    gripper = np.zeros(n)
    gripper[close:release + 1] = 1.0
    segment = np.zeros(n, dtype=np.int8)       # 0 reach, 1 carry, 2 retreat
    segment[lift:release + 1] = 1
    segment[release + 1:] = 2
    dxy = np.linalg.norm(pos[:, :2] - drop[:2], axis=1)
    grasp_height = float(on_plane[lift, 2])

    # quality
    pre, post = slice(0, lift), slice(release + 1, n)
    q = {
        "hand_coverage_reach": float(seen[pre].mean()) if lift else 0.0,
        "hand_coverage_retreat": float(seen[post].mean()) if release + 1 < n else 0.0,
        "hand_coverage_carry": float(seen[lift:release + 1].mean()),
        "hand_depth_scale_reach": k_in,
        "hand_depth_scale_retreat": float(k_out),
        "grasp_height_m": grasp_height,
        "release_dist_to_drop_m": float(dxy[release]),
    }
    flags = []
    if q["hand_coverage_reach"] < 0.5:
        flags.append("low hand coverage on reach")
    if q["hand_coverage_carry"] < 0.6:
        flags.append("low hand coverage during carry")
    if not 0.02 < grasp_height < bottle_h:
        flags.append(f"grasp height {grasp_height:.2f} m is not on the bottle")
    if q["release_dist_to_drop_m"] > 0.15:
        flags.append("release not above the drop spot")
    if not 0.12 < bottle_h < 0.30:
        flags.append(f"odd bottle height {bottle_h:.2f} m")

    meta = {
        "episode": name, "fps": fps, "frames": n, "source_resolution": [W0, H0],
        "focal_px_at_3840": f_full * 3840 / W0,
        "sheet_sides_m": None if sheet_sides is None else [round(float(v), 4) for v in sheet_sides],
        "focal_note": focal_note,
        "stale_frames_dropped": list(skips),
        "camera_position_m": cam.origin.round(4).tolist(),
        "bottle_start_xy_m": base[:2].round(4).tolist(), "bottle_height_m": round(bottle_h, 4),
        "bottle_drop_xy_m": drop[:2].round(4).tolist(),
        "t_gripper_close_s": round(float(t[close]), 3), "t_lift_s": round(float(t[lift]), 3),
        "t_release_s": round(float(t[release]), 3),
        "quality": {k: round(v, 3) for k, v in q.items()}, "flags": flags,
    }
    np.savez_compressed(out_dir / f"{name}.npz", t=t, pos=pos, gripper=gripper, segment=segment, hand_pnp=hand,
                        hand_on_plane=on_plane, grip_px=grip_px, bottle_px=track, aperture=aperture)
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    debug_plot(out_dir / f"{name}_debug.png", first_small, cam, t, pos, gripper,
               np.where(seen[:, None], on_plane, np.nan), lift, release, base, drop, box, name)
    return meta


def debug_plot(path, first, cam, t, pos, gripper, hand, lift, release, base, drop, box, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 6.5))
    ax = fig.add_axes([0.01, 0.06, 0.55, 0.88])
    ax.imshow(cv2.cvtColor(first, cv2.COLOR_BGR2RGB))
    uv = cam.project(pos)
    sc = ax.scatter(uv[:, 0], uv[:, 1], c=t, s=5, cmap="viridis")
    floor = cam.project(np.c_[pos[:, :2], np.zeros(len(pos))])
    ax.plot(floor[:, 0], floor[:, 1], color="white", lw=0.8, alpha=0.7, label="path shadow on floor")
    for p, lab, col in [(base, "bottle start", "red"), (drop, "drop", "orange")]:
        q = cam.project([p[0], p[1], 0])[0]
        ax.plot(*q, "x", color=col, ms=12, mew=3, label=lab)
    for i, lab in [(lift, "lift"), (release, "release")]:
        q = uv[i]
        ax.plot(*q, "o", mfc="none", mec="white", ms=12, mew=2)
        ax.annotate(lab, q, xytext=(8, -8), textcoords="offset points", color="white", fontsize=10)
    x, y, w, h = box
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="red", lw=1))
    ax.set_xlim(0, first.shape[1]); ax.set_ylim(first.shape[0], 0)
    ax.axis("off"); ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"{name}: gripper path (colour = time)")
    fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.01, label="s")

    labels = ["x (m)", "y (m)", "z (m)"]
    for k in range(3):
        a = fig.add_axes([0.62, 0.70 - 0.21 * k, 0.36, 0.18])
        a.plot(t, hand[:, k], ".", ms=2, color="0.6", label="hand on carry plane")
        a.plot(t, pos[:, k], lw=1.8, color="tab:blue", label="final")
        a.axvspan(t[lift], t[release], color="tab:green", alpha=0.08)
        a.set_ylabel(labels[k]); a.tick_params(labelbottom=False)
        for ref, col in [(base, "red"), (drop, "orange")]:
            if k < 2:
                a.axhline(ref[k], color=col, lw=0.8, ls="--")
        if k == 0:
            a.legend(fontsize=7, ncol=2, loc="upper left")
    a = fig.add_axes([0.62, 0.07, 0.36, 0.12])
    a.step(t, gripper, where="post", color="k"); a.set_yticks([0, 1], ["open", "closed"])
    a.set_xlabel("time (s)")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", default="data/clips")
    p.add_argument("--out", default="data/trajectories")
    p.add_argument("--episodes", nargs="+", help="e.g. ep_003 ep_029 (default: all)")
    p.add_argument("--focal", type=float, help="focal length in px for 3840-wide video (default: estimate)")
    p.add_argument("--hand-model", default="models/hand_landmarker.task")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = sorted(Path(args.clips).glob("ep_*.mp4"))
    if args.episodes:
        clips = [c for c in clips if c.stem in args.episodes]
    model = ensure_hand_model(args.hand_model)

    rows = []
    for clip in clips:
        try:
            m = process(clip, out, model, args.focal)
            note = f" ({m['focal_note']})" if m["focal_note"] else ""
            print(f"{m['episode']}: lift {m['t_lift_s']}s, release {m['t_release_s']}s, "
                  f"focal {m['focal_px_at_3840']:.0f}px{note}  {'; '.join(m['flags']) or 'ok'}")
            rows.append({"episode": m["episode"], "status": "ok" if not m["flags"] else "check",
                         "flags": "; ".join(m["flags"]), "t_lift_s": m["t_lift_s"],
                         "t_release_s": m["t_release_s"], "focal_px_at_3840": round(m["focal_px_at_3840"]),
                         "bottle_height_m": m["bottle_height_m"],
                         "start_x": m["bottle_start_xy_m"][0], "start_y": m["bottle_start_xy_m"][1],
                         "drop_x": m["bottle_drop_xy_m"][0], "drop_y": m["bottle_drop_xy_m"][1],
                         **m["quality"]})
        except Exception as e:  # keep going; one bad clip should not stop the batch
            print(f"{clip.stem}: FAILED - {e}")
            rows.append({"episode": clip.stem, "status": "failed", "flags": str(e)})

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"\n{ok}/{len(rows)} clean, results in {out}/ (open the *_debug.png files)")


if __name__ == "__main__":
    main()
