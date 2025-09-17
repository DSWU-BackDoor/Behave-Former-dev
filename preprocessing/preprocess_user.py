# -*- coding: utf-8 -*-
"""
dataEunkyo 디렉터리 안 user*.json 파일을 읽어서
(각 파일 안에는 {} 객체 5개 정도가 줄 단위로 들어있음)
HuMIdb 스타일로 전처리 후 CSV 1개로 저장
"""

# -*- coding: utf-8 -*-
"""
dataEunkyo 디렉터리 안 user*.json 파일을 읽어서
(각 파일 안에는 {} 객체 5개 정도가 줄 단위로 들어있음)
HuMIdb 스타일로 전처리 후 CSV 1개로 저장
"""

import os, json, math, argparse, glob, uuid
import numpy as np
import pandas as pd

EPS = 1e-6

# ---------------- HuMIdb-like helpers ----------------
def _clamp_xy_to_screen(x_arr, y_arr, w, h):
    x = np.clip(np.asarray(x_arr, dtype=np.float64), 0, max(w, EPS))
    y = np.clip(np.asarray(y_arr, dtype=np.float64), 0, max(h, EPS))
    return x, y

def _normalize_xy(x, y, w, h):
    return (x / max(w, EPS)).astype(np.float64), (y / max(h, EPS)).astype(np.float64)

def _duration_seconds(t0, t1, time_unit="ms"):
    dt = float(t1) - float(t0)
    if time_unit == "ms": return max(dt/1000.0, EPS)
    if time_unit == "s":  return max(dt, EPS)
    return max(dt/1000.0, EPS)

def _extract_minimal_from_raw(event_time, x_raw, y_raw, w, h, time_unit="ms"):
    if len(event_time) == 0: return None
    x_c, y_c = _clamp_xy_to_screen(x_raw, y_raw, w, h)
    x_n, y_n = _normalize_xy(x_c, y_c, w, h)
    duration_s = _duration_seconds(event_time[0], event_time[-1], time_unit=time_unit)
    sx, sy = float(x_n[0]), float(y_n[0])
    ex, ey = float(x_n[-1]), float(y_n[-1])
    dist = math.hypot(ex - sx, ey - sy)
    speed = dist / max(duration_s, EPS)
    if not np.isfinite(speed): speed = 0.0
    return [sx, sy, ex, ey, duration_s, speed]

def _repair_minimal_6d(item, w, h, time_unit="ms"):
    sx = float(item.get("start_x", item.get("sx", 0.0)))
    sy = float(item.get("start_y", item.get("sy", 0.0)))
    ex = float(item.get("end_x",   item.get("ex", 0.0)))
    ey = float(item.get("end_y",   item.get("ey", 0.0)))
    dur = float(item.get("duration", item.get("duration_s", 0.0)))
    spd = float(item.get("speed", 0.0))

    if max(sx, sy, ex, ey) > 1.0:
        sx_, sy_ = _clamp_xy_to_screen([sx], [sy], w, h); sx, sy = float(sx_[0]), float(sy_[0])
        ex_, ey_ = _clamp_xy_to_screen([ex], [ey], w, h); ex, ey = float(ex_[0]), float(ey_[0])
        sx, sy = sx / max(w, EPS), sy / max(h, EPS)
        ex, ey = ex / max(w, EPS), ey / max(h, EPS)
        dist = math.hypot(ex - sx, ey - sy)
        dur_s = dur if time_unit == "s" else max(dur/1000.0, EPS)
        spd = dist / max(dur_s, EPS)
        dur = dur_s
    else:
        if time_unit != "s":
            dur = max(dur/1000.0, EPS)

    if not np.isfinite(spd) or spd < 0: spd = 0.0
    if not np.isfinite(dur) or dur <= 0: dur = EPS
    return [sx, sy, ex, ey, dur, spd]

# ---------------- 전처리 ----------------
def preprocess_payload(payload: dict, default_session=None):
    userid = payload.get("userid", "")
    dev = payload.get("device", {})
    w = int(dev.get("screenWidth", 1080))
    h = int(dev.get("screenHeight", 2400))
    orientation = int(dev.get("orientation", 1))
    time_unit = payload.get("time_unit", "ms")
    swipes = payload.get("swipes", [])

    session_id = payload.get("session_id", default_session or uuid.uuid4().hex[:8])

    rows = []
    for i, s in enumerate(swipes):
        if "event_time" in s and "x" in s and "y" in s:
            vec = _extract_minimal_from_raw(s["event_time"], s["x"], s["y"], w, h, time_unit)
        else:
            vec = _repair_minimal_6d(s, w, h, time_unit)
        if vec is None: continue
        sx, sy, ex, ey, dur, spd = vec
        swipe_idx = int(s.get("swipe_idx", i))
        rows.append({
            "userid": userid,
            "session_id": session_id,
            "swipe_idx": swipe_idx,
            "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey,
            "duration_s": dur, "speed": spd,
            "screenWidth": w, "screenHeight": h, "orientation": orientation
        })
    return rows

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="dataEunkyo", help="유저 json 파일 디렉터리")
    ap.add_argument("--out_csv", default="out", help="결과 CSV 경로")
    args = ap.parse_args()

    all_rows = []
    for path in sorted(glob.glob(os.path.join(args.indir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                payload = json.loads(line)
                rows = preprocess_payload(payload)
                all_rows.extend(rows)

    if not all_rows:
        print("[WARN] 전처리된 결과가 없음")
        return

    df = pd.DataFrame(all_rows)
    df.sort_values(by=["userid","session_id","swipe_idx"], inplace=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")

    print(f"[OK] saved {args.out_csv} (rows={len(df)})")

if __name__ == "__main__":
    main()
