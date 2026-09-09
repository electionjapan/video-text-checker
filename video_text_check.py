#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""動画内の指定文字列をOCRで検出するコア処理。

PaddleOCR 3.x を主対象にしつつ、旧2.x形式の結果も読み取れるようにしている。
"""
from __future__ import annotations

import argparse
import csv
import difflib
import html
import json
import math
import os
import shutil
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence



LogCallback = Callable[[str], None]
ProgressCallback = Callable[[float, int, int, int], None]


@dataclass
class OCRItem:
    text: str
    confidence: float
    box: list[list[float]]
    source_indices: tuple[int, ...] = field(default_factory=tuple)
    kind: str = "item"


@dataclass
class Match:
    target: str
    recognized_text: str
    confidence: float
    similarity: float
    box: list[list[float]]
    exact: bool


@dataclass
class HitEvent:
    target: str
    start_sec: float
    end_sec: float
    frame: int
    recognized_text: str
    confidence: float
    similarity: float
    image: str
    sightings: int
    center_x: float
    center_y: float
    exact: bool


def normalize_text(s: str | None) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    # OCRで混ざりやすい空白類を除去。改行・タブも比較上は不要。
    return "".join(ch for ch in s if not ch.isspace())


def split_targets(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        raw = value.replace("，", ",").replace("、", ",").replace(";", ",").replace("；", ",")
        parts: list[str] = []
        for line in raw.splitlines():
            parts.extend(line.split(","))
    else:
        parts = [str(x) for x in value]

    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        t = part.strip()
        key = normalize_text(t)
        if t and key and key not in seen:
            out.append(t)
            seen.add(key)
    return out


def fuzzy_score(candidate: str, target: str) -> float:
    cand = normalize_text(candidate)
    tgt = normalize_text(target)
    if not cand or not tgt:
        return 0.0
    if tgt in cand:
        return 1.0

    # target と同程度の長さの窓を走査する。
    # OCRが1文字欠落したケースも拾えるよう ±1 文字の窓も比較する。
    best = 0.0
    lengths = sorted({max(1, len(tgt) - 1), len(tgt), len(tgt) + 1})
    for win in lengths:
        if len(cand) >= win:
            for i in range(len(cand) - win + 1):
                score = difflib.SequenceMatcher(None, cand[i:i + win], tgt).ratio()
                best = max(best, score)
        else:
            best = max(best, difflib.SequenceMatcher(None, cand, tgt).ratio())
    return best


def is_match(candidate: str, target: str, threshold: float) -> tuple[bool, float, bool]:
    cand = normalize_text(candidate)
    tgt = normalize_text(target)
    if not cand or not tgt:
        return False, 0.0, False
    if tgt in cand:
        return True, 1.0, True
    score = fuzzy_score(cand, tgt)
    return score >= threshold, score, False


def format_timecode(seconds: float, filename_safe: bool = False) -> str:
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    sep = "-" if filename_safe else ":"
    return f"{h:02d}{sep}{m:02d}{sep}{s:02d}.{ms:03d}"


def _safe_box(box) -> list[list[float]]:
    try:
        pts = [[float(p[0]), float(p[1])] for p in box]
        if len(pts) >= 4:
            return pts[:4]
    except Exception:
        pass
    return [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]


def _bounds(box: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _union_box(items: Sequence[OCRItem]) -> list[list[float]]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for item in items:
        x0, y0, x1, y1 = _bounds(item.box)
        xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
    if not items:
        return _safe_box(None)
    x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _box_center(box: Sequence[Sequence[float]]) -> tuple[float, float]:
    x0, y0, x1, y1 = _bounds(box)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _line_compatible(a: OCRItem, b: OCRItem) -> bool:
    ax0, ay0, ax1, ay1 = _bounds(a.box)
    bx0, by0, bx1, by1 = _bounds(b.box)
    ah = max(1.0, ay1 - ay0)
    bh = max(1.0, by1 - by0)
    overlap = max(0.0, min(ay1, by1) - max(ay0, by0))
    overlap_ratio = overlap / min(ah, bh)
    ac = (ay0 + ay1) / 2
    bc = (by0 + by1) / 2
    center_ok = abs(ac - bc) <= 0.65 * max(ah, bh)
    return overlap_ratio >= 0.25 or center_ok


def build_candidates(items: Sequence[OCRItem], min_confidence: float) -> list[OCRItem]:
    """単独OCR枠に加え、同一行の近接OCR枠を結合した候補も作る。"""
    filtered: list[OCRItem] = []
    for idx, item in enumerate(items):
        if item.confidence >= min_confidence and normalize_text(item.text):
            filtered.append(OCRItem(item.text, item.confidence, item.box, (idx,), "item"))

    if not filtered:
        return []

    # 行グループ化
    ordered = sorted(filtered, key=lambda x: (_box_center(x.box)[1], _bounds(x.box)[0]))
    lines: list[list[OCRItem]] = []
    for item in ordered:
        best_line = None
        best_delta = float("inf")
        cy = _box_center(item.box)[1]
        for line in lines:
            if any(_line_compatible(item, other) for other in line):
                line_cy = sum(_box_center(x.box)[1] for x in line) / len(line)
                delta = abs(cy - line_cy)
                if delta < best_delta:
                    best_delta = delta
                    best_line = line
        if best_line is None:
            lines.append([item])
        else:
            best_line.append(item)

    candidates = list(filtered)

    # 同じ行でも大きく離れたテキストは別セグメントにする。
    for line in lines:
        line = sorted(line, key=lambda x: _bounds(x.box)[0])
        segments: list[list[OCRItem]] = []
        current: list[OCRItem] = []
        prev_x1 = None
        prev_h = None
        for item in line:
            x0, y0, x1, y1 = _bounds(item.box)
            h = max(1.0, y1 - y0)
            if current and prev_x1 is not None:
                gap = x0 - prev_x1
                gap_limit = max(40.0, 4.0 * max(h, prev_h or h))
                if gap > gap_limit:
                    segments.append(current)
                    current = []
            current.append(item)
            prev_x1 = x1
            prev_h = h
        if current:
            segments.append(current)

        for seg in segments:
            if len(seg) < 2:
                continue
            text = "".join(x.text for x in seg)
            conf = sum(x.confidence for x in seg) / len(seg)
            src = tuple(i for x in seg for i in x.source_indices)
            candidates.append(OCRItem(text, conf, _union_box(seg), src, "line"))

    return candidates


def find_matches(items: Sequence[OCRItem], targets: Sequence[str], threshold: float, min_confidence: float) -> list[Match]:
    candidates = build_candidates(items, min_confidence)
    matches: list[Match] = []

    # targetごとに「単独枠で既に一致したsource」を覚え、同じものを行結合で二重計上しない。
    for target in targets:
        source_hits: set[int] = set()
        target_matches: list[Match] = []
        for cand in candidates:
            ok, score, exact = is_match(cand.text, target, threshold)
            if not ok:
                continue
            if cand.kind == "line" and source_hits.intersection(cand.source_indices):
                continue
            if cand.kind == "item":
                source_hits.update(cand.source_indices)
            target_matches.append(Match(
                target=target,
                recognized_text=cand.text,
                confidence=float(cand.confidence),
                similarity=float(score),
                box=cand.box,
                exact=exact,
            ))

        # ほぼ同じ場所の重複候補は強い方だけ残す。
        kept: list[Match] = []
        for m in sorted(target_matches, key=lambda x: (x.exact, x.similarity, x.confidence), reverse=True):
            cx, cy = _box_center(m.box)
            duplicate = False
            for k in kept:
                kx, ky = _box_center(k.box)
                mx0, my0, mx1, my1 = _bounds(m.box)
                kx0, ky0, kx1, ky1 = _bounds(k.box)
                scale = max(20.0, min(max(mx1-mx0, my1-my0), max(kx1-kx0, ky1-ky0)))
                if math.hypot(cx-kx, cy-ky) <= scale * 0.8:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(m)
        matches.extend(kept)

    return matches


class OCRBackend:
    """Streamlit Community Cloud向け軽量OCRバックエンド。

    PaddleOCR 3.7 の公式ONNX Runtimeエンジンを使い、
    PP-OCRv5 mobile detection / recognition の2モデルだけをロードする。
    """

    def __init__(self, lang: str, log: LogCallback):
        # PaddleXの接続先チェックを省略。モデルのダウンロード自体は必要。
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")

        self.log = log
        self.mode = "v3-onnx-mobile"

        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCRの読み込みに失敗しました。Cloud logs を確認してください。"
            ) from exc

        try:
            # PaddleOCR公式ドキュメントにあるPP-OCRv5 mobile構成。
            # ONNX Runtimeを使い、Community CloudでPaddlePaddle本体を不要にする。
            self.ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                engine="onnxruntime",
                device="cpu",
                enable_mkldnn=False,
                cpu_threads=2,
                text_recognition_batch_size=1,
                text_det_limit_side_len=1280,
                text_det_limit_type="max",
            )
        except Exception as exc:
            raise RuntimeError(
                "OCRモデルの初期化に失敗しました。"
                "初回モデル取得・ONNX Runtime・CloudメモリのいずれかをCloud logsで確認してください。"
            ) from exc

        self.log("OCRエンジン準備完了（PaddleOCR 3.x / ONNX Runtime / PP-OCRv5 mobile）")

    def recognize(self, frame) -> list[OCRItem]:
        output = self.ocr.predict(frame)
        items: list[OCRItem] = []

        for res in output:
            data = getattr(res, "json", None)
            if callable(data):
                data = data()
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, dict):
                try:
                    data = dict(res)
                except Exception:
                    continue

            if isinstance(data.get("res"), dict):
                data = data["res"]

            texts = data.get("rec_texts") or []
            scores = data.get("rec_scores") or []
            polys = data.get("rec_polys")
            if polys is None:
                polys = data.get("dt_polys")
            if polys is None:
                polys = []

            n = min(len(texts), len(scores), len(polys))
            for i in range(n):
                items.append(
                    OCRItem(
                        str(texts[i]),
                        float(scores[i]),
                        _safe_box(polys[i]),
                    )
                )
        return items


def _looks_like_v2_line(line) -> bool:
    try:
        return len(line) >= 2 and len(line[0]) >= 4 and len(line[1]) >= 2 and isinstance(line[1][0], str)
    except Exception:
        return False



def _get_cv2():
    """OpenCVは動画処理開始時にだけ読み込む。Cloud起動時のネイティブ依存エラーを避ける。"""
    try:
        import cv2
        return cv2
    except Exception as exc:
        raise RuntimeError(
            "OpenCVの読み込みに失敗しました。Cloud logs の詳細を確認してください。"
        ) from exc


def draw_match(frame, box, out_path: str | Path) -> None:
    cv2 = _get_cv2()
    image = frame.copy()
    pts = [(int(round(p[0])), int(round(p[1]))) for p in box]
    if len(pts) >= 4:
        for i in range(4):
            cv2.line(image, pts[i], pts[(i + 1) % 4], (0, 0, 255), 4)
    cv2.imwrite(str(out_path), image)


def _default_log(msg: str) -> None:
    print(msg, flush=True)


def _default_progress(percent: float, checked: int, total_samples: int, hits: int) -> None:
    pass


def run(
    video_path: str,
    targets: str | Sequence[str] = "化学調味料",
    sample_fps: float = 5.0,
    threshold: float = 0.82,
    out_dir: str = "./result",
    lang: str = "japan",
    min_confidence: float = 0.35,
    merge_gap_sec: float = 1.0,
    log_callback: LogCallback | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    ocr_backend: OCRBackend | None = None,
) -> dict:
    log = log_callback or _default_log
    progress = progress_callback or _default_progress
    cancel_event = cancel_event or threading.Event()

    target_list = split_targets(targets)
    if not target_list:
        raise ValueError("検出ワードが指定されていません。")
    if sample_fps <= 0:
        raise ValueError("チェック回数は0より大きくしてください。")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("類似度しきい値は0〜1で指定してください。")
    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError("OCR信頼度しきい値は0〜1で指定してください。")

    cv2 = _get_cv2()

    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {video}")

    out = Path(out_dir)
    hits_dir = out / "hits"
    out.mkdir(parents=True, exist_ok=True)
    if hits_dir.exists():
        shutil.rmtree(hits_dir)
    hits_dir.mkdir(parents=True, exist_ok=True)

    if ocr_backend is None:
        log("OCRモデルを読み込み中です。初回だけモデルのダウンロードが入る場合があります…")
        ocr = OCRBackend(lang, log)
    else:
        ocr = ocr_backend
        log("OCRモデル準備済み（キャッシュを使用）")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if video_fps <= 0.01:
        video_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(video_fps / min(sample_fps, video_fps))))
    actual_sample_fps = video_fps / step
    total_samples = math.ceil(total_frames / step) if total_frames > 0 else 0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    diag = max(1.0, math.hypot(frame_w, frame_h))

    log(f"動画: {video.name}")
    log(f"動画FPS {video_fps:.3f} / 約 {actual_sample_fps:.2f} 回/秒でチェック")
    log("検出ワード: " + " / ".join(target_list))

    events: list[HitEvent] = []
    recent_by_target: dict[str, list[int]] = {t: [] for t in target_list}
    frame_idx = 0
    checked = 0
    cancelled = False
    last_progress_pct = -1.0

    try:
        while True:
            if cancel_event.is_set():
                cancelled = True
                log("キャンセル要求を受け付けました。途中までの結果を保存します。")
                break

            ret = cap.grab()
            if not ret:
                break

            if frame_idx % step == 0:
                ret, frame = cap.retrieve()
                if ret:
                    checked += 1
                    timestamp_sec = frame_idx / video_fps
                    items = ocr.recognize(frame)
                    matches = find_matches(items, target_list, threshold, min_confidence)

                    for match in matches:
                        cx, cy = _box_center(match.box)
                        merged_idx = None
                        best_dist = float("inf")
                        # 古い候補を整理しつつ、近い位置の直近イベントを探す。
                        keep_recent: list[int] = []
                        for idx in recent_by_target.get(match.target, []):
                            ev = events[idx]
                            if timestamp_sec - ev.end_sec <= merge_gap_sec:
                                keep_recent.append(idx)
                                dist = math.hypot(cx - ev.center_x, cy - ev.center_y) / diag
                                if dist <= 0.18 and dist < best_dist:
                                    best_dist = dist
                                    merged_idx = idx
                        recent_by_target[match.target] = keep_recent

                        if merged_idx is not None:
                            ev = events[merged_idx]
                            ev.end_sec = timestamp_sec
                            ev.sightings += 1
                            # より確からしい認識を代表表示に採用。
                            if (match.exact, match.similarity, match.confidence) > (ev.exact, ev.similarity, ev.confidence):
                                ev.recognized_text = match.recognized_text
                                ev.confidence = match.confidence
                                ev.similarity = match.similarity
                                ev.exact = match.exact
                                ev.center_x = cx
                                ev.center_y = cy
                            continue

                        tc_safe = format_timecode(timestamp_sec, filename_safe=True)
                        img_name = f"hit_{len(events)+1:04d}_{tc_safe}.jpg"
                        img_path = hits_dir / img_name
                        draw_match(frame, match.box, img_path)
                        event = HitEvent(
                            target=match.target,
                            start_sec=timestamp_sec,
                            end_sec=timestamp_sec,
                            frame=frame_idx,
                            recognized_text=match.recognized_text,
                            confidence=match.confidence,
                            similarity=match.similarity,
                            image=str(Path("hits") / img_name),
                            sightings=1,
                            center_x=cx,
                            center_y=cy,
                            exact=match.exact,
                        )
                        events.append(event)
                        recent_by_target.setdefault(match.target, []).append(len(events)-1)
                        log(f"検出 {format_timecode(timestamp_sec)}  [{match.target}]  認識: {match.recognized_text}")

                    if total_frames > 0:
                        pct = min(100.0, (frame_idx + 1) / total_frames * 100.0)
                    else:
                        pct = 0.0
                    if pct - last_progress_pct >= 0.25 or (total_frames > 0 and frame_idx + step >= total_frames):
                        last_progress_pct = pct
                        progress(pct, checked, total_samples, len(events))

            frame_idx += 1
    finally:
        cap.release()

    duration_sec = (max(0, total_frames - 1) / video_fps) if total_frames else 0.0
    write_csv(events, out / "report.csv")
    write_html(
        events,
        out / "report.html",
        video_path=str(video),
        targets=target_list,
        checked=checked,
        sample_fps=actual_sample_fps,
        threshold=threshold,
        min_confidence=min_confidence,
        cancelled=cancelled,
        duration_sec=duration_sec,
    )

    if not cancelled:
        progress(100.0, checked, total_samples, len(events))
    log(f"完了: {checked}フレームをOCRし、{len(events)}件の表示イベントを検出しました。")
    log(f"レポート: {out / 'report.html'}")
    return {
        "checked": checked,
        "events": len(events),
        "cancelled": cancelled,
        "report": str(out / "report.html"),
        "csv": str(out / "report.csv"),
        "output_dir": str(out),
        "actual_sample_fps": actual_sample_fps,
    }


def write_csv(events: Sequence[HitEvent], path: str | Path) -> None:
    fields = [
        "target", "start_time", "end_time", "duration_sec", "frame", "recognized_text",
        "confidence", "similarity", "match_type", "sightings", "image"
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in events:
            writer.writerow({
                "target": e.target,
                "start_time": format_timecode(e.start_sec),
                "end_time": format_timecode(e.end_sec),
                "duration_sec": round(max(0.0, e.end_sec - e.start_sec), 3),
                "frame": e.frame,
                "recognized_text": e.recognized_text,
                "confidence": round(e.confidence, 3),
                "similarity": round(e.similarity, 3),
                "match_type": "exact" if e.exact else "fuzzy",
                "sightings": e.sightings,
                "image": e.image,
            })


def write_html(
    events: Sequence[HitEvent],
    path: str | Path,
    video_path: str,
    targets: Sequence[str],
    checked: int,
    sample_fps: float,
    threshold: float,
    min_confidence: float,
    cancelled: bool,
    duration_sec: float,
) -> None:
    rows = []
    for e in events:
        duration = max(0.0, e.end_sec - e.start_sec)
        rows.append(f"""
        <tr>
          <td><strong>{html.escape(e.target)}</strong></td>
          <td>{format_timecode(e.start_sec)}<br><span class="sub">〜 {format_timecode(e.end_sec)}</span></td>
          <td>{duration:.1f}秒<br><span class="sub">確認 {e.sightings}回</span></td>
          <td><a href="{html.escape(e.image)}"><img src="{html.escape(e.image)}" alt="検出フレーム"></a></td>
          <td>{html.escape(e.recognized_text)}</td>
          <td>{e.confidence:.3f}</td>
          <td>{e.similarity:.3f}<br><span class="badge">{'完全一致' if e.exact else '類似一致'}</span></td>
        </tr>
        """)

    status = "途中キャンセル" if cancelled else "完了"
    target_text = " / ".join(html.escape(x) for x in targets)
    html_text = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>動画テキストチェック結果</title>
<style>
body {{ font-family: "Meiryo", "Yu Gothic UI", sans-serif; margin: 28px; background:#f5f7fa; color:#202124; }}
.wrap {{ max-width: 1400px; margin:auto; }}
h1 {{ font-size:24px; margin-bottom:8px; }}
.summary {{ background:white; border:1px solid #e1e5ea; border-radius:12px; padding:16px 20px; margin:16px 0 22px; line-height:1.8; }}
.count {{ font-size:30px; font-weight:700; }}
table {{ border-collapse:collapse; width:100%; background:white; border-radius:12px; overflow:hidden; }}
th,td {{ border-bottom:1px solid #e7eaee; padding:10px; text-align:left; vertical-align:top; }}
th {{ background:#eef2f6; position:sticky; top:0; }}
img {{ max-width:320px; max-height:190px; border:1px solid #ddd; }}
.sub {{ color:#6b7280; font-size:12px; }}
.badge {{ display:inline-block; margin-top:3px; padding:2px 6px; border-radius:9px; background:#eef2ff; font-size:11px; }}
.empty {{ text-align:center; padding:44px; color:#4b5563; }}
</style>
</head>
<body><div class="wrap">
<h1>動画テキストチェック結果</h1>
<div class="summary">
  <div><span class="count">{len(events)}</span> 件検出　／　状態: {status}</div>
  <div>対象動画: {html.escape(Path(video_path).name)}</div>
  <div>検出ワード: {target_text}</div>
  <div>動画長: {format_timecode(duration_sec)} ／ OCR確認フレーム: {checked} ／ 約 {sample_fps:.2f} 回/秒</div>
  <div class="sub">類似度しきい値 {threshold:.2f} ／ OCR信頼度しきい値 {min_confidence:.2f}</div>
</div>
<table>
<thead><tr><th>ワード</th><th>表示時刻</th><th>継続</th><th>画面</th><th>OCR認識</th><th>信頼度</th><th>一致度</th></tr></thead>
<tbody>
{''.join(rows) if rows else '<tr><td colspan="7" class="empty">検出されませんでした</td></tr>'}
</tbody></table>
</div></body></html>"""
    Path(path).write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="動画内の指定文字列をOCRで検出")
    parser.add_argument("video", help="入力動画ファイル")
    parser.add_argument("--target", action="append", default=None, help="検出ワード。複数回指定可")
    parser.add_argument("--fps", type=float, default=5.0, help="1秒あたりのOCR回数")
    parser.add_argument("--threshold", type=float, default=0.82, help="類似一致しきい値")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="OCR信頼度の下限")
    parser.add_argument("--out", default="./result", help="出力先")
    parser.add_argument("--lang", default="japan", help="PaddleOCR言語")
    args = parser.parse_args()
    run(
        args.video,
        targets=args.target or ["化学調味料"],
        sample_fps=args.fps,
        threshold=args.threshold,
        min_confidence=args.min_confidence,
        out_dir=args.out,
        lang=args.lang,
    )


if __name__ == "__main__":
    main()
