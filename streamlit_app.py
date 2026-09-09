from __future__ import annotations

import io
import os
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


APP_NAME = "動画テキストチェッカー"
# Community Cloud では /tmp 配下。ローカル実行時も一時領域に保存する。
APP_ROOT = Path(tempfile.gettempdir()) / "VideoTextChecker_Streamlit"
JOBS_ROOT = APP_ROOT / "jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title=APP_NAME, page_icon="🎬", layout="wide")

@st.cache_resource(show_spinner=False)
def get_core_module():
    """OCR/動画処理モジュールは必要になるまで読み込まない。"""
    import video_text_check
    return video_text_check



st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: Meiryo, "Yu Gothic UI", sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1350px; }
[data-testid="stMetricValue"] { font-size: 2rem; }
.note { color:#5f6368; font-size:0.9rem; }
.warning-box { border:1px solid #d93025; border-radius:10px; padding:12px 14px; background:#fff8f7; }
</style>
""",
    unsafe_allow_html=True,
)


def prune_old_jobs(hours: int = 12) -> None:
    cutoff = time.time() - hours * 3600
    try:
        for p in JOBS_ROOT.iterdir():
            if p.is_dir() and p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


@st.cache_resource(show_spinner="OCRエンジンを準備しています…（初回は軽量モデルを取得します）")
def get_ocr_backend():
    core = get_core_module()
    return core.OCRBackend(lang="japan", log=lambda _msg: None)


def make_job_dir(filename: str) -> Path:
    safe_stem = Path(filename).stem[:60].replace(" ", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = JOBS_ROOT / f"{stamp}_{safe_stem}"
    suffix = 1
    while path.exists():
        path = JOBS_ROOT / f"{stamp}_{safe_stem}_{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def write_upload(upload, dest: Path) -> None:
    upload.seek(0)
    with open(dest, "wb") as f:
        while True:
            chunk = upload.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    upload.seek(0)


def result_zip_bytes(out_dir: Path) -> bytes:
    buff = io.BytesIO()
    with zipfile.ZipFile(buff, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(out_dir))
    return buff.getvalue()


def load_result_table(csv_path: Path):
    import pandas as pd
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    rename = {
        "target": "検出ワード",
        "start_time": "開始",
        "end_time": "終了",
        "duration_sec": "継続秒",
        "recognized_text": "OCR認識",
        "confidence": "OCR信頼度",
        "similarity": "類似度",
        "match_type": "一致種別",
        "sightings": "確認回数",
        "image": "画像",
        "frame": "フレーム",
    }
    return df.rename(columns=rename)


def show_results(result: dict, out_dir: Path) -> None:
    st.divider()
    st.subheader("チェック結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("検出イベント", result.get("events", 0))
    c2.metric("OCRしたフレーム", result.get("checked", 0))
    c3.metric("実チェック頻度", f"{result.get('actual_sample_fps', 0):.2f} 回/秒")

    csv_path = out_dir / "report.csv"
    html_path = out_dir / "report.html"
    df = load_result_table(csv_path)

    if df.empty:
        st.success("指定ワードは検出されませんでした。")
    else:
        display_cols = [
            c for c in ["検出ワード", "開始", "終了", "継続秒", "OCR認識", "OCR信頼度", "類似度", "一致種別", "確認回数"]
            if c in df.columns
        ]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        st.markdown("#### 検出画面")
        max_show = min(len(df), 30)
        for i in range(max_show):
            row = df.iloc[i]
            cols = st.columns([1.25, 2.75])
            img_rel = row.get("画像")
            img_path = out_dir / str(img_rel) if img_rel else None
            with cols[0]:
                if img_path and img_path.exists():
                    st.image(str(img_path), use_container_width=True)
            with cols[1]:
                st.markdown(
                    f"**{row.get('検出ワード','')}**  ｜  {row.get('開始','')} ～ {row.get('終了','')}  "
                    f"  \nOCR: `{row.get('OCR認識','')}`  ｜  類似度: {row.get('類似度','')}  ｜  確認: {row.get('確認回数','')}回"
                )
        if len(df) > max_show:
            st.info(f"画面上では先頭{max_show}件を表示しています。全件はCSV/HTMLレポートで確認できます。")

    b1, b2 = st.columns(2)
    with b1:
        if csv_path.exists():
            st.download_button(
                "CSVをダウンロード",
                data=csv_path.read_bytes(),
                file_name="report.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with b2:
        st.download_button(
            "レポート一式をZIPでダウンロード",
            data=result_zip_bytes(out_dir),
            file_name=f"VideoTextChecker_result_{datetime.now():%Y%m%d_%H%M%S}.zip",
            mime="application/zip",
            use_container_width=True,
        )

    if html_path.exists():
        with st.expander("HTMLレポートをブラウザ内でプレビュー"):
            components.html(html_path.read_text(encoding="utf-8"), height=850, scrolling=True)


prune_old_jobs()

st.title("🎬 動画テキストチェッカー")
st.caption("動画内の指定文字列をOCRで探します。Cloud軽量版：ONNX Runtime / PP-OCRv5 mobile / 最大750MB。")

st.markdown(
    """
<div class="warning-box">
<b>Community Cloud試用時の注意</b><br>
アップロードした動画は外部クラウド上で処理されます。放送前素材・社外秘素材・個人情報を含む動画では試さず、
まず公開済み／テスト用の短い動画で動作確認してください。
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("検出設定")
    targets_text = st.text_area(
        "探すワード",
        value="化学調味料",
        height=120,
        help="複数の場合は1行に1語。カンマ区切りでも構いません。",
    )
    sample_fps = st.slider("1秒あたりのチェック回数", min_value=1, max_value=10, value=2, step=1)
    threshold = st.slider("類似一致しきい値", min_value=0.60, max_value=1.00, value=0.82, step=0.01)
    min_conf = st.slider("OCR信頼度の下限", min_value=0.00, max_value=1.00, value=0.35, step=0.05)
    st.caption("Cloud試作版は2回/秒から。最終運用ではPC性能に応じて5回/秒などへ上げます。")

uploaded = st.file_uploader(
    "動画をここにドラッグ＆ドロップ",
    type=["mp4", "mov", "avi", "mkv", "ts", "m2ts", "mts", "wmv"],
    help="まずは1～3分程度の公開済み／テスト動画で確認してください。",
)

if uploaded is not None:
    size_mb = uploaded.size / (1024 * 1024)
    st.success(f"選択済み: {uploaded.name}（{size_mb:.1f} MB）")
    if size_mb > 300:
        st.warning("300MBを超える動画はCommunity Cloudのメモリ制限に触れやすいため、短い動画で先に検証してください。")

st.markdown("##### 実行")
st.caption("初回はOCRモデルのダウンロードが入るため、2回目以降より準備に時間がかかります。")

run_clicked = st.button("▶ チェック開始", type="primary", use_container_width=True, disabled=uploaded is None)

if run_clicked:
    try:
        core = get_core_module()
    except Exception as exc:
        st.error("動画処理モジュールの読み込みに失敗しました。")
        st.exception(exc)
        st.stop()

    targets = core.split_targets(targets_text)
    if not targets:
        st.error("探すワードを1つ以上入力してください。")
        st.stop()

    job_dir = make_job_dir(uploaded.name)
    out_dir = job_dir / "result"
    input_dir = job_dir / "input"
    input_dir.mkdir()
    saved_video = input_dir / Path(uploaded.name).name

    copy_bar = st.progress(0.0, text="動画を作業領域へコピーしています…")
    write_upload(uploaded, saved_video)
    copy_bar.progress(1.0, text="動画の準備完了")

    progress_bar = st.progress(0.0, text="OCRエンジンを準備しています…")
    status = st.empty()
    log_box = st.empty()
    logs: list[str] = []

    def log_callback(msg: str):
        logs.append(msg)
        if len(logs) > 12:
            del logs[:-12]
        log_box.code("\n".join(logs), language=None)

    def progress_callback(pct: float, checked: int, total: int, hits: int):
        progress_bar.progress(
            min(1.0, max(0.0, pct / 100.0)),
            text=f"{pct:.1f}%  ｜ OCR {checked}/{total if total else '?'}  ｜ 検出 {hits}件",
        )
        status.info(f"処理中… {pct:.1f}%　検出イベント: {hits}件")

    try:
        ocr_backend = get_ocr_backend()
        result = core.run(
            video_path=str(saved_video),
            targets=targets,
            sample_fps=float(sample_fps),
            threshold=float(threshold),
            out_dir=str(out_dir),
            lang="japan",
            min_confidence=float(min_conf),
            merge_gap_sec=1.0,
            log_callback=log_callback,
            progress_callback=progress_callback,
            ocr_backend=ocr_backend,
        )
        progress_bar.progress(1.0, text="完了")
        status.success("チェックが完了しました。")
        st.session_state["last_result"] = result
        st.session_state["last_out_dir"] = str(out_dir)
        show_results(result, out_dir)
    except Exception as exc:
        status.error("処理に失敗しました。下のエラーをコピーして共有してください。")
        st.exception(exc)

elif "last_result" in st.session_state and "last_out_dir" in st.session_state:
    p = Path(st.session_state["last_out_dir"])
    if p.exists():
        show_results(st.session_state["last_result"], p)
