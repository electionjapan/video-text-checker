# 動画テキストチェッカー / VideoTextChecker

動画を一定間隔でOCRし、指定した日本語ワードの表示箇所を検出するStreamlitアプリです。

## Community Cloudで試す

1. このフォルダの**中身**をGitHubリポジトリのルートへアップロードします。
2. Streamlit Community CloudでGitHubアカウントを接続します。
3. `Create app` → `Yup, I have an app` を選択します。
4. Repository: 作成したリポジトリ
5. Branch: `main`
6. Main file path: `streamlit_app.py`
7. Advanced settingsで **Python 3.12** を選択します。
8. Deployします。

## 最初のテスト

外部クラウドに動画をアップロードするため、最初は公開済み／テスト用の1～3分程度のMP4を使用してください。
放送前素材、社外秘素材、個人情報を含む素材ではCommunity Cloud版を使用しないでください。

## 主な機能

- MP4/MOV/AVI/MKV/TS/M2TS/MTS/WMVアップロード
- 複数検索ワード
- 完全一致＋類似一致
- 分割されたOCR枠の隣接結合
- 同じテロップの連続検出を1イベントへ統合
- 検出タイムコードと該当フレーム表示
- CSV / HTML / 結果ZIPダウンロード

## 現在の依存関係

- Streamlit 1.47.0
- PaddlePaddle 3.0.0 (CPU)
- PaddleOCR 3.1.0
- OpenCV headless

## 注意

Community CloudはCPU・メモリに上限があります。長尺動画や高頻度OCRの本番運用は、動作確認後に社内LAN上のStreamlitサーバーへ移す想定です。


## Streamlit Community Cloud deployment

- Upload limit: **750 MB**
- Recommended Python version: **3.12**
- OCR stack: PaddleOCR 3.7.0 / PaddlePaddle 3.3.1
- Main file: `streamlit_app.py`

If an existing Community Cloud app was created with a different Python version,
delete that app and redeploy it with Python 3.12 from Advanced settings.
