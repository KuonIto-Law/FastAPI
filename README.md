# FastAPI × Gemini 物体検出 API

画像をアップロードすると、Google Gemini 2.5 Flash が物体を検出し、バウンディングボックス付きの画像を返す REST API です。

## デモ

入力画像に対して `/visualize` エンドポイントを呼び出すと、検出された物体にラベルと枠が描画されます。

![富士山検出デモ](MTfuji_detected.jpeg)

---

## 技術スタック

| カテゴリ | 技術 |
|---|---|
| Web フレームワーク | FastAPI |
| AI モデル | Google Gemini 2.5 Flash (multimodal) |
| LLM パイプライン | LangChain (LCEL) |
| スキーマ定義 | Pydantic v2 |
| 画像処理 | Pillow |
| 実行環境 | Python 3.11+ / uvicorn |

---

## エンドポイント

### `POST /detect`
画像を受け取り、検出結果を JSON で返します。

**Request:** `multipart/form-data`
- `file`: 画像ファイル（JPEG / PNG）
- `query`: 検出対象の文字列（例: `"人"`, `"富士山"`）

**Response:**
```json
{
  "description": "富士山を背景にした風景写真です。",
  "detected_objects": [
    {
      "label": "富士山",
      "box_2d": [120, 200, 800, 900]
    }
  ]
}
```

---

### `POST /visualize`
画像を受け取り、検出ボックスとラベルを描画した JPEG 画像を返します。

**Request:** `/detect` と同じ

**Response:** `image/jpeg`（バウンディングボックス描画済み）

---

## セットアップ

```bash
# 依存パッケージのインストール
pip install -e .

# 環境変数の設定
cp .env.example .env
# .env に GOOGLE_API_KEY を記入
```

```bash
# サーバー起動
uvicorn main:app --reload
```

起動後、`http://localhost:8000/docs` で Swagger UI から動作確認できます。

---

## CLI での実行

API サーバーを立てずに、コマンドラインから直接検出することも可能です。

```bash
python detect.py MTfuji.jpeg --query "富士山" --output result.jpeg
```

---

## 設計のポイント

- **依存性注入 (`Depends`)**: ファイルバリデーションを `get_image_file` 関数に切り出し、エンドポイントの責務を分離
- **非同期処理 (`async/await`)**: I/O バウンドな処理（ファイル読み込み・LLM 呼び出し）を非同期で実装
- **StreamingResponse**: 画像レスポンスをディスクに保存せずメモリ上から直接返すことでオーバーヘッドを削減
- **LangChain LCEL パイプライン**: 前処理 → プロンプト → LLM → 構造化出力を `|` 演算子でチェーンし、可読性の高いパイプラインを構成
- **Pydantic による構造化出力**: `with_structured_output(DetectionResult)` で LLM の出力を型安全なオブジェクトに変換
- **APIRouter**: 機能単位でルートを分割し、将来の機能追加に対応しやすい構造

---

## ディレクトリ構成

```
.
├── main.py        # FastAPI アプリ・エンドポイント定義
├── chain.py       # LangChain パイプライン（Gemini 呼び出し）
├── schema.py      # Pydantic スキーマ（DetectionResult）
├── detect.py      # CLI スクリプト
└── pyproject.toml # 依存パッケージ定義
```
