# このファイルは、FastAPIを使用して画像の物体検出と可視化のAPIエンドポイントを提供するためのものです。
# つまり、ユーザーが画像を持って来て、それを受け取ったり、表示したりする窓口のような存在

import io
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont

from chain import chain
from schema import DetectionResult

# 門番、ちゃんと画像がアップロードされているかを検証するための関数、jpegとpngのみ許可する
# Depends()を使った依存性注入の書き方
# ValueErrorではなくHTTPExceptionを使うことで、正しい400エラーを返せる
async def get_image_file(file: UploadFile = File(...)) -> UploadFile:
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    return file

# 「窓口の仕切り」
# appに直接エンドポイントを書く代わりに、routerという仕切りを通ってそこにまとめている。
# コンビニ全体(APP)の中に、特定の担当コーナー(router)を作るイメージ
# 今は1ファイルだけですが、ファイルが増えたときに /detect 系は router_a、別の機能は router_b のように整理できる
router = APIRouter()

# エンドポイント本体
# @はデコレーターで、「この下の関数に設定を付け加える」という記号
# 「/detect というURLにPOSTリクエストが来たら、この下の関数を呼び出してね」とFastAPIに登録しています。
# response_model=DetectionResult は「返すJSONの形はこの型ですよ」とSwagger UIに伝えるための設定で、/docs のレスポンス例の表示に使われます。
@router.post("/detect", response_model=DetectionResult)
async def detect_objects( #asyncは「非同期で動く関数」という宣言
    file: Annotated[UploadFile, Depends(get_image_file)],
    query: str = Form(default="object"), 
) -> Any:
    image_bytes = await file.read() # awaitは「この処理が終わるまで待つ」という意味
    return await chain.ainvoke({"image": image_bytes, "query": query}) #chain.pyで定義下パイプラインを実行して、結果をそのまま返している 
# chain.ainvoke(...) — chainを非同期で実行する。a が「async（非同期）」の意味
#{"image": image_bytes, "query": query} — chainへの入力。画像バイナリと検出対象の文字列を渡す
# ここがGeminiに外注しているコード
#return — chainが返してきた DetectionResult オブジェクトをそのままレスポンスとして返す

@router.post("/visualize") #こっちの関数にはresponse_modelがなく、それは返すのがJSONではなく画像ファイルだから
async def visualize_objects(
    file: Annotated[UploadFile, Depends(get_image_file)],
    query: str = Form(default="object"),
) -> StreamingResponse: #「この関数は画像データをストリームで返す」という意味だが、ストリームで返すとは、画像を描画してあとにメモリ上のbufを直接返すということで、ファイルに保存してから返さないから、手間がかからない。ファイルに保存してから返す場合は、「画像を描画 → result.jpeg としてディスクに保存 → ファイルを読み込んで返す → ファイルを削除」という処理になるが、今回は「画像を描画 → メモリ上のbufを直接返す」という処理になる。
    image_bytes = await file.read()
    result = await chain.ainvoke({"image": image_bytes, "query": query})

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB") #受け取った画像バイナリをPILの画像オブジェクトに変換する。io.BytesIO はバイナリデータをファイルのように扱えるようにするラッパー。.convert("RGB") はPNGなど透過情報（アルファチャンネル）を持つ画像をJPEGで保存できる形式に変換するため。
    w, h = img.size #画像の幅（w）と高さ（h）をピクセル単位で取得する。後でGeminiが返した0〜1000スケールの座標を実際のピクセル座標に変換するために使う。
    draw = ImageDraw.Draw(img) #画像の上に線や文字を描くための「ペン」を用意する。


# ラベル文字のフォントを読み込む。arial.ttf が見つからない場合はデフォルトフォントで代替する。max(16, h // 40) は画像サイズに合わせてフォントサイズを自動調整している（最小16px）。
    try:
        font = ImageFont.truetype("arial.ttf", size=max(16, h // 40))
    except OSError:
        font = ImageFont.load_default()

# ここの説明をもう少しする
# 複数の物体が検出された場合に備えて、forループで一つずつ描画していく。Geminiが返したbox_2dは0〜1000のスケールなので、実際のピクセル座標に変換するために、x_min / 1000 * w のような計算をしている。描画は、draw.rectangleで検出ボックスを描き、draw.textでラベルを描いている。
    for obj in result.detected_objects:
        y_min, x_min, y_max, x_max = obj.box_2d
        x0, y0 = x_min / 1000 * w, y_min / 1000 * h
        x1, y1 = x_max / 1000 * w, y_max / 1000 * h
        draw.rectangle([x0, y0, x1, y1], outline="lime", width=3) #変換した座標に緑色（lime）の矩形を描く。width=3 は線の太さ。
        draw.text((x0 + 4, y0 + 2), obj.label, fill="lime", font=font) #矩形の左上にラベル名を緑色で描く。+4, +2 は枠線の内側に少しずらすための余白。

# 描画済みの画像をJPEGとしてメモリ上のバッファに保存する。buf.seek(0) はバッファの読み取り位置を先頭に戻す操作（これがないと空のデータが返ってしまう）。
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

# バッファに保存した画像をHTTPレスポンスとして返す。media_type="image/jpeg" を指定することでブラウザが「これは画像だ」と認識できる。    
    return StreamingResponse(buf, media_type="image/jpeg")

# コーナーをコンビニの店舗に登録するイメージ、router = APIRouter()で作った仕切り（router）を、本体のアプリ（app）に登録している。これで初めて /detect や /visualize が外からアクセスできるようになる。
app = FastAPI()
app.include_router(router)
