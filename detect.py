import argparse
import asyncio
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from chain import chain


async def run(image_path: str, query: str, output_path: str) -> None:
    image_bytes = Path(image_path).read_bytes()
    result = await chain.ainvoke({"image": image_bytes, "query": query})

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=max(16, h // 40))
    except OSError:
        font = ImageFont.load_default()

    for obj in result.detected_objects:
        y_min, x_min, y_max, x_max = obj.box_2d
        x0, y0 = x_min / 1000 * w, y_min / 1000 * h
        x1, y1 = x_max / 1000 * w, y_max / 1000 * h
        draw.rectangle([x0, y0, x1, y1], outline="lime", width=3)
        draw.text((x0 + 4, y0 + 2), obj.label, fill="lime", font=font)

    img.save(output_path)
    print(f"saved: {output_path}")
    for obj in result.detected_objects:
        print(f"  {obj.label}: {obj.box_2d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="入力画像のパス")
    parser.add_argument("--query", default="object", help="検出対象（例: 富士山）")
    parser.add_argument("--output", default="output.jpeg", help="出力画像のパス")
    args = parser.parse_args()

    asyncio.run(run(args.image, args.query, args.output))
