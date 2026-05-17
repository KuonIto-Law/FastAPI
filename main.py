import io
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont

from chain import chain
from schema import DetectionResult


# Validates that the uploaded file is a supported image type (JPEG or PNG).
# Using Depends() keeps this validation separate from the endpoint logic.
async def get_image_file(file: UploadFile = File(...)) -> UploadFile:
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    return file


# Groups /detect and /visualize routes under a single router.
# This makes it easy to add new route groups as the project grows.
router = APIRouter()


@router.post("/detect", response_model=DetectionResult)
async def detect_objects(
    file: Annotated[UploadFile, Depends(get_image_file)],
    query: str = Form(default="object"),
) -> Any:
    image_bytes = await file.read()
    return await chain.ainvoke({"image": image_bytes, "query": query})


@router.post("/visualize")
async def visualize_objects(
    file: Annotated[UploadFile, Depends(get_image_file)],
    query: str = Form(default="object"),
) -> StreamingResponse:
    image_bytes = await file.read()
    result = await chain.ainvoke({"image": image_bytes, "query": query})

    # Convert raw bytes to a PIL image; force RGB so JPEG export works on PNGs with alpha.
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=max(16, h // 40))
    except OSError:
        font = ImageFont.load_default()

    for obj in result.detected_objects:
        # Gemini returns coordinates on a 0-1000 scale; convert to pixel coordinates.
        y_min, x_min, y_max, x_max = obj.box_2d
        x0, y0 = x_min / 1000 * w, y_min / 1000 * h
        x1, y1 = x_max / 1000 * w, y_max / 1000 * h
        draw.rectangle([x0, y0, x1, y1], outline="lime", width=3)
        draw.text((x0 + 4, y0 + 2), obj.label, fill="lime", font=font)

    # Write the annotated image to an in-memory buffer and return it directly,
    # avoiding a disk write/read round-trip.
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


app = FastAPI()
app.include_router(router)
