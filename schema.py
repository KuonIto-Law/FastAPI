from pydantic import BaseModel

class DetectedObject(BaseModel):
    label: str                          # 検出した物体の名前
    box_2d: tuple[int, int, int, int]   # [y_min, x_min, y_max, x_max] 0-1000スケール

class DetectionResult(BaseModel):
    description: str                       # 画像全体の説明
    detected_objects: list[DetectedObject] # 検出した物体のリスト（何個でもOK）
