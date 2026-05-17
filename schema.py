from pydantic import BaseModel

class DetectedObject(BaseModel):
    label: str                          # name of the detected object
    box_2d: tuple[int, int, int, int]   # [y_min, x_min, y_max, x_max] on a 0-1000 scale

class DetectionResult(BaseModel):
    description: str                       # overall description of the image
    detected_objects: list[DetectedObject] # list of all detected objects
