# Local vision models

`face_detection_yunet_2023mar.onnx` is the lightweight YuNet face detector
from the official [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
It is used only by `smart-layout`, is loaded locally, and is never downloaded at
runtime. The model is distributed by OpenCV Zoo under the MIT license.

Set `SMART_LAYOUT_YUNET_MODEL` to use a different compatible local ONNX file.
