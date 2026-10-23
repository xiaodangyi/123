"""使用已有 YOLOv8 COCO 权重进行检测，输出原图坐标和明确的类别状态。"""

from hashlib import sha256
import math
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .config import Category, ROOT


def file_sha256(path: Path) -> str:
    """分块计算文件指纹，便于核对基线输入与权重是否一致。"""
    digest = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(path: Path) -> np.ndarray:
    """通过字节解码支持中文路径，明确拒绝损坏或空文件。"""
    if not path.is_file():
        raise FileNotFoundError(f'输入图片不存在：{path.name}')
    content = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(content, cv2.IMREAD_COLOR) if content.size else None
    if image is None:
        raise ValueError(f'无法解码图片：{path.name}，请检查文件格式或内容。')
    return image


def make_report(
    categories: list[Category], rows: list[list[float]], width: int, height: int,
) -> dict:
    """转换原图像素框；未训练类用空值表示，不能伪造零计数。"""
    mapping = {item.coco_id: item for item in categories if item.coco_id is not None}
    counts = {item.id: 0 for item in categories}
    detections = []
    for x1, y1, x2, y2, confidence, source_id in rows:
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2, confidence, source_id)):
            raise ValueError('模型返回了非有限坐标或分数。')
        if source_id != int(source_id) or int(source_id) not in mapping:
            raise ValueError('模型返回了未请求的类别，不能按项目类别直接解释。')
        if not 0 <= confidence <= 1:
            raise ValueError('模型返回了无效置信度。')
        box = [max(0.0, min(float(x1), width)), max(0.0, min(float(y1), height)),
               max(0.0, min(float(x2), width)), max(0.0, min(float(y2), height))]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('模型返回了面积为零或顺序错误的检测框。')
        category = mapping[int(source_id)]
        counts[category.id] += 1
        detections.append({
            'instance_id': len(detections) + 1, 'category_id': category.id,
            'category_key': category.key, 'category_name': category.name,
            'source_class_id': int(source_id), 'detection_confidence': float(confidence),
            'bbox_xyxy': box,
        })
    summary = []
    for category in categories:
        supported = category.coco_id is not None
        summary.append({
            'category_id': category.id, 'category_name': category.name,
            'status': 'supported' if supported else 'not_trained',
            'status_text': '基线支持' if supported else '尚未训练',
            'count': counts[category.id] if supported else None,
        })
    return {
        'schema_version': 1, 'status': 'ok',
        'coordinate_system': '原图像素坐标，左上角为原点，xyxy',
        'image_size': {'width': width, 'height': height},
        'detections': detections, 'summary': summary,
        'total_supported_detections': len(detections),
        'message': '未检出基线支持的目标；不代表图中没有其他物品。' if not detections
                   else '结果只覆盖预训练支持的三类，需人工核对漏检和误检。',
    }


class BaselineDetector:
    """只承载预训练基线；拒绝把自定义八类权重当作 COCO 编号解释。"""

    def __init__(self, weights: Path, categories: list[Category], device: str = 'auto'):
        os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'environment/ultralytics-settings'))
        os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'environment/matplotlib-cache'))
        os.environ['YOLO_OFFLINE'] = 'true'
        import torch
        from ultralytics import YOLO, settings

        settings.update({'sync': False, 'wandb': False, 'mlflow': False, 'comet': False})
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda:0' and not torch.cuda.is_available():
            raise RuntimeError('已指定 GPU，但 CUDA 不可用；请先检查环境或选择 --device cpu。')
        started = perf_counter()
        self.model = YOLO(str(weights), task='detect')
        for category in categories:
            if category.coco_id is not None and self.model.names.get(category.coco_id) != category.key:
                raise ValueError('权重类别不是预期的 COCO 字典；此阶段仅支持 YOLOv8 COCO 检测权重。')
        self.model.to(device)
        self.load_ms = (perf_counter() - started) * 1000
        self.categories = categories
        self.device = device
        self.weights = weights
        self.weights_hash = file_sha256(weights)
        self.gpu_name = torch.cuda.get_device_name(0) if device == 'cuda:0' else None

    def predict(self, image: np.ndarray, confidence: float, iou: float, image_size: int) -> dict:
        """筛选支持类别，再由 YOLO 完成缩放、推理、抑制重复框与坐标还原。"""
        started = perf_counter()
        result = self.model.predict(
            source=image, device=self.device, imgsz=image_size, conf=confidence, iou=iou,
            classes=[item.coco_id for item in self.categories if item.coco_id is not None],
            agnostic_nms=True, max_det=300, half=False, augment=False, save=False, verbose=False,
        )[0]
        rows = result.boxes.data.detach().cpu().tolist() if result.boxes is not None else []
        elapsed = (perf_counter() - started) * 1000
        report = make_report(self.categories, rows, image.shape[1], image.shape[0])
        report['model'] = {
            'route': 'yolov8_coco_baseline', 'weights_name': self.weights.name,
            'weights_sha256': self.weights_hash, 'device': self.device, 'gpu_name': self.gpu_name,
        }
        report['parameters'] = {
            'confidence_threshold': confidence, 'nms_iou_threshold': iou,
            'image_size': image_size, 'max_detections': 300, 'half_precision': False,
            'class_agnostic_nms': True,
        }
        report['counting_note'] = '八类按互斥物品处理，跨类别抑制高度重叠框；严重重叠的不同实物仍可能漏检。'
        report['timings_ms'] = {
            'model_load_and_transfer': round(self.load_ms, 3),
            'prediction_wall': round(elapsed, 3),
            **{f'model_{key}': round(value, 3) for key, value in result.speed.items()},
        }
        report['timing_note'] = '单张调用耗时，首张包含推理器初始化与预热；不能视为稳定吞吐率。'
        return report
