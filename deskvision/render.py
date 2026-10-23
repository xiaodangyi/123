"""自主绘制中文检测框、计数与支持状态，不调用参考项目的可视化脚本。"""

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = ((24, 112, 191), (18, 130, 95), (160, 75, 30))


def find_font(override: str | None) -> Path:
    """优先显式字体，再查常见中文字体；缺失时明确提示。"""
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError('指定的中文字体文件不存在。')
        return path
    font_root = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
    candidates = [font_root / 'msyh.ttc', font_root / 'simhei.ttf',
                  Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
                  Path('/System/Library/Fonts/PingFang.ttc')]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError('未找到中文字体，请通过 --font 指定支持中文的字体文件。')


def render_result(image_bgr: np.ndarray, report: dict, font_path: Path) -> Image.Image:
    """原图保持原尺寸，计数放在右侧面板，不改变导出的框坐标。"""
    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    width, height = image.size
    font_size = max(16, min(28, round(min(width, height) / 25)))
    font = ImageFont.truetype(str(font_path), font_size)
    panel_font = ImageFont.truetype(str(font_path), 19)
    small_font = ImageFont.truetype(str(font_path), 15)
    canvas = Image.new('RGB', (width + 340, max(height, 500)), '#f2f5f8')
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for detection in report['detections']:
        box = detection['bbox_xyxy']
        color = COLORS[detection['category_id'] % len(COLORS)]
        draw.rectangle(tuple(box), outline=color, width=max(2, font_size // 8))
        label = f"{detection['instance_id']} {detection['category_name']} {detection['detection_confidence']:.2f}"
        bounds = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        x = max(0, min(round(box[0]), width - text_width - 8))
        y = max(0, round(box[1]) - text_height - 8)
        draw.rectangle((x, y, x + text_width + 8, y + text_height + 8), fill=color)
        draw.text((x + 4 - bounds[0], y + 4 - bounds[1]), label, font=font, fill='white')
    x = width + 20
    trained = report.get('model', {}).get('route') == 'yolov8_desk8_finetuned'
    draw.text((x, 20), 'YOLOv8 八类微调' if trained else 'YOLOv8 训练前基线', font=panel_font, fill='#152a3a')
    draw.text((x, 56), '八类已参与训练；效果待验收' if trained else '仅支持前三类；计数可能有误', font=small_font, fill='#586b78')
    for index, item in enumerate(report['summary']):
        value = str(item['count']) if item['count'] is not None else '尚未训练'
        draw.text((x, 100 + index * 36), f"{item['category_name']}：{value}",
                  font=panel_font, fill='#263e50' if item['count'] is not None else '#7b8188')
    draw.text((x, 408), f"支持类别检出总数：{report['total_supported_detections']}",
              font=panel_font, fill='#152a3a')
    draw.text((x, 448), '原图区域坐标不变，请人工核对', font=small_font, fill='#586b78')
    if not report['detections']:
        draw.text((x, 476), '本图未检出支持类别', font=small_font, fill='#a24820')
    return canvas
