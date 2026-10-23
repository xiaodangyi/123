"""图片检测命令行入口；每次保存独立基线，不覆盖原图。"""

import argparse
from datetime import datetime
from importlib import metadata
import json
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4
from zoneinfo import ZoneInfo

from .config import ROOT, load_categories, local_weights
from .detector import BaselineDetector, TrainedDetector, file_sha256, read_image
from .render import find_font, render_result


def probability(value: str) -> float:
    """阈值端点不接收，避免无意抑制全部结果或保留无效分数。"""
    number = float(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError('阈值必须在 0 与 1 之间。')
    return number


def image_size(value: str) -> int:
    """明确约束尺寸，避免库静默调整后记录与运行参数不同。"""
    number = int(value)
    if not 32 <= number <= 2048 or number % 32:
        raise argparse.ArgumentTypeError('图片输入尺寸须为 32～2048 之间的 32 倍数。')
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description='使用本地 YOLOv8 权重检测图片，保存中文标注和八类状态。', add_help=False)
    parser._optionals.title = '可选参数'
    parser.add_argument('-h', '--help', action='help', help='显示帮助并退出')
    parser.add_argument('--image', required=True, help='输入图片路径，支持中文和空格')
    parser.add_argument('--mode', choices=['baseline', 'finetuned'], default='baseline', help='COCO 基线或八类微调权重，默认 baseline')
    parser.add_argument('--weights', help='已有 .pt 权重路径；微调模式必须显式指定')
    parser.add_argument('--output', help='输出根目录，每次创建独立子目录；默认按模式存入 runs')
    parser.add_argument('--conf', type=probability, default=0.25, help='检测分数阈值，默认 0.25')
    parser.add_argument('--iou', type=probability, default=0.7, help='跨类别重复框抑制阈值，默认 0.7')
    parser.add_argument('--imgsz', type=image_size, default=640, help='模型输入尺寸，默认 640')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'], default='auto', help='计算设备，默认自动选择')
    parser.add_argument('--font', help='中文字体文件；Windows 默认查找微软雅黑')
    args = parser.parse_args()
    run_dir = None
    try:
        source = Path(args.image).expanduser().resolve()
        started = perf_counter()
        image = read_image(source)
        decode_ms = (perf_counter() - started) * 1000
        if args.mode == 'finetuned' and not args.weights:
            raise ValueError('微调模式必须通过 --weights 显式选择八类权重。')
        weights = local_weights(args.weights)
        categories = load_categories()
        font_path = find_font(args.font)
        print('正在读取已有权重并执行检测；此命令不训练、不下载模型。', flush=True)
        detector_type = TrainedDetector if args.mode == 'finetuned' else BaselineDetector
        detector = detector_type(weights, categories, args.device)
        report = detector.predict(image, args.conf, args.iou, args.imgsz)
        timestamp = datetime.now(ZoneInfo('Asia/Shanghai'))
        report['created_at'] = timestamp.isoformat()
        report['input'] = {'file_name': source.name, 'sha256': file_sha256(source)}
        report['versions'] = {name: metadata.version(name) for name in ('torch', 'torchvision', 'ultralytics', 'opencv-python', 'Pillow')}
        report['category_config'] = {
            'sha256': file_sha256(ROOT / 'configs/classes.yaml'),
            'mapping': [{'id': item.id, 'key': item.key, 'name': item.name, 'coco_id': item.coco_id} for item in categories],
        }
        report['timings_ms']['image_decode'] = round(decode_ms, 3)
        visualization = render_result(image, report, font_path)
        output_root = Path(args.output).expanduser().resolve() if args.output else ROOT / 'runs' / args.mode
        run_dir = output_root / f"{timestamp:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        report['artifacts'] = {'visualization': 'annotated.png', 'result': 'result.json'}
        visualization.save(run_dir / 'annotated.png')
        (run_dir / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        for item in report['summary']:
            value = item['count'] if item['count'] is not None else item['status_text']
            print(f"{item['category_name']}：{value}")
        print(report['message'])
        print(f'结果目录：{run_dir}')
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f'检测失败：{error}', file=sys.stderr)
        if run_dir is not None:
            print(f'输出未完整保存，请勿作为有效基线：{run_dir}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
