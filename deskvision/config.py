"""读取类别与本地模型路径，防止混淆项目编号和预训练编号。"""

from dataclasses import dataclass
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Category:
    """项目类别；coco_id 为空表示预训练基线不支持。"""

    id: int
    key: str
    name: str
    coco_id: int | None


def load_categories(path: Path = ROOT / 'configs/classes.yaml') -> list[Category]:
    """在加载模型前检查类别字典，不自动修复错误编号。"""
    config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict) or config.get('schema_version') != 1:
        raise ValueError('类别配置版本无效。')
    categories = [Category(**item) for item in config['classes']]
    if not categories or [item.id for item in categories] != list(range(len(categories))):
        raise ValueError('项目类别编号必须从 0 连续排列。')
    if len({item.key for item in categories}) != len(categories):
        raise ValueError('类别标识不能重复。')
    mapped = [item.coco_id for item in categories if item.coco_id is not None]
    if any(type(value) is not int or value < 0 for value in mapped) or len(set(mapped)) != len(mapped):
        raise ValueError('预训练类别编号必须是互不重复的非负整数。')
    return categories


def local_weights(override: str | None) -> Path:
    """只接受已有权重文件，禁止把模型名称当作自动下载请求。"""
    if override is None:
        config_path = ROOT / '.local/config.json'
        if not config_path.is_file():
            raise FileNotFoundError('请配置 .local/config.json 的 yolo_weights，或传入 --weights。')
        override = json.loads(config_path.read_text(encoding='utf-8-sig')).get('yolo_weights')
    if not isinstance(override, str) or not override.strip():
        raise ValueError('缺少有效的 YOLO 权重路径。')
    path = Path(override).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if path.suffix.lower() != '.pt' or not path.is_file():
        raise FileNotFoundError('未找到本地 .pt 权重；本工具不会自动下载模型。')
    return path
