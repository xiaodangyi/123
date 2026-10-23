"""检查首轮训练条件，并仅在明确的可见终端中启动 YOLOv8 微调。"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import traceback

import torch
import yaml

from .audit_data import require_training_ready, validate_export
from .config import ROOT, local_weights
from .data import read_json, save_json, sha256

TRAIN_CONFIG = ROOT / 'configs/training.yaml'


def load_training_config():
    config = yaml.safe_load(TRAIN_CONFIG.read_text(encoding='utf-8'))
    required = {'schema_version', 'dataset', 'model', 'epochs', 'image_size', 'batch_size',
                'workers', 'device', 'amp', 'cache', 'seed', 'save_period', 'output_dir'}
    if not isinstance(config, dict) or set(config) != required or config['schema_version'] != 1:
        raise ValueError('训练配置字段或版本不符合预期。')
    expected = {'model': 'yolov8n', 'epochs': 3, 'image_size': 640, 'batch_size': 4,
                'workers': 0, 'device': 0, 'amp': False, 'cache': False,
                'seed': 42, 'save_period': 1, 'output_dir': 'runs/train'}
    if any(type(config[key]) is not type(value) or config[key] != value
           for key, value in expected.items()):
        raise ValueError('首轮参数必须使用已核对的 YOLOv8n、3 轮、640、batch 4、GPU 0 配置。')
    if not isinstance(config['dataset'], str):
        raise ValueError('训练数据目录必须是文字路径。')
    dataset = (ROOT / config['dataset']).resolve()
    if dataset == (ROOT / 'data').resolve() or not dataset.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('训练数据版本必须位于项目 data 目录内。')
    return config


def environment_preflight():
    """在数据尚未就绪时也可单独验证环境；不读取训练图、不启动训练。"""
    config = load_training_config()
    local_config = read_json(ROOT / '.local/config.json')
    expected_python = (ROOT / local_config['ai_cuda_root'] / 'python.exe').resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError('当前解释器不是本机配置的 ai-cuda Python。')
    weights = local_weights(None)
    dataset = (ROOT / config['dataset']).resolve()
    if not torch.cuda.is_available():
        raise RuntimeError('没有可用 CUDA GPU；为避免悄悄改用 CPU，本次停止。')
    if torch.cuda.device_count() <= config['device']:
        raise RuntimeError('配置的 CUDA 设备编号不存在。')
    device = torch.cuda.get_device_properties(config['device'])
    if device.total_memory < 6 * 1024**3:
        raise RuntimeError('GPU 显存低于首轮配置要求，请先调整配置并重新检查。')
    from importlib.metadata import version
    if version('ultralytics') != '8.3.203':
        raise RuntimeError('Ultralytics 版本与已验证环境不一致。')
    # 实际执行一次 GPU 张量运算，避免只检测到驱动却不能执行内核。
    tensor = torch.ones((16, 16), device=f'cuda:{config["device"]}')
    if (tensor @ tensor).sum().item() != 4096:
        raise RuntimeError('CUDA 张量运算结果不符。')
    return config, dataset, weights, device


def preflight():
    require_training_ready(read_json(ROOT / 'configs/data_review.json'))
    result = environment_preflight()
    validate_export(result[1])
    return result


def print_preflight(result):
    config, dataset, weights, device = result
    print('首轮训练条件检查通过。', flush=True)
    print(f'解释器：{sys.executable}', flush=True)
    print(f'Ultralytics：8.3.203；权重：{weights}（SHA-256 {sha256(weights)}）', flush=True)
    print(f'数据集：{dataset}（数据清单 SHA-256 {sha256(dataset / "manifest.json")}）', flush=True)
    print(f'显卡：{device.name}，显存 {device.total_memory / 1024**3:.1f} GiB', flush=True)
    print(f'参数：YOLOv8n，{config["epochs"]} 轮，输入 {config["image_size"]}，批量 {config["batch_size"]}，工作进程 {config["workers"]}，AMP 关闭，数据缓存关闭。', flush=True)
    print('每轮结束后会输出训练损失与验证指标。loss 越低通常表示训练样本拟合得更好；精确率、召回率和 mAP 在验证图上估计识别质量。三轮只检查训练链路，不代表最终效果。', flush=True)
    print('独立测试集留待模型确定后再评估，避免反复看测试成绩来调参。', flush=True)


def run_training(result):
    if os.environ.get('DESKVISION_VISIBLE_TRAINING') != '1':
        raise RuntimeError('训练只能由打开可见训练终端的脚本启动。')
    config, dataset, weights, device = result
    from ultralytics import YOLO

    run_name = 'desk8-pilot-' + datetime.now().strftime('%Y%m%d-%H%M%S')
    output = (ROOT / config['output_dir']).resolve()
    metadata_dir = ROOT / 'runs/train-metadata'
    metadata_path = metadata_dir / (run_name + '.json')
    metadata_dir.mkdir(parents=True, exist_ok=True)
    state = {
        'status': 'running',
        'run_name': run_name,
        'started_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'training_config': config,
        'training_config_sha256': sha256(TRAIN_CONFIG),
        'dataset_manifest_sha256': sha256(dataset / 'manifest.json'),
        'dataset_yaml_sha256': sha256(dataset / 'data.yaml'),
        'weights_sha256': sha256(weights),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'gpu': device.name,
        'epochs': [],
    }
    save_json(metadata_path, state)
    print(f'运行记录：{metadata_path}', flush=True)
    print('训练开始。Epoch 是完整遍历训练图像的一轮；GPU 进度行会持续刷新。', flush=True)
    def record_epoch(trainer):
        train_loss = [round(float(value), 5) for value in trainer.tloss.detach().cpu().tolist()]
        metrics = {}
        for key, value in trainer.metrics.items():
            try:
                metrics[str(key)] = round(float(value), 5)
            except (TypeError, ValueError):
                continue
        row = {'epoch': trainer.epoch + 1, 'train_loss': train_loss, 'validation_metrics': metrics}
        state['epochs'].append(row)
        save_json(metadata_path, state)
        print(f'本轮汇总：第 {row["epoch"]}/{config["epochs"]} 轮，loss={train_loss}，验证指标={metrics}', flush=True)

    try:
        model = YOLO(str(weights))
        model.add_callback('on_fit_epoch_end', record_epoch)
        model.train(
            data=str(dataset / 'data.yaml'), epochs=config['epochs'], imgsz=config['image_size'],
            batch=config['batch_size'], workers=config['workers'], device=config['device'],
            amp=config['amp'], cache=config['cache'], seed=config['seed'],
            save_period=config['save_period'], project=str(output), name=run_name,
            exist_ok=False, plots=True, verbose=True,
        )
    except (Exception, KeyboardInterrupt) as error:
        state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        state['finished_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
        state['error'] = f'{type(error).__name__}: {error}'
        state['traceback_tail'] = traceback.format_exc().splitlines()[-12:]
        save_json(metadata_path, state)
        raise
    run_dir = Path(model.trainer.save_dir).resolve()
    state['status'] = 'completed'
    state['finished_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
    state['run_dir'] = str(run_dir)
    state['best_weights'] = str((run_dir / 'weights' / 'best.pt').resolve())
    state['last_weights'] = str((run_dir / 'weights' / 'last.pt').resolve())
    save_json(metadata_path, state)
    print(f'三轮试跑完成；权重和曲线位于：{run_dir}', flush=True)
    print('现在只根据训练日志和验证集判断是否需要改数据或参数；正式测试集尚未使用。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='通过可见训练终端开始三轮小批试跑')
    mode.add_argument('--environment-only', action='store_true', help='只检查配置、解释器、权重与 CUDA 运算，不检查数据，不训练')
    args = parser.parse_args()
    if args.environment_only:
        config, dataset, weights, device = environment_preflight()
        print(f'环境检查通过：{sys.executable}；Ultralytics 8.3.203；{device.name}，{device.total_memory / 1024**3:.1f} GiB；CUDA 矩阵运算通过。', flush=True)
        print(f'本地权重：{weights}（SHA-256 {sha256(weights)}）', flush=True)
        print('数据尚未在本命令中验收，没有启动训练。', flush=True)
        return
    result = preflight()
    print_preflight(result)
    if args.run:
        run_training(result)
    else:
        print('这里只完成检查，没有启动训练。使用第四阶段可见终端脚本才会开始训练。', flush=True)


if __name__ == '__main__':
    main()
