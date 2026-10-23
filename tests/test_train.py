"""验证准备检查不误启动训练，且配置不能静默改变试跑范围。"""

import copy
import unittest
from unittest.mock import patch

import yaml

from deskvision.train import TRAIN_CONFIG, load_training_config, preflight, run_training


class TrainingEntryTests(unittest.TestCase):
    def test_pilot_config_rejects_invalid_or_unreviewed_parameters(self):
        original = yaml.safe_load(TRAIN_CONFIG.read_text(encoding='utf-8'))
        load_training_config()
        for key, value in (('batch_size', 0), ('workers', -1), ('amp', 'false'),
                           ('epochs', 30), ('device', False), ('output_dir', '../outside'),
                           ('dataset', '../outside')):
            changed = copy.deepcopy(original)
            changed[key] = value
            with self.subTest(key=key), patch('deskvision.train.yaml.safe_load', return_value=changed):
                with self.assertRaises(ValueError):
                    load_training_config()

    def test_unready_data_stops_before_environment_or_model_loading(self):
        with patch('deskvision.train.read_json', return_value={'training_ready': False}), \
                patch('deskvision.train.environment_preflight') as environment:
            with self.assertRaisesRegex(ValueError, '尚未获准训练'):
                preflight()
            environment.assert_not_called()

    def test_training_requires_visible_terminal_marker(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, '可见训练终端'):
                run_training(None)


if __name__ == '__main__':
    unittest.main()
