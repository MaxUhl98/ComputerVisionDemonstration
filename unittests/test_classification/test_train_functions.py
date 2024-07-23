import torch.nn
import classification.train_functions as t
from unittests.test_classification.mock_model import MockModel
from unittests.test_classification.mock_config import MockConfig
from classification.data.image_data_class import ImageDataset
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
import os
from logging import Logger
from unittests.test_classification.mock_optimizer import MockOptimizer


class TestTrainFunctions:

    def set_up(self):
        if os.getcwd().rsplit('\\',1)[1] == 'test_classification':
            os.chdir('../..')
        torch.set_default_device('cpu')
        self.cfg = MockConfig()
        self.model = MockModel(2)
        paths = sum([list(Path(dir_path).glob('**/*/*.jpg')) for dir_path in self.cfg.train_data_paths], [])
        data = pd.DataFrame(
            {'path': paths, 'target': [self.cfg.class_mappings.get(data_path.parent.name) for data_path in paths]})
        self.data = ImageDataset(data)
        self.loader = DataLoader(self.data, batch_size=1)
        self.loss_fn = nn.L1Loss()
        self.optimizer = MockOptimizer()
        self.device = 'cpu'
        self.logger = Logger('test')

    @staticmethod
    def test_average_meter():
        meter = t.AverageMeter()
        meter.update(10)
        assert meter.val == 10
        assert meter.count == 1
        assert meter.avg == 10
        assert meter.sum == 10
        meter.update(20)
        assert meter.val == 20
        assert meter.count == 2
        assert meter.avg == 15
        assert meter.sum == 30
        meter.reset()
        assert meter.val == 0
        assert meter.count == 0
        assert meter.avg == 0
        assert meter.sum == 0

    def test_train_step(self):
        self.set_up()
        loss, acc = t.train_step(self.model, self.loader, self.loss_fn, self.optimizer, self.device,
                                 lr_scheduling=False,
                                 logger=self.logger, cfg=self.cfg)
        assert loss == .5
        assert acc == .5

    def test_test_step(self):
        self.set_up()
        loss, acc = t.test_step(self.model, self.loader, self.loss_fn, self.device)
        assert loss == .5
        assert acc == .5

    def test_train(self):
        self.set_up()
        results = t.train(self.model, self.loader, self.loader, loss_fn=self.loss_fn,optimizer= self.optimizer, device=self.device,
                          lr_scheduling=False,
                          logger=self.logger, cfg=self.cfg, epochs=1, save_best=False)
        assert results["train_loss"][0] == .5
        assert results["val_loss"][0] == .5
        assert results["train_acc"][0] == .5
        assert results["val_acc"][0] == .5

    def test_kfold_train(self):
        self.set_up()
        results = t.k_fold_train([self.model for _ in range(self.cfg.num_folds)], paths_to_data=self.cfg.train_data_paths,
                                 optimizers=[self.optimizer for _ in range(self.cfg.num_folds)],loss_fn=self.loss_fn,
                                 device=self.device,logger=self.logger,cfg=self.cfg)
        fold_0_results = results['fold_0']
        fold_1_results = results['fold_1']
        assert fold_0_results["train_loss"][0] == .5
        assert fold_0_results["val_loss"][0] == .5
        assert fold_0_results["train_acc"][0] == .5
        assert fold_0_results["val_acc"][0] == .5
        assert fold_1_results["train_loss"][0] == .5
        assert fold_1_results["val_loss"][0] == .5
        assert fold_1_results["train_acc"][0] == .5
        assert fold_1_results["val_acc"][0] == .5

