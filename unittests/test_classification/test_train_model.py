import classification.train_model as tm
from unittests.test_classification.mock_config import MockConfig
from classification.models.ViT import CustomizedViT
import os

def test_get_models_and_logger():
    if os.getcwd().rsplit('\\', 1)[1] == 'test_classification':
        os.chdir('../..')
    cfg = MockConfig()
    models, logger = tm.get_models_and_logger(cfg)

    assert len(models) == cfg.num_folds
    assert logger.name == cfg.model_name + '_train'
    assert models[0].__class__ == CustomizedViT


