import classification.train_model as tm
from tests.test_classification.mock_config import MockConfig
from tests.test_classification.mock_model import MockModel
from classification.data.image_data_class import ImageDataset
from classification.models.ViT import CustomizedViT


def test_get_models_and_logger():
    cfg = MockConfig()
    models, logger = tm.get_models_and_logger(cfg)

    assert len(models) == cfg.num_folds
    assert logger.name == cfg.model_name + '_train'
    assert models[0].__class__ == CustomizedViT


