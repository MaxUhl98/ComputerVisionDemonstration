import utils.helpers as h
from logging import Logger
from demo_configuration import DemonstrationConfig


def test_get_logger():
    logger = h.get_logger('test_logger', 'test_files/fake_logs')
    assert logger.name == 'test_logger'
    assert logger.__class__ == Logger
    assert logger.handlers[1].baseFilename.split(r'test_files')[1] == r'\fake_logs\test_logger.log'


def test_get_label_mapping_dictionary():
    assert h.get_label_mapping_dictionary(DemonstrationConfig.train_data_paths[0]) == DemonstrationConfig.class_mappings
