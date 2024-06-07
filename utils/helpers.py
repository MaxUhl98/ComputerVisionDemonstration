import logging
import os


def get_logger(name: str, base_filepath: str = 'logs/model_experiments') -> logging.Logger:
    """Creates a logging.Logger object that writes a logfile named name.log into the folder at base_filepath
    (throws an error if the folder does not exist)

    :param name: Name of the logger and the logging file
    :param base_filepath: Path to the folder in which the logging file will get saved
    :return: Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(f'{base_filepath}/{name}.log', encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def get_label_mapping_dictionary(train_dir_path: os.PathLike) -> dict[str, int]:
    """Takes the folder names inside the directory at train_dir_path as labels and their pythonic counting order as
    numeric encoding for the label. Returns {dir_name: number}

    :param train_dir_path: Path to the directory containing the sub-folder with the respective images
    :return: Dictionary containing the {label name as string: number representing the class}
    """
    sub_dir_paths = [dir_path[0] for dir_path in os.walk(train_dir_path)][1:]
    return {data_path.rsplit("""\\""", maxsplit=1)[1]: num for num, data_path in enumerate(sub_dir_paths)}
