import yaml
import logging
from typing import Union

from wrapyfi.utils import SingletonOptimized


class ConfigManager(metaclass=SingletonOptimized):
    """
    The configuration manager is a singleton which is invoked once throughout the runtime.
    """
    def __init__(self, config: Union[dict, str], **kwargs):
        """
        Initializing the ConfigManager. The configuration can be provided as a yaml file name or as a dictionary.

        :param config: Callable[..., Any]: Loads a yaml configuration file when "str" type provided. Directly set when "dict" type provided
        """
        if isinstance(config, str):
            self.config = self.__loadfile(config)
            logging.info(f"Loaded Wrapyfi configuration: {self.config}")
        elif isinstance(config, dict):
            self.config = config
            logging.info(f"Loaded Wrapyfi configuration: {self.config}")
        else:
            self.config = []

    @staticmethod
    def __loadfile(filename: str):
        """
        Loading the yaml configuration file.

        :param filename: str: The file name
        :return: dict: The configuration object
        """
        with open(filename, "r") as fp:
            config = yaml.safe_load(fp)
        return config

    def __writefile(self, filename: str):
        """
        Saving the yaml configuration file.

        :param filename: str: The file name
        """
        with open(filename, "w") as fp:
            yaml.safe_dump(self.config, fp)

