import yaml
import logging

from wrapify.utils import SingletonOptimized


class ConfigManager(metaclass=SingletonOptimized):
    """
    The configuration manager is a singleton which is invoked once throughout the runtime
    """
    def __init__(self, config, **kwargs):
        """
        Initializing the ConfigManager
        :param config: Loads a yaml configuration file when "str" type provided. Directly set when "dict" type provided
        """
        if isinstance(config, str):
            self.config = self.__loadfile__(config)
            logging.info(f"Loaded Wrapify configuration: {self.config}")
        elif isinstance(config, dict):
            self.config = config
            logging.info(f"Loaded Wrapify configuration: {self.config}")
        else:
            self.config = []

    @staticmethod
    def __loadfile__(filename):
        """
        Loading the yaml configuration file
        :param filename: The file name
        :return: The configuration object
        """
        with open(filename, "r") as fp:
            config = yaml.safe_load(fp)
        return config

    def __writefile__(self, filename):
        """
        Saving the yaml configuration file
        :param filename: The file name
        :return: None
        """
        with open(filename, "w") as fp:
            yaml.safe_dump(self.config, fp)

