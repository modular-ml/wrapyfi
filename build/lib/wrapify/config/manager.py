import threading
import yaml

lock = threading.Lock()


class SingletonOptimized(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(SingletonOptimized, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ConfigManager(metaclass=SingletonOptimized):
    """
    The configuration manager is a singleton which is invoked once throughout the runtime
    """
    def __init__(self, config):
        """
        Initializing the ConfigManager
        :param config: Loads a yaml configuration file when "str" type provided. Directly set when "dict" type provided
        """
        if isinstance(config, str):
            self.config = self.__loadfile__(config)
            print("loaded configuration", self.config)
        elif isinstance(config, dict):
            self.config = config
            print("loaded configuration", self.config)
        else:
            self.config = None

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

