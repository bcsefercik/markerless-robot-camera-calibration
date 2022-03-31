import os
import yaml
import argparse

from copy import deepcopy

from utils.structs import SingletonMeta
from utils.logger import Logger
import utils.helpers as helpers

BASE_PATH = os.path.abspath(os.path.dirname(__file__))

class Config(metaclass=SingletonMeta):
    def __init__(self, path="config/default.yaml", overriding_config=dict()):
        self.config = {
            'config': path,
            'log_path': os.path.join(
                os.path.dirname(BASE_PATH),
                'log.log'
            )
        }

        parser = argparse.ArgumentParser()

        parser.add_argument('--config', type=str)
        parser.add_argument('--log_path', type=str)
        parser.add_argument('--exp_path', type=str)
        parser.add_argument('--override', type=str, default=None)

        xargs = vars(parser.parse_args())
        xconfig = {k: v for k, v in xargs.items() if v is not None}

        self.config.update(xconfig)

        with open(self.config['config'], 'r') as f:
            self.config.update(yaml.safe_load(f))

        if (not overriding_config) and xconfig.get('override'):
            with open(xconfig['override'], 'r') as f:
                overriding_config = yaml.safe_load(f)

        self.override(self.config, overriding_config)
        self.config.update(xconfig)

        self.config_dict = deepcopy(self.config)

        for k, v in self.config.items():
            if isinstance(v, dict):
                self.config[k] = helpers.convert_dict_to_namespace(v)

        self.__dict__.update(self.config)

    def __call__(self):
        return self.config_dict

    def override(self, config, override_config):
        for k in (override_config.keys() - config.keys()):
            config[k] = override_config[k]

        for k, v in config.items():
            if isinstance(override_config, dict) and k in override_config:
                if isinstance(v, dict):
                    self.override(config[k], override_config[k])
                else:
                    config[k] = override_config[k]


Logger(filename=Config().log_path).get()
