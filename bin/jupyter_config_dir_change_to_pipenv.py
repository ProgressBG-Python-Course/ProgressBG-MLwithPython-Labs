import os

from jupyter_core.paths import ENV_CONFIG_PATH, jupyter_config_dir
from traitlets.config.manager import BaseJSONConfigManager

# print(f'jupyter_config_dir: {jupyter_config_dir()}')
# print(f'ENV_CONFIG_PATH: {ENV_CONFIG_PATH}')


def set_ENV_config():
  cfg_name = 'jupyter_notebook_config'

  user_cm = BaseJSONConfigManager(config_dir=jupyter_config_dir())

  # record existing user config content
  usr_cfg = user_cm.get(cfg_name)

  # remove the user config to prevent it overriding env
  if os.path.exists (user_cm.file_name(cfg_name)):
      os.remove(user_cm.file_name(cfg_name))

  # now write env config
  env_cm = BaseJSONConfigManager(config_dir=ENV_CONFIG_PATH[0])

  # first anything we read from user config:
  env_cm.update(cfg_name, usr_cfg)

  # now ensure notebook's config API writes to env config:
  env_cm.update(cfg_name, {'ConfigManager': {'write_config_dir':
      os.path.join(ENV_CONFIG_PATH[0], 'nbconfig')
  }})

set_ENV_config()