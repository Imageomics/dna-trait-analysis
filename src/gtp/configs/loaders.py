import yaml

from gtp.configs.project import GenotypeToPhenotypeConfigs

DEFAULT_YAML_CONFIG_PATH = "configs/default.yaml"

def load_configs(config_path: str = None) -> object:
    """Loads configs from a YAML file.

    Args:
        config_path (str, optional): Path to YAML file representing configurations. Defaults to DEFAULT_YAML_CONFIG_PATH.

    Returns:
        obj: Object containing values in the configuration file
    """
    if not config_path:
        config_path = DEFAULT_YAML_CONFIG_PATH
    
    with open(config_path, 'r') as f:
        yaml_configs = yaml.safe_load(f)
    
    project_configs = GenotypeToPhenotypeConfigs(src_yaml=yaml_configs, **yaml_configs)    
    return project_configs