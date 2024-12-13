from dataclasses import dataclass

@dataclass
class DevConfigs():
    """
    Configurations for development / experimentation. Should not be permanent or for production.
    Move configurations into a dedicated object.
    """
    tmp: str