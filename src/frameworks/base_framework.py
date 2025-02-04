from abc import ABC, abstractmethod
from src.configs import FrameworkConfigs, DataConfigs


class BaseFramework(ABC):
    def __init__(
        self,
        framework_configs: FrameworkConfigs,
        data_configs: DataConfigs,
        model,
        **kwargs,
    ):
        self.framework_configs = framework_configs
        self.data_configs = data_configs
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    def generate(self):
        pass
