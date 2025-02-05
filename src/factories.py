from typing import Optional
from src import metrics, models, frameworks, datasets
from src.configs import DataConfigs, DecoderConfigs, ModelConfigs, FrameworkConfigs


def get_dataset(data_configs: DataConfigs, **kwargs):
    return getattr(datasets, data_configs.name)(
        data_configs=data_configs,
        **kwargs,
    )


def get_model(
    model_configs: ModelConfigs,
    decoder_configs: DecoderConfigs,
):
    return getattr(models, decoder_configs.method)(
        model_configs=model_configs,
        decoder_configs=decoder_configs,
    )


def get_metrics(data_configs: DataConfigs, framework_configs: FrameworkConfigs):
    return getattr(metrics, data_configs.name)(framework_configs)


def get_framework(framework_configs: FrameworkConfigs, data_configs: DataConfigs, model, **kwargs):
    return getattr(frameworks, framework_configs.name)(
        framework_configs=framework_configs,
        data_configs=data_configs,
        model=model,
        **kwargs,
    )