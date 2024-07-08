from typing import Optional

from src import datasets, models, metrics
from src.configs import (
    DataConfigs,
    ModelConfigs,
    DecoderConfigs,
)


def get_dataset(
    data_configs: DataConfigs,
):
    return getattr(datasets, data_configs.name)(
        data_configs=data_configs,
    )


def get_model(
    model_configs: ModelConfigs,
    decoder_configs: DecoderConfigs,
):
    return getattr(models, decoder_configs.name)(
        model_configs=model_configs,
        decoder_configs=decoder_configs,
    )


def get_metrics(data_configs: DataConfigs):
    return getattr(metrics, data_configs.name)()
