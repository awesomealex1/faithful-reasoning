from typing import Optional

from src import datasets, models, metrics
from src.configs import (
    DataConfigs,
    ModelConfigs,
    DecoderConfigs,
    PromptConfigs,
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
    prompt_configs: PromptConfigs,
):
    return getattr(models, decoder_configs.name)(
        model_configs=model_configs,
        decoder_configs=decoder_configs,
        prompt_configs=prompt_configs,
    )


def get_metrics(data_configs: DataConfigs):
    return getattr(metrics, data_configs.name)()
