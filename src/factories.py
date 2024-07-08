from typing import Optional

from src import datasets, decoders
from src.configs import (
    DataConfigs,
    DecoderConfigs,
)
from src.model import HFModel


def get_dataset(
    data_configs: DataConfigs,
):
    return getattr(datasets, data_configs.name)(
        data_configs=data_configs,
    )


def get_decoder(
    model: HFModel,
    decoder_configs: DecoderConfigs,
):
    return getattr(decoders, decoder_configs.name)(
        model=model,
        decoder_configs=decoder_configs,
    )
