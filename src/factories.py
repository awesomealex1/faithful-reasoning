from typing import Optional

from src import datasets, decoders
from src.configs import (
    DataConfigs,
    DecoderConfigs,
)


def get_dataset(
    data_configs: DataConfigs,
):
    return getattr(datasets, data_configs.name)(
        data_configs=data_configs,
    )


def get_decoder(
    decoder_configs: DecoderConfigs,
):
    return getattr(decoders, decoder_configs.name)(
        decoder_configs=decoder_configs,
    )
