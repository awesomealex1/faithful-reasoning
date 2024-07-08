from typing import List, Optional, Tuple

import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs import DecoderConfigs, ModelConfigs, PromptConfigs

from src.models.base_model import BaseModel


class DoLa(BaseModel):
    def __init__(self) -> None:
        pass
