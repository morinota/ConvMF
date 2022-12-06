import os
from dataclasses import dataclass
from typing import Dict, Hashable, List, NamedTuple, Optional

import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm


@dataclass
class EmbeddingLayerInfo:
    layer: nn.Embedding
    embed_dim: int
    vocab_num: int
