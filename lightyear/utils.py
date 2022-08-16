
import gc

import torch
from torch import nn
from torch.nn.utils import prune


def quantize_model(model: nn.Module, precision=torch.qint8, layers={nn.LayerNorm, nn.Linear, nn.Dropout}):
	model_q = torch.quantization.quantize_dynamic(
		model, layers, dtype=precision)
	del model
	gc.collect()
	return model_q

def prune_model(model: nn.Module, amount=0.2, layers=(nn.Linear,)):
    for module in model.modules():
        if isinstance(module, layers):
            prune.l1_unstructured(module, 'weight', amount=amount)
            prune.remove(module, 'weight')
    return model
