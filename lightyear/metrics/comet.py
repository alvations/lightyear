from functools import partial

import torch
from torch import nn
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelPruning
from torch.utils.data import DataLoader

import comet

def compute_amount(epoch):
    """Apply the lottery ticket hypothesis"""
    if epoch == 10:
        return 0.5
    elif epoch == 50:
        return 0.25
    elif 75 < epoch < 99:
        return 0.01


class COMETScore:
    def __init__(self, modelname='wmt20-comet-da'):
        _cuda = 1 if torch.cuda.is_available() else 0
        self.model = torch.quantization.quantize_dynamic(
            comet.models.load_from_checkpoint(
                comet.download_utils.download_model(modelname)),
             {nn.LayerNorm, nn.Linear, nn.Dropout},
             dtype=torch.qint8)

        self.trainer = Trainer(gpus=_cuda, deterministic=True, logger=False,
            callbacks=[ModelPruning("l1_unstructured", amount=compute_amount)])
        self.prep_sample = partial(self.model.prepare_sample, inference=True)

    def score(self, hyp: str, ref: str, src: str=None, normalize=True, clip=True, *args, **kwargs):
        data = {"src": src, "mt": hyp, "ref": ref} if src else {"src": ref, "mt": hyp, "ref": ref}
        dataloader = DataLoader(
            dataset=[data],
            batch_size=1,
            collate_fn=self.prep_sample,
            num_workers=0,
        )
        predictions = self.trainer.predict(
            self.model, dataloaders=dataloader, return_predictions=True,
        )
        score = torch.cat(predictions, dim=0).tolist()[0]
        score =  score * 100 if normalize else score
        return {'comet_score': {'score': score}}
