from functools import partial
import multiprocessing

import torch
from torch import nn
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelPruning
from torch.utils.data import DataLoader

from ..utils import prune_model, quantize_model

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
    def __init__(self, modelname='wmt20-comet-da', cuda=True):
        _cuda = 1 if cuda and torch.cuda.is_available() else 0

        self.model = comet.models.load_from_checkpoint(
                comet.download_utils.download_model(modelname))

        self.trainer = Trainer(gpus=_cuda, deterministic=True, logger=False,
            enable_progress_bar=False,
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
        with torch.no_grad():
            predictions = self.trainer.predict(
                self.model, dataloaders=dataloader, return_predictions=True,
            )
        score = torch.cat(predictions, dim=0).tolist()[0]
        score =  score * 100 if normalize else score
        return {'comet_score': {'score': score}}

    def score_batch(self, hyps, refs, srcs=None, normalize=True, clip=True, *args, **kwargs):
        if srcs:
            data = [{"src": s, "mt": h, "ref": r} for h, r, s in zip(hyps, refs, srcs)]
        else:
            data = [{"src": r, "mt": h, "ref": r} for h, r in zip(hyps, refs)]

        dataloader = DataLoader(
            dataset=data,
            batch_size=1,
            collate_fn=self.prep_sample,
            num_workers=0,
        )
        predictions = self.trainer.predict(
            self.model, dataloaders=dataloader, return_predictions=True,
        )
        score = torch.cat(p, dim=0).tolist()
        score =  score * 100 if normalize else score
        return {'comet_score': {'score': score}}
