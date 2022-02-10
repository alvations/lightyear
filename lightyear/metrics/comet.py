from functools import partial

import torch
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader

import comet

class COMETScore:
    def __init__(self, modelname='wmt20-comet-da'):
        _cuda = 1 if torch.cuda.is_available() else 0
        self.model = comet.models.load_from_checkpoint(
            comet.download_utils.download_model(modelname))
        self.trainer = Trainer(gpus=_cuda, deterministic=True, logger=False)
        self.prep_sample = partial(self.model.prepare_sample, inference=True)

    def score(self, hyp: str, ref: str, src: str=None, normalize=True, clip=True, *args, **kwargs):
        data = {"src": src, "mt": hyp, "ref": ref} if src else {"src": ref, "mt": hyp, "ref": ref}
        dataloader = DataLoader(
            dataset=[data],
            batch_size=1,
            collate_fn=self.prep_sample,
            num_workers=1,
        )
        predictions = self.trainer.predict(
            self.model, dataloaders=dataloader, return_predictions=True,
        )
        score = torch.cat(predictions, dim=0).tolist()[0]
        score =  score * 100 if normalize else score
        return {'comet_score': {'score': score}}
