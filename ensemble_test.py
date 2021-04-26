from pytorch_lightning import Trainer

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_ensemble import LitEnsemble

test_year = "2019"
ckp_paths = [
    "lightning_logs/version_0/checkpoints/epoch=259-step=97759.ckpt",
    "lightning_logs/version_1/checkpoints/epoch=275-step=103775.ckpt",
]

if __name__ == "__main__":
    trainer = Trainer(logger=False, gpus=1)

    dm = CROHMEDatamodule(test_year=test_year)

    model = LitEnsemble(ckp_paths)

    trainer.test(model, datamodule=dm)
