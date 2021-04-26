from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.cli import LightningCLI

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

seed_everything(7)

cli = LightningCLI(LitBTTR, CROHMEDatamodule)
