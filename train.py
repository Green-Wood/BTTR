from pytorch_lightning.utilities.cli import LightningCLI

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

cli = LightningCLI(LitBTTR, CROHMEDatamodule)
