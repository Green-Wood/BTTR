<div align="center">    
 
# Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer    
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
--> 
</div>
 
## Description   
Convert offline handwritten mathematical expression to LaTeX sequence using bidirectionally trained transformer.   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/Green-Wood/BTTR

# install project   
cd BTTR
conda create -y -n bttr python=3.7
conda activate bttr
conda install --yes -c pytorch pytorch=1.7.0 torchvision cudatoolkit=11.0
pip install -e .   
 ```   
 Next, navigate to any file and run it.
 ```bash
# module folder
cd BTTR

# train bttr model  
python train.py --config config.yaml  
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from bttr.datamodule import CROHMEDatamodule
from bttr import LitBTTR
from pytorch_lightning import Trainer

# model
model = LitBTTR()

# data
dm = CROHMEDatamodule(test_year=test_year)

# train
trainer = Trainer()
trainer.fit(model, datamodule=dm)

# test using the best model!
trainer.test(datamodule=dm)
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
