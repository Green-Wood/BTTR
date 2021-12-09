<div align="center">    
 
# Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer    

[![arXiv](https://img.shields.io/badge/arXiv-2105.02412-b31b1b.svg)](https://arxiv.org/abs/2105.02412)

[![Springer](https://badgen.net/badge/Springer/BTTR-paper/purple)](https://link.springer.com/chapter/10.1007%2F978-3-030-86331-9_37)
 
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
conda install --yes -c pytorch pytorch=1.7.0 torchvision cudatoolkit=<your-cuda-version>
pip install -e .   
 ```   
 Next, navigate to any file and run it. It may take **6~7** hours to coverage on **4** gpus using ddp.
 ```bash
# module folder
cd BTTR

# train bttr model using 4 gpus and ddp
python train.py --config config.yaml  
```

For single gpu user, you may change the `config.yaml` file to
```yaml
gpus: 1
# gpus: 4
# accelerator: ddp
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

## Note
Metrics used in validation is not accurate.

For more accurate metrics:
1. use `test.py` to generate result.zip
2. download and install [crohmelib](http://saskatoon.cs.rit.edu:10001/root/crohmelib), [lgeval](http://saskatoon.cs.rit.edu:10001/root/lgeval), and [tex2symlg](https://www.cs.rit.edu/~crohme2019/downloads/convert2symLG.zip) tool.
3. convert tex file to symLg file using `tex2symlg` command
4. evaluate two folder using `evaluate` command

### Citation   
```
@article{zhao2021handwritten,
  title={Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer},
  author={Zhao, Wenqi and Gao, Liangcai and Yan, Zuoyu and Peng, Shuai and Du, Lin and Zhang, Ziyin},
  journal={arXiv preprint arXiv:2105.02412},
  year={2021}
}
```
```
@inproceedings{Zhao2021HandwrittenME,
  title={Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer},
  author={Wenqi Zhao and Liangcai Gao and Zuoyu Yan and Shuai Peng and Lin Du and Ziyin Zhang},
  booktitle={ICDAR},
  year={2021}
}
```   
