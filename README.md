<div align="center">    
 
# Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer    

[![arXiv](https://img.shields.io/badge/arXiv-2105.02412-b31b1b.svg)](https://arxiv.org/abs/2105.02412)

[Springer](https://link.springer.com/chapter/10.1007%2F978-3-030-86331-9_37)
 
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
@InProceedings{10.1007/978-3-030-86331-9_37,
author="Zhao, Wenqi
and Gao, Liangcai
and Yan, Zuoyu
and Peng, Shuai
and Du, Lin
and Zhang, Ziyin",
editor="Llad{\'o}s, Josep
and Lopresti, Daniel
and Uchida, Seiichi",
title="Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer",
booktitle="Document Analysis and Recognition -- ICDAR 2021",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="570--584",
abstract="Encoder-decoder models have made great progress on handwritten mathematical expression recognition recently. However, it is still a challenge for existing methods to assign attention to image features accurately. Moreover, those encoder-decoder models usually adopt RNN-based models in their decoder part, which makes them inefficient in processing long sequences. In this paper, a transformer-based decoder is employed to replace RNN-based ones, which makes the whole model architecture very concise. Furthermore, a novel training strategy is introduced to fully exploit the potential of the transformer in bidirectional language modeling. Compared to several methods that do not use data augmentation, experiments demonstrate that our model improves the ExpRate of current state-of-the-art methods on CROHME 2014 by 2.23{\%}. Similarly, on CROHME 2016 and CROHME 2019, we improve the ExpRate by 1.92{\%} and 2.28{\%} respectively.",
isbn="978-3-030-86331-9"
}
```   
