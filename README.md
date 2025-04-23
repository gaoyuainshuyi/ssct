## Introduction

**The framework of SSCT:**

## Requirements 
Utilize `pip install -r requirements.txt` for the following dependencies.

*  Python 3.7.11  
*  PyTorch 1.7.1   
*  NumPy 1.21.5 
*  Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Download data and vocab
We follow [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features and vocabularies, which can be downloaded by using:

```bash
wget https://iudata.blob.core.windows.net/scan/data.zip
wget https://iudata.blob.core.windows.net/scan/vocab.zip
```
Another download link is available below：

```bash
https://drive.google.com/drive/u/0/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC
```

```
data
├── coco
│   ├── precomp  # pre-computed BUTD region features for COCO, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   └── id_mapping.json  # mapping from coco-id to image's file name
│   
│
├── f30k
│   ├── precomp  # pre-computed BUTD region features for Flickr30K, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   └── id_mapping.json  # mapping from f30k index to image's file name
│   
│
└── vocab  # vocab files provided by SCAN (only used when the text backbone is BiGRU)
```

## Pre-trained models and evaluation
Modify the **model_path**, **split**, **fold5** in the `eval.py` file. 
Note that `fold5=True` is only for evaluation on mscoco1K (5 folders average) while `fold5=False` for mscoco5K and flickr30K.

Then run `python eval.py` in the terminal.

## Training new models from scratch 

Run `./train_xxx_xxx.sh` in the terminal:
