# DotHash: Estimating Set Similarity Metrics for Link Prediction and Document Deduplication

This repository contains the source code for the research paper published at Knowledge Discovery and Data Mining Conference (KDD) 2023.


## Requirements

The code is written in Python 3.10. The required packages to run the experiments can be found in `requirements.txt`. To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Experiments

The experiments are divided into two parts: (1) link prediction and (2) document deduplication. The experiments can be run using the following commands:

### Link Prediction

```bash
python link_prediction.py --help
```

### Document Deduplication

Experiments with the `core` dataset require data to be downloaded from [Google Drive](https://drive.google.com/file/d/1uBPhyHnv74ApCw7ldMnrHpWs1Yv2pbYh/view?usp=share_link) and placed in the `data` directory.

```bash 
python document_deduplication.py --help
```


## Citation

If you use this code for your research, please cite our paper:

```
@inproceedings{dothash,
  title={DotHash: Estimating Set Similarity Metrics for Link Prediction and Document Deduplication},
  author={Nunes, Igor and Heddes, Mike and Vergés, Pere and Abraham, Danny and Veidenbaum, Alex and Nicolau, Alexandru and Givargis, Tony},
  booktitle={Proceedings of the 29th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2023}
}
```