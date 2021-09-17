# Legal-tech-2021

## Prepare
### Install Environment
```=bash
conda env create -f environment.yml
```

### Data
Please refer to [legal-tech hackathon 2021](https://hackathon.lawsnote.com/) due to the license.

## Usage
### Concat files in the same folder
```=bash
python concat_file.py $1
```

### Pre-process files to the processed file
```=bash
python eda.py $1
```
Note that you should modify processing function to corresponding needs.

### Get training embeddings
```=bash
python recommendation.py --filename $1 --TOPK $2
```
Note that you should modify function in `recommend_similar` to specify what mode you want to test.