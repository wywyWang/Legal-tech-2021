# Legal-tech-2021
Our project **National Judges: Miss JuBERT** won the 3rd Place and National Judges Award in [Legal-tech Hackathon 2021](https://hackathon.lawsnote.com/). Presentation slides can be seen at [link](https://drive.google.com/file/d/15SJCkzFskRitCXg_4Yh8Vq0-R8b0QBIu/view?usp=sharing).

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

## Contributors
Our team is composed of Wei-Wei Du, Wei-Yao Wang, Li June Wei, Yi-Han Yang.
Part of the codes are contributed from [Wei-Wei Du](https://github.com/wwweiwei).

## Citation and Contact
If you found this useful, please give us stars!
If you have any question, please send to *sf1638.cs05@nctu.edu.tw*.
