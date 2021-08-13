from transformers import (
  BertTokenizerFast,
  BertModel,
)
import torch


tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('ckiplab/bert-base-chinese')


def transform_embedding(text):
    tokenized_words = tokenizer(text)
    input_ids = torch.tensor(tokenized_words['input_ids']).unsqueeze(0)
    outputs = model(input_ids)
    return outputs