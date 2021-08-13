import pandas as pd
import pickle
import argparse
import os
import numpy as np
from ast import literal_eval
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from ArticutAPI import Articut
from scipy import spatial


def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--filename",
                        type=str,
                        required=True,
                        help="filename of data")
    opt.add_argument("--reason",
                        type=str,
                        required=True,
                        help="query reason")
    opt.add_argument("--issue",
                        type=str,
                        required=True,
                        help="query related issues")
    opt.add_argument("--TOPK",
                        type=int,
                        required=True,
                        help="Top K related")
    
    return opt


def split_train_test(df):
    df['year'] = df['date'].apply(lambda x: int(x.split('-')[0]))

    train = df[df['year'] != 2020].reset_index(drop=True)
    test = df[df['year'] == 2020].reset_index(drop=True)

    print('Train size: {}, test size: {}'.format(len(train), len(test)))

    # train.to_csv('train.csv', index=False)
    # test.to_csv('test.csv', index=False)

    return train, test


def rule_based(data, query):
    top_k = query['TOPK']
    match_indexes = []
    same_issue_count = [0 for _ in range(len(data))]
    for idx in range(len(data)):
        if data['reason'][idx] == query['reason']:
            related_issues = literal_eval(data['relatedIssues'][idx])

            for query_law in query['issue']:
                for data_law in related_issues:
                    if query_law['lawName'] == data_law['lawName'] and query_law['issueRef'] == data_law['issueRef']:
                        same_issue_count[idx] += 1
                        break

    same_issue_count = np.array(same_issue_count)
    non_zero_indexes = np.nonzero(same_issue_count)[0]

    top_indexes = same_issue_count.argsort()[-top_k:][::-1]

    # remove zero indexes
    top_nonzero_indexes = []
    for index in top_indexes:
        if index not in non_zero_indexes:
            top_nonzero_indexes.append(index)

    # if len(top_nonzero_indexes) != len(top_indexes):
    #     print("HI")

    if len(top_nonzero_indexes) == top_k:
        print("HI")

    matches = data.loc[top_nonzero_indexes]
    del matches['judgement']
    matches.to_csv('rule_based_results/rule_based_{}.csv'.format(query['no']), index=False)
    return matches


def using_bert(data, query):
    from lawbert import transform_embedding
    max_length = 512

    # store embedding from BERT of training data
    if not os.path.isfile('train_bert.pickle'):
        train_embedding = {}
        for idx in tqdm(range(len(data))):
            if len(query['truth']) >= max_length:
                truth = data['truth'][idx][:max_length]
            else:
                truth = data['truth'][idx]
            outputs_truth = transform_embedding(truth)

            outputs_reason = transform_embedding(data['reason'][idx])['pooler_output'][0].tolist()

            # parse new reason by sections of law
            new_reason = set(literal_eval(data['new_reason'][idx]))
            outputs_new_reason = None
            for reason in new_reason:
                if outputs_new_reason is None:
                    outputs_new_reason = transform_embedding(reason)['pooler_output'][0].tolist()
                else:
                    outputs_new_reason = [sum(x) for x in zip(outputs_new_reason, transform_embedding(reason)['pooler_output'][0].tolist())]

            train_embedding[data['no'][idx]] = (outputs_reason, outputs_new_reason, outputs_truth['pooler_output'][0].tolist())

        with open('train_bert.pickle', 'wb') as file:
            pickle.dump(train_embedding, file)

    top_k = query['TOPK']

    # get query embedding from BERT
    if len(query['truth']) >= max_length:
        query['truth'] = query['truth'][:max_length]
    outputs_truth = transform_embedding(query['truth'])
    query_truth_embedding = outputs_truth['pooler_output'][0].tolist()

    query_reason_embedding = transform_embedding(query['reason'])['pooler_output'][0].tolist()

    # parse new reason by sections of law
    new_reason = set(query['reason'])
    query_new_reason_embedding = None
    for reason in new_reason:
        if query_new_reason_embedding is None:
            query_new_reason_embedding = transform_embedding(reason)['pooler_output'][0].tolist()
        else:
            query_new_reason_embedding = [sum(x) for x in zip(query_new_reason_embedding, transform_embedding(reason)['pooler_output'][0].tolist())]

    with open('train_bert.pickle', 'rb') as f:
        trained_embeddings = pickle.load(f)

    similarity_index = {}
    for no, trained_embedding in trained_embeddings.items():
        similarity_truth = 1 - spatial.distance.cosine(query_truth_embedding, trained_embedding[2])
        similarity_new_reason = 1 - spatial.distance.cosine(query_new_reason_embedding, trained_embedding[1])
        similarity_reason = 1 - spatial.distance.cosine(query_reason_embedding, trained_embedding[0])
        similarity_index[no] = (similarity_truth + similarity_new_reason + similarity_reason)
    
    similarity_index = dict(sorted(similarity_index.items(), key=lambda item: item[1], reverse=True))

    top_indexes = []
    for no, similarity in similarity_index.items():
        if len(top_indexes) >= top_k:
            break
        else:
            top_indexes.append(data.index[data['no'] == no].tolist()[0])

    matches = data.loc[top_indexes]
    del matches['judgement']
    matches.to_csv('bert_results/bert_{}.csv'.format(query['no']), index=False)
    return matches


def recommend_similar(train, test):
    y_true, y_pred = [], []
    for idx in tqdm(range(len(test))):
        query = {
            'no': test['no'][idx],
            'reason': test['reason'][idx],
            'new_reason': literal_eval(test['new_reason'][idx]),
            'issue': literal_eval(test['new_relatedIssues'][idx]),
            'truth': test['truth'][idx],
            'penalty': test['maxpenalty'][idx],
            'TOPK': config['TOPK']
        }

        # matches = rule_based(train, query)
        matches = using_bert(train, query)

        if len(matches) != 0:
            y_pred.append(sum(matches['maxpenalty']) / len(matches['maxpenalty']))
        else:
            y_pred.append(0)
        y_true.append(query['penalty'])

    mae = mean_absolute_error(y_true, y_pred)
    print("MAE : {}".format(mae))


if __name__ == '__main__':
    opt = get_argument()
    config = vars(opt.parse_args())

    # read data
    filename = config['filename']
    df = pd.read_csv(filename)

    train, test = split_train_test(df)

    if not os.path.exists('rule_based_results'):
        os.makedirs('rule_based_results')
    if not os.path.exists('bert_results'):
        os.makedirs('bert_results')

    recommend_similar(train, test)