import pandas as pd
import argparse
import os
import numpy as np
from ast import literal_eval
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error


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

    matches = data.loc[top_nonzero_indexes]
    matches.to_csv('results/rule_based_{}.csv'.format(query['no']), index=False)
    return matches


def recommend_similar(train, test):
    y_true, y_pred = [], []
    for idx in tqdm(range(len(test))):
        query = {
            'no': test['no'][idx],
            'reason': test['reason'][idx],
            'issue': literal_eval(test['new_relatedIssues'][idx]),
            'truth': test['truth'][idx],
            'penalty': test['maxpenalty'][idx],
            'TOPK': config['TOPK']
        }

        matches = rule_based(train, query)

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

    if not os.path.exists('results'):
        os.makedirs('results')

    recommend_similar(train, test)