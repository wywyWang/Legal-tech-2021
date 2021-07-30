import pandas as pd
import argparse
import os
from ast import literal_eval


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
    
    return opt


def rule_based(data, query):
    match_indexes = []
    for idx in range(len(data)):
        if data['reason'][idx] == query['reason']:
            related_issues = literal_eval(data['relatedIssues'][idx])
            split_query = query['issue'].split(' ')
            for law in related_issues:
                split_issues = law['issueRef'].split(' ')
                if law['lawName'] == split_query[0] and law['issueRef'] == split_query[1]:
                    match_indexes.append(idx)
                    break

    matches = data.loc[match_indexes]
    matches.to_csv('results/rule_based.csv', index=False)



def recommend_similar(filename, query):
    df = pd.read_csv(filename)
    print(query)
    matches = rule_based(df, query)


if __name__ == '__main__':
    opt = get_argument()
    config = vars(opt.parse_args())

    filename = config['filename']
    query = {
        'reason': config['reason'],
        'issue': config['issue']
    }

    if not os.path.exists('results'):
        os.makedirs('results')

    recommend_similar(filename, query)