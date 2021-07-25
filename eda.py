import datatable as dt
import pandas as pd
from tqdm import tqdm
from collections import Counter
from ast import literal_eval
import sys
import os


def read_csv(folder):
    """Read all available csv paths

    Args:
        folder ([type]): [description]

    Returns:
        List: available csv
    """
    csv_paths = [(f, os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith('.csv') and '地方法院_刑事' in f]
    return csv_paths


def process_file(filename):
    csv_paths = read_csv(filename)

    all_columns = {
        'court', 'date', 'no', 'sys', 'reason', 'judgement', 'attachAsJudgement', 'attachments', 'type', 'historyHash', 'mainText', 'opinion', 'relatedIssues', 'party'
    }
    unused_columns = ['attachAsJudgement', 'attachments']

    data_len = 0
    for csv_name, csv_path in tqdm(csv_paths):
        processed_name = 'no_text_data/' + csv_name
        df = dt.fread(csv_path)

        # remove unused columns
        if 'attachAsJudgement' in df.keys():
            del df[:, unused_columns]
        else:
            del df[:, 'attachments']

        # remove all text
        del df[:, ['judgement', 'opinion']]

        data_len += df.shape[0]
        df.to_csv(processed_name)
    print("Total data count: {}".format(data_len))


def concat_file(filename):
    """Concat all no-text data into one file for EDA

    Args:
        filename (string): Folder path of no-text data
    """
    csv_paths = read_csv(filename)

    data_len = 0
    df_total = None
    for csv_name, csv_path in tqdm(csv_paths):
        df = dt.fread(csv_path)
        data_len += df.shape[0]

        if df_total is None:
            df_total = df.copy()
        else:
            df_total.rbind(df)

    print("Total data count: {}".format(data_len))
    df_total.to_csv('eda_concat.csv')


def filter_law(filename):
    df = dt.fread(filename).to_pandas()
    print(df.shape)

    # filter unused law data
    criminal_law_filter = [
        '125 2', '126 2', '135 2', '136 2', '177 2', '185 2', '185-1 2', '185-1 4', '185-2 3', '185-3 2',
        '185-3 3', '185-4', '186-1 2', '187-2 2', '187-3 2', '189 2', '189-2 2', '190 2', '190-1 3',
        '190-1 4', '191-1 3', '226 1', '271 1', '272', '273 1', '274 1', '275 1', '277 2', '278', '279',
        '282', '283', '286 3', '286 4', '290 2', '291 2', '293 2', '294 2', '302 2', '325 2', '328 3',
        '332 1', '333 3', '334 1', '334 2', '347 2', '348 1', '348 2', '353 2',
        '185-1 2', '185-1 4', '185-2 3', '187-3 2', '226', '226-1', '272 1', '278 2', '286 3', '286 4', 
        '328 3', '332 2', '333 3', '334', '347 2', '348']
    other_law_filter = {'懲治走私條例': '4', '藥事法': '82 2', '兒童及少年性剝削防制條例': '37', '民用航空法': ['101 3', '110 2']}
    keep_index = []
    for idx in tqdm(range(len(df))):
        # we only need 判決
        if df['type'][idx] == '裁定':
            continue

        related_issues = literal_eval(df['relatedIssues'][idx])
        for law in related_issues:
            split_issues = law['issueRef'].split(' ')
            if '刑法' in law['lawName']:
                for candidate in criminal_law_filter:
                    split_candidate = candidate.split(' ')
                    if split_candidate[0] == split_issues[0]:
                        # only one
                        if len(split_candidate) == 1:
                            keep_index.append(idx)
                            break
                        elif len(split_candidate) == 2:
                            if len(split_issues) < 2:
                                continue
                            if split_candidate[1] == split_issues[1]:
                                keep_index.append(idx)
                                break
                            else:
                                pass
                        else:
                            # maximum of filter's length  is 2
                            raise NotImplementedError
            elif law['lawName'] in other_law_filter.keys():
                if law['lawName'] == '民用航空法':
                    for candidate in other_law_filter[law['lawName']]:
                        split_candidate = candidate.split(' ')
                        if split_candidate[0] == split_issues[0] and split_candidate[1] == split_issues[1]:
                            keep_index.append(idx)
                            break
                        else:
                            pass
                else:
                    split_candidate = other_law_filter[law['lawName']].split(' ')
                    if split_candidate[0] == split_issues[0]:
                        if len(split_candidate) == 1:
                            keep_index.append(idx)
                            break
                        elif len(split_candidate) == 2:
                            if len(split_issues) < 2:
                                continue
                            if split_candidate[1] == split_issues[1]:
                                keep_index.append(idx)
                                break
                            else:
                                pass
                        else:
                            # maximum of filter's length  is 2
                            raise NotImplementedError
    
    # keep_data = pd.concat(keep_data, ignore_index=True, sort=False)
    print(len(keep_index), len(set(keep_index)))
    keep_index = set(keep_index)                    # avoid count same index
    df = df.loc[keep_index].reset_index(drop=True)
    print(df.shape)
    df.to_csv('filter_concat.csv', index=False)


def print_value_counts(data, column, normalize=True):
    counts = data[column].value_counts(normalize=normalize)
    filename = column + '_count.csv'
    counts.to_csv(filename, index=True, header=True)


def EDA(filename):
    df = dt.fread(filename).to_pandas()

    print_value_counts(df, 'reason', normalize=False)
    print_value_counts(df, 'type', normalize=False)
    print_value_counts(df, 'court', normalize=False)
    print_value_counts(df, 'historyHash', normalize=False)

    df['year'] = df['date'].apply(lambda x: x.split('-')[0])
    print_value_counts(df, 'year', normalize=True)

    # parse related issues
    law_counter = Counter()
    for idx in tqdm(range(len(df))):
        related_issues = literal_eval(df['relatedIssues'][idx])
        for law in related_issues:
            law_counter.update([law['lawName']])

    with open('law_count', 'w') as result_log:
        result_log.write(str(len(law_counter)))
        result_log.write('\n')
        result_log.write(str(law_counter))


if __name__ == '__main__':
    # process_file(sys.argv[1])
    # concat_file(sys.argv[1])
    filter_law(sys.argv[1])
    # EDA(sys.argv[1])