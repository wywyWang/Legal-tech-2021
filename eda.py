import datatable as dt
import pandas as pd
from tqdm import tqdm
from collections import Counter
from ast import literal_eval
import sys
import os
import re
from datetime import datetime
from ArticutAPI import Articut


def read_csv(folder):
    """Read all available csv paths

    Args:
        folder ([type]): [description]

    Returns:
        List: available csv
    """
    csv_paths = [(f, os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith('.csv') and '刑事' in f and '司法院－刑事補償_刑事' not in f and '最高' not in f]
    return csv_paths


def process_file(filename):
    """Remove some unused text to reduce file size

    Args:
        filename ([type]): folder of concat data
    """
    csv_paths = read_csv(filename)

    all_columns = {
        'court', 'date', 'no', 'sys', 'reason', 'judgement', 'attachAsJudgement', 'attachments', 'type', 'historyHash', 'mainText', 'opinion', 'relatedIssues', 'party'
    }
    unused_columns = ['attachAsJudgement', 'attachments']

    data_len = 0
    for csv_name, csv_path in tqdm(csv_paths):
        print(csv_name)
        processed_name = 'no_text_data/' + csv_name
        df = dt.fread(csv_path)

        # remove unused columns
        if 'attachAsJudgement' in df.keys():
            del df[:, unused_columns]
        else:
            del df[:, 'attachments']

        # remove all text
        del df[:, ['opinion']]

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
        print(csv_name)
        df = dt.fread(csv_path).to_pandas()
        data_len += df.shape[0]

        process_df = filter_law(df)

        if df_total is None:
            df_total = process_df.copy()
        else:
            df_total = pd.concat([df_total, process_df], ignore_index=True)

    print("Total data count: {}".format(data_len))
    df_total.to_csv('eda_concat.csv')


def filter_law(df):
    print(df.shape)

    # filter unused law data
    criminal_law_filter = [
        '125 2', '126 2', '135 3', '136 2', '177 2', '185 2', '185-1 2', '185-1 4', '185-2 3', '185-3 2',
        '185-3 3', '185-4', '186-1 2', '187-2 2', '187-3 2', '189 2', '189-2 2', '190 2', '190-1 3',
        '190-1 4', '191-1 3', '226 1', '271 1', '272', '273 1', '274 1', '275 1', '277 2', '278', '279',
        '282', '283', '286 3', '286 4', '290 2', '291 2', '293 2', '294 2', '302 2', '325 2', '328 3',
        '332 1', '333 3', '334 1', '334 2', '347 2', '348 1', '348 2', '353 2',
        '226', '226-1', '272 1', '278 2', 
        '332 2', '333 3', '334', '348']

    criminal_law_section = \
    {
        '125 2': '瀆職罪', '126 2': '瀆職罪', '135 3': '妨害公務罪', '136 2': '妨害公務罪', '177 2': '公共危險罪', 
        '185 2': '公共危險罪', '185-1 2': '公共危險罪', '185-1 4': '公共危險罪', '185-2 3': '公共危險罪', '185-3 2': '公共危險罪',
        '185-3 3': '公共危險罪', '185-4': '公共危險罪', '186-1 2': '公共危險罪', '187-2 2': '公共危險罪', '187-3 2': '公共危險罪', 
        '189 2': '公共危險罪', '189-2 2': '公共危險罪', '190 2': '公共危險罪', '190-1 3': '公共危險罪','190-1 4': '公共危險罪', 
        '191-1 3': '公共危險罪', '226 1': '妨害性自主罪', '271 1': '殺人罪', '272': '殺人罪', '273 1': '殺人罪', '274 1': '殺人罪', 
        '275 1': '殺人罪', '277 2': '傷害罪', '278': '傷害罪', '279': '傷害罪','282': '傷害罪', 
        '283': '傷害罪', '286 3': '傷害罪', '286 4': '傷害罪', '290 2': '墮胎罪', '291 2': '墮胎罪', 
        '293 2': '遺棄罪', '294 2': '遺棄罪', '302 2': '妨害自由罪', '325 2': '搶奪強盜及海盜罪', '328 3': '搶奪強盜及海盜罪',
        '332 1': '搶奪強盜及海盜罪', '333 3': '搶奪強盜及海盜罪', '334 1': '搶奪強盜及海盜罪', '334 2': '搶奪強盜及海盜罪', '347 2': '恐嚇及擄人勒贖罪', 
        '348 1': '恐嚇及擄人勒贖罪', '348 2': '恐嚇及擄人勒贖罪', '353 2': '毀棄損壞罪', 
        '226': '妨害性自主罪', '226-1': '妨害性自主罪', '272 1': '殺人罪', 
        '278 2': '傷害罪', '332 2': '搶奪強盜及海盜罪', 
        '333 3': '搶奪強盜及海盜罪', '334': '搶奪強盜及海盜罪', '348': '恐嚇及擄人勒贖罪'
    }

    keep_index, remove_index = [], []
    new_related_issues_all = []
    new_match_reason_all = []
    for idx in tqdm(range(len(df))):
        # we only need 判決
        if df['type'][idx] == '裁定':
            new_related_issues_all.append([])
            new_match_reason_all.append([])
            # truth_all.append([])
            continue

        related_issues = literal_eval(df['relatedIssues'][idx])
        new_related_issues, new_match_reason, truth = [], [], []
        need_to_process_text_flag = 0
        for law in related_issues:
            if '訴訟法' not in law['lawName']:
                new_related_issues.append(law)

            split_issues = law['issueRef'].split(' ')
            if '毒品危害防制條例' in law['lawName']:
                remove_index.append(idx)
            if '刑法' in law['lawName']:
                for candidate in criminal_law_filter:
                    split_candidate = candidate.split(' ')
                    if split_candidate[0] == split_issues[0]:
                        # only one
                        if len(split_candidate) == 1:
                            keep_index.append(idx)
                            new_match_reason.append(criminal_law_section[candidate])
                            need_to_process_text_flag = 1
                        elif len(split_candidate) == 2:
                            if len(split_issues) < 2:
                                continue
                            if split_candidate[1] == split_issues[1]:
                                keep_index.append(idx)
                                new_match_reason.append(criminal_law_section[candidate])
                                need_to_process_text_flag = 1
                            else:
                                pass
                        else:
                            # maximum of filter's length  is 2
                            raise NotImplementedError

        new_related_issues_all.append(new_related_issues)
        new_match_reason_all.append(set(new_match_reason))

        # # get truth from judgement if is our data
        # if need_to_process_text_flag:
        #     process_text = df['judgement'][idx].replace('\r\n', '').replace('\u3000', '')
        #     process_text = " ".join(process_text.split())
        #     process_text = re.sub('事[ ]*實', '事實', process_text)
        #     process_text = re.sub('理[ ]*由', '理由', process_text)
        #     if '簡' in df['no'][idx]:
        #         truth_condition = '事實(.|\n|\r)*中華民國'
        #         match = re.search(truth_condition, process_text)
        #         # 5 for 事實及理由 and 4 for 中華民國
        #         truth_all.append(process_text[match.start()+5:match.end()-4])
        #     else:
        #         truth_condition = '事實(.|\n|\r)*[ ]*理由'
        #         match = re.search(truth_condition, process_text)
        #         # 2 for 事實 and 理由
        #         truth_all.append(process_text[match.start()+2:match.end()-2])
        #         # if df['no'][idx] == '108,易,28':
        #         #     print(df['judgement'][idx])
        #         #     print()
        #         #     print(process_text[231:8376])
        #         #     print(match)
        #         #     1/0
        #     if match == None:
        #         raise NotImplementedError
        # else:
        #     truth_all.append([])

    df['new_relatedIssues'] = new_related_issues_all
    # del df['relatedIssues']
    df['new_reason'] = new_match_reason_all
    # df['truth'] = truth_all
    # del df['judgement']
    # print(len(keep_index), len(set(keep_index)), len(remove_index), len(set(remove_index)))
    keep_index = set(keep_index)                    # avoid count same index
    remove_index = set(remove_index)
    keep_index = [keep for keep in keep_index if keep not in remove_index]
    df = df.loc[keep_index].reset_index(drop=True)
    print(df.shape)

    return df
    # df.to_csv('filter_concat_new.csv', index=False)


def filter_data(filename):
    df = dt.fread(filename).to_pandas()
    history_file = open('historyHash_count.csv', 'r').readlines()
    print(df.shape)

    def transform_time(time):
        present = datetime.now()
        return present - datetime.strptime(time.split('T')[0], '%Y-%m-%d')

    # skip header and first empty row
    for i in tqdm(range(2, len(history_file))):
        id, count = history_file[i].strip().split(',')
        
        match_rows = df[df['historyHash'] == id].copy()
        match_rows['time_distance'] = match_rows['date'].apply(lambda x: transform_time(x))
        keep_index = match_rows['time_distance'].idxmin()
        deleted_indexes = match_rows[match_rows['time_distance'] != match_rows['time_distance'].min()].index

        df = df.drop(deleted_indexes).reset_index(drop=True)
    
    print(df.shape)
    df.to_csv('filter_eda.csv', index=False)


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
        related_issues = literal_eval(df['new_relatedIssues'][idx])
        for law in related_issues:
            law_counter.update([law['lawName']])

    with open('law_count', 'w') as result_log:
        result_log.write(str(len(law_counter)))
        result_log.write('\n')
        result_log.write(str(law_counter))


def filter_withdraw_penalty(filename):
    df = dt.fread(filename).to_pandas()
    print(df.shape)

    remove_index = []
    for idx in range(len(df)):
        if '免訴' in df['mainText'][idx] or '駁回' in df['mainText'][idx] or '不受理' in df['mainText'][idx] or ('原判決' in df['mainText'][idx] and '撤銷' in df['mainText'][idx]):
            remove_index.append(idx)
        if '簡' in df['no'][idx]:
            remove_index.append(idx)

    print(len(set(remove_index)))
    df = df.loc[~df.index.isin(remove_index)].reset_index(drop=True)
    print(df.shape)
    df.to_csv('filter_withdraw.csv', index=False)


def filter_penalty_lawyer(filename):
    df = dt.fread(filename).to_pandas()

    # add year column
    df['year'] = df['date'].apply(lambda x: x.split('-')[0])

    remove_index = []
    plaintiff_lawyer, defendant_lawyer = [0 for _ in range(len(df))], [0 for _ in range(len(df))]
    for idx in range(len(df)):
        # add lawyer
        party = literal_eval(df['party'][idx])
        for each_party in party:
            if 'agentAdLitem' in each_party['group']:
                if 'lawyer' not in each_party['group']:
                    print(each_party)
                    1/0
                else:
                    if 'plaintiff' in each_party['group']:
                        plaintiff_lawyer[idx] = 1
                    if 'defendant' in each_party['group']:
                        defendant_lawyer[idx] = 1

        if '無罪' in df['mainText'][idx] or '過失傷害' in df['mainText'][idx] or '受傷' in df['mainText'][idx] or '致人傷' in df['mainText'][idx] or '致人受傷' in df['mainText'][idx]:
            if '致人於死' in df['mainText']:
                print(idx)
                continue
            elif '無期徒刑' in df['mainText'][idx] or '死刑' in df['mainText'][idx]:
                print(idx)
                continue
            else:
                if df['maxpenalty'][idx] <= 3650:
                    remove_index.append(idx)
                else:
                    print(idx)

    df['plaintiff_lawyer'] = plaintiff_lawyer
    df['defendant_lawyer'] = defendant_lawyer
    print("Remove: {}".format(len(set(remove_index))))
    df = df.loc[~df.index.isin(remove_index)].reset_index(drop=True)
    df.drop(['C1', 'C0', 'sys', 'mypenalty', 'mypenalty_int', 'idx'], axis=1, inplace=True)
    print(df.shape)
    df.to_csv('new_data.csv', index=False)


def filter_truth(filename):
    df = dt.fread(filename).to_pandas()
    truth_all = []

    not_match_count = 0
    special_idx = [922, 1059, 1835, 2077, 2686, 2876, 3223, 3225, 3321, 3566, 3793, 3799]                # not follow form
    for idx in tqdm(range(len(df))):
        # get truth from judgement if is our data
        if df['judgement'][idx][0] == '[':
            process_text = literal_eval(df['judgement'][idx])
            filter_text = []
            for text in process_text:
                if type(text).__name__ == 'dict':
                    pass
                else:
                    filter_text.append(text)
            process_text = " ".join(filter_text)
        else:
            process_text = df['judgement'][idx]

        process_text = " ".join(process_text.split())

        if '簡' in df['no'][idx]:
            raise NotImplementedError
        else:
            # truth_condition = '(([ 、]+((犯罪)*事實(及(理由|證據)+)*|(事[ ]+實))[： ]+)(.*|\n*|\r*)([、。 ]+理[ ]*由[： ]+)|([）、。 ]+.*[證依][ ]*據.*[，： ]+)|([、。 ]+程[ ]*序.*[： ]+)|([。、 ]+論[ ]*罪([ ]*科[ ]*刑)*.*[： ]+)|([。、 ]+上開犯罪事實[，： ]+)|(處罰條文：[： ]+))|(([。、： ]+犯罪事實要旨[:： ]+)(.*|\n*|\r*)([。、： ]+處罰條文[:： ]+|法條[:： ]+))'
            head_condition_order = ['[、。 ]{1}(((犯罪)*事實(要旨)*)|(事[ ]+實))[，、。： ]{1}',
                                    '[、。 ]{1}(((本件){0,1}(犯罪)*事實(及(理由|證據){1})*))[，、。： ]{1}']
            middle_condition = '(.*|\\n*|\\r*)'
            tail_condition_order = ['[）、。 ]{1}理[ ]*由[，:： ]{1}',
                                    '[）、。 ]{1}論罪(科刑)*[，:： ]{1}',
                                    '[）、。 ]{1}論罪(科刑)之理由*[，:： ]{1}',
                                    '[）、。 ]{1}處罰條文[，:： ]{1}',
                                    '[）、。 ]{1}.{0,5}法條[，:： ]{1}',
                                    '[）、。 ]{1}[證依][ ]*據.{0,3}[，:： ]{1}',
                                    '[）、。 ]{1}證據部分',
                                    '[）、。 ]{1}本件犯罪事實及證據[，:： ]{1}',
                                    '偵查起訴[。，:： ]{1}',
                                    '[）、。 ]{1}二[、，:： ]{1}']

            for head_id, head_condition in enumerate(head_condition_order):
                break_flag = 0
                for tail_id, tail_condition in enumerate(tail_condition_order):
                    if head_id != 1 and tail_id == 9:
                        continue
                    truth_condition = head_condition + middle_condition + tail_condition
                    # print(truth_condition)
                    search_condition = re.search(truth_condition, process_text)

                    if search_condition is not None:
                        break_flag = 1
                        break
                if break_flag:
                    break

            if search_condition is None:
                if idx in special_idx:
                    match = process_text
                else:
                    match = process_text
                    not_match_count += 1
            else:
                match = process_text[search_condition.start():search_condition.end()]

        truth_all.append(match)

    df['truth'] = truth_all
    print(len(set(special_idx)))
    df = df.loc[~df.index.isin(special_idx)].reset_index(drop=True)
    df.to_csv('new_truth_data.csv', index=False)
    print("Not match count: {}".format(not_match_count))


if __name__ == '__main__':
    # process_file(sys.argv[1])
    # concat_file(sys.argv[1])
    # filter_data(sys.argv[1])
    # filter_withdraw_penalty(sys.argv[1])
    # filter_penalty_lawyer(sys.argv[1])
    filter_truth(sys.argv[1])
    # EDA(sys.argv[1])