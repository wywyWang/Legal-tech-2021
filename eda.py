import datatable as dt
from tqdm import tqdm
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


def EDA(filename):
    df = dt.fread(filename).to_pandas()

    # print(df['reason'].value_counts(normalize=False))
    # print(df['type'].value_counts(normalize=True))
    # print(df['court'].value_counts(normalize=True))

    df['year'] = df['date'].apply(lambda x: x.split('-')[0])
    print(df['year'].value_counts(normalize=True))


if __name__ == '__main__':
    # process_file(sys.argv[1])
    # concat_file(sys.argv[1])
    EDA(sys.argv[1])