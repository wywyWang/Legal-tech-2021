import pandas as pd
import sys
import os
import json
from tqdm import tqdm


def get_filename(folder):
    """Get folder then concat all json files

    Args:
        folder (string): folder path
    """
    all_json = []
    json_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.json')]

    columns = {
        'court', 'date', 'no', 'sys', 'reason', 'judgement', 'attachAsJudgement', 'attachments', 'type', 'historyHash', 'mainText', 'opinion', 'relatedIssues', 'party'
    }

    for idx, json_path in tqdm(enumerate(json_paths), total=len(json_paths)):
        with open(json_path) as json_data:
            data = json.load(json_data)
        df = pd.DataFrame.from_dict(data, orient='index').T
        all_json.append(df)
    
    all_json = pd.concat(all_json, ignore_index=True, sort=False)
    all_json['idx'] = all_json.index
    all_json.to_csv('concat.csv', index=False)


if __name__ == '__main__':
    get_filename(sys.argv[1])