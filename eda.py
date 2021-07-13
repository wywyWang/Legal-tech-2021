import datatable as dt
import sys


def EDA(filename):
    df = dt.fread(filename).to_pandas()
    print(df.head())


if __name__ == '__main__':
    EDA(sys.argv[1])