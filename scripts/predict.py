import os
import pandas as pd
import torch
from scripts.model import SimpleCNN


def predict_dummy(test_csv='data/raw/test.csv', out_csv='submission.csv'):
    # loads test ids and outputs random/class-uniform predictions (placeholder)
    test = pd.read_csv(test_csv)
    genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
    preds = [genres[i % len(genres)] for i in range(len(test))]
    sub = pd.DataFrame({'id': test['id'], 'genre': preds})
    sub.to_csv(out_csv, index=False)
    print('Wrote', out_csv)


if __name__ == '__main__':
    if not os.path.exists('data/raw/test.csv'):
        print('Place test.csv at data/raw/test.csv or change the path')
    else:
        predict_dummy()
