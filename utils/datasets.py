from scipy.io import arff
import pandas as pd


def _get_X_y(df, dummies):
    # Guess which column represents the class
    if 'class' in df.columns:
        clss = 'class'
    elif 'Class' in df.columns:
        clss = 'Class'
    elif 'CLASS' in df.columns:
        clss = 'CLASS'
    elif 'y' in df.columns:
        clss = 'y'
    else:
        clss = df.columns[-1]

    if dummies:
        X = pd.get_dummies(df.drop(columns=clss)).values
    else:
        X = df.drop(columns=clss).values
    y = df[clss].values

    return X, y


def load_arff(path, dummies):
    data, metadata = arff.loadarff(path)
    df = pd.DataFrame(data)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.dropna(axis=1, inplace=True)
    for col in df.columns:
        if type(df[col][0]) is bytes:
            df[col] = df[col].str.decode("utf-8")

    return _get_X_y(df, dummies)


def load_csv(path, dummies):
    df = pd.read_csv(path)
    return _get_X_y(df, dummies)