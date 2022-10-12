import os
import numpy as np
import pandas as pd
from dateutil import parser as date_parser
from dj_utils.utils import safe_mkdir
from sklearn.preprocessing import LabelEncoder

data_dir = "data/"
safe_mkdir(data_dir)


def get_csv_from_url(url, names=None, delimiter=","):
    return pd.read_csv(url, names=names, delimiter=delimiter)


class Dataset:
    def __init__(self, name, data, target_col, protected_col=None, encoding=None):
        """

        :param name:
        :param data:
        :param target_col:
        :param protected_col:
        :param encoding: None or a dictionary of (col_name: fit encoding sklearn class)
        """
        self.name = name
        self.data = data
        self.target_col = target_col
        self.protected_col = protected_col
        self.X_func = lambda x: self.data.drop(self.target_col, axis=1)
        self.y_func = lambda y: self.data[self.target_col]
        self.length = len(self.data)
        self.encoding = encoding
        self.encode_cats()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.X_func(None).iloc[i], self.y_func(None).iloc[i]

    def encode_cats(self):
        if self.encoding is None:
            self.encoding = {}
        cat_cols = set(self.data.select_dtypes(include=["category"]).columns.values).union(set(self.encoding.keys()))
        for cat_col in cat_cols:
            if cat_col not in self.encoding:
                le = LabelEncoder()
                self.data[cat_col] = le.fit_transform(self.data[cat_col])
                self.encoding[cat_col] = le
            else:
                encoder = self.encoding[cat_col]
                self.data[cat_col] = encoder.transform(self.data[cat_col])

    def decode_cats(self, subset=None):
        if self.encoding is None:
            return
        for cat_col in self.encoding:
            if subset is None or cat_col in subset:
                self.data[cat_col] = self.encoding[cat_col].inverse_transform(self.data[cat_col])

    def get_decoded_col(self, arr, col):
        if col in self.encoding:
            return self.encoding[col].inverse_transform(arr)
        else:
            raise ValueError

    def get_split_data(self, train_size=0.8):
        assert 0.01 < train_size < 0.99
        total = len(self)
        train_len = int(total*train_size)
        shuffled = self.data.sample(frac=1).reset_index()
        X, y = shuffled.drop(self.target_col, axis=1), shuffled[self.target_col]
        return X[:train_len], y[:train_len], X[train_len:], y[train_len:] 
        
    @staticmethod
    def data_generator(X, y, batch_size):
        i=0
        while i<len(y)-1:
            yield X[i:i+batch_size], y[i:i+batch_size]
            i += batch_size

    def get_simple_generators(self, train_size=0.8, batch_size=None, val_batch_size=None):
        if batch_size is None:
            batch_size = len(self)
        if val_batch_size is None:
            val_batch_size = len(self)
        X_train, y_train, X_val, y_val = self.get_split_data(train_size=train_size)
        return Dataset.data_generator(X_train, y_train, batch_size=batch_size), Dataset.data_generator(
            X_val, y_val, batch_size=val_batch_size)

    def get_cv_generators(self, batch_size=None, folds=5, val_batch_size=None, train_size=None):
        """
        Will return a generator object cv_generator
        cv_generator yeilds a generator train_gen and generator val_gen every loop
        train_gen yeilds X_train_batch, y_train_batch every loop same for val_gen

        if train_size is not None, then folds must be None, in which case folds is automatically calculated
        """
        assert folds is None or train_size is None
        if train_size is not None:
            assert 0.1 < train_size < 0.9 # I'm assuming folds > 10 will kill the system
            folds = int(1/(1-train_size))
        if batch_size is None:
            batch_size = len(self)
        if val_batch_size is None:
            val_batch_size = len(self)
        step_size = len(self)//folds
        points = [0+i*step_size for i in range(folds)]

        def cv_generator():
            for i in range(folds):
                start = points[i]
                if i == folds-1:
                    end = len(self.data)
                else:
                    end = points[i+1]
                val_data = self.data[start:end]
                train_data = self.data.drop(index=range(start, end))
                X_train = train_data.drop(self.target_col, axis=1)
                y_train = train_data[self.target_col]
                X_val = val_data.drop(self.target_col, axis=1)
                y_val = val_data[self.target_col]
                yield Dataset.data_generator(X_train, y_train, batch_size=batch_size), Dataset.data_generator(X_val, y_val, batch_size=val_batch_size)
        return cv_generator()


def get_dataset(name):
    assert name in ["propublica compas", "compas", "adults", "german", "hmda"]
    if name.lower() in ["propublica compas", "compas"]:
        return get_compas()
    elif name.lower() in ["adults"]:
        return get_adults()
    elif name.lower() in ["german"]:
        return get_german()
    elif name.lower() in ["hmda"]:
        return get_hmda()


def preprocess(data, drop_cols=None, date_cols=None, cat_cols=None, float_cols=None, auto_detect_date=True, numerical_to_float=True, below_threshold_other=0.95, few_unique_to_cat=10, drop_na_col_perc=0.4, dropna=True):
    if drop_cols is not None:
        data.drop(drop_cols, axis=1, inplace=True)
    if date_cols is not None:
        for date_col in date_cols:
            data[date_col] = pd.to_datetime(data[date_col])
    if auto_detect_date:
        cols = list(data.select_dtypes(include=["object"]).columns.values)
        for col in cols:
            try:
                data[col] = data[col].map(date_parser.parse)
            except:
                pass
    if cat_cols is not None:
        for cat_col in cat_cols:
            data[cat_col] = data[cat_col].astype("category")
    if float_cols is not None:
        for float_col in float_cols:
            data[float_col] = data[float_col].astype(float)
    if few_unique_to_cat > 0:
        for col in data:
            if len(data[col].unique()) <= few_unique_to_cat:
                data[col] = data[col].astype("category")
    if 1 > below_threshold_other > 0:
        cols = list(data.select_dtypes(include=["object", "category"]).columns.values)
        for col in cols:
            percs = data[col].value_counts() / len(data[col])
            top_density = 0
            trigger = False
            for i in range(len(percs)):
                if top_density <= below_threshold_other:
                    top_density += percs[i]
                else:
                    data[col] = data[col].astype("object")
                    data[col][data[col] == percs.index[i]] = "UNLIKELYCAT"
                    trigger = True
            if trigger:
                data[col] = data[col].astype("category")

    if numerical_to_float:
        num_cols = list(data.select_dtypes(include=[np.number]).columns.values)
        for col in num_cols:
            data[col] = data[col].astype("float32")
    nas = data.isna().mean()
    for col in data:
        if nas[col] >= drop_na_col_perc:
            data.drop(col, axis=1, inplace=True)
    if dropna:
        data.dropna(inplace=True)


def get_compas():
    data = get_csv_from_url("https://raw.githubusercontent.com/propublica/"
                            "compas-analysis/master/compas-scores-two-years.csv")
    drop_cols = ["id", "name", "first", "last", "c_case_number"]
    preprocess(data, drop_cols=drop_cols)
    d = Dataset(name="COMPAS", data=data, target_col="two_year_recid", protected_col="race")
    return d


def get_adults():
    names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
             "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "target"]
    data = get_csv_from_url("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", names=names)
    drop_cols = []
    preprocess(data, drop_cols=drop_cols)
    print(data.columns)
    d = Dataset(name="Adults", data=data, target_col="target", protected_col="race")
    return d


def get_german():
    names = ["c_status", "duration", "c_history", "purpose", "credit_amnt", "savings_account/bonds", "employment_since",
             "installment_rate", "personal_status", "other_debtors", "residence_since", "property", "age",
             "other_plans", "hosuing", "ncredits", "job", "ndependants", "telephone", "foreign", "target"]
    data = get_csv_from_url("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
                            names=names, delimiter=" ")
    drop_cols = []
    preprocess(data, drop_cols=drop_cols)
    d = Dataset(name="Adults", data=data, target_col="target", protected_col="race")
    return d


def get_hmda():
    data = pd.read_csv(data_dir+"/ny_hmda_2015.csv")
    filter1 = (data['action_taken'] >= 1) & (data['action_taken'] <= 3)
    data = data[filter1]
    drop_cols = ["action_taken_name", "agency_name", "state_name", "applicant_race_1"]
    preprocess(data, drop_cols=drop_cols)
    d = Dataset(name="HMDA", data=data, target_col="action_taken", protected_col="applicant_race_name_1")
    # You can get this from: https://www.kaggle.com/datasets/jboysen/ny-home-mortgage
    return d




