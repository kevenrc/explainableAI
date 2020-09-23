from fastai import *
from fastai.tabular import *
import numpy as np
import pandas as pd

train = pd.read_csv('data/preprocessed/train.csv')
test = pd.read_csv('data/preprocessed/test.csv')

variable_names = list(train.columns[1:])

cont_names = variable_names

procs = ["FillMissing", "Categorify", "Normalize"]

test = TabularList.from_df(raw_data, cat_names=[], cont_names=cont_names, procs=procs)

data = (TabularList.from_df(train, path='.', cat_names=[], cont_names=cont_names, procs=procs).split_by_idx(list(range(0,200)))
        .label_from_df(cols=dep_var)
        .add_test(test, label=0)
        .databunch())

data.show_batch(rows=10)
