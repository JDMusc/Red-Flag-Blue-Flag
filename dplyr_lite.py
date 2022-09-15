from collections.abc import Iterable
from inspect import signature
import operator as o
from toolz import pipe as p

import numpy as np
import pandas as pd


def series_filter(fn):
    return lambda arr: arr[arr.map(fn)]

def filter(*args, **kwargs):
    is_df_fn = len(args) > 0
    are_cols_fns = len(kwargs) > 0
    assert is_df_fn | are_cols_fns
    assert not (is_df_fn & are_cols_fns) 

    def ret(df:pd.DataFrame):
        if is_df_fn:
            fn = args[0]
            ixs = fn(df, *args[1:], **kwargs)
        if are_cols_fns:
            ixs = np.repeat(True, df.shape[0])
            for (col, fn) in kwargs.items():
                if callable(fn):
                    ixs = ixs & [fn(_) for _  in df[col]]
                else:
                    ixs = ixs & [_ == fn for _ in df[col]]

        return df.loc[ixs]
    
    return ret


def filter_index(*args, index_col = '_index', **kwargs):
    return p_fn(
        mutate(**{index_col: lambda _: _.index}),
        filter(*args, **kwargs),
        exclude(index_col)
    )


def select(col):
    col = iterize(col)
    return lambda df: df[col]


def select_fn(fn):
    return lambda df: df[[c for c in df.columns if fn(c)]]


def pull(col):
    selector = select(col)
    
    return lambda df: p(
        df,
        selector,
        lambda _: _.values
    )

def exclude(col):
    col = iterize(col)

    def fn(df):
        tmp = df.copy()
        for co in col:
            del tmp[co]
        return tmp
    
    return fn

def select_ixs(ixs):
    return lambda df: df.iloc[ixs]

def apply(fn, *args, **kwargs):
    if type(fn) is str:
        return lambda df: getattr(df, fn)(*args, **kwargs)
    return lambda df: fn(df, *args, **kwargs)


def apply_return(fn_, *args, **kwargs):
    
    fn = apply(fn_, *args, **kwargs)

    def ret(df):
        fn(df)
        return df
    
    return ret


curry = apply

def groupby(col):
    return lambda df: df.groupby(by = col)


def distinct(col = None):
    selector = identity() if col is None else select(col)
    return lambda df: selector(df).drop_duplicates()


def distinct_fn(fn):
    return p_fn(
        select_fn(fn),
        distinct()
    )

def identity():
    return lambda _: _

def count(col, reset_index = True):
    n_col = col if type(col) is str else col[0]
    return lambda df: p(
        df,
        apply('groupby', by = col),
        apply('agg', n = (n_col, len)),
        apply('reset_index') if reset_index else identity()
    )


def arrange(col, **kwargs):
    return lambda df: df.sort_values(col, **kwargs)


def attr(a): return lambda df: getattr(df, a)


def group_mutate(**kwargs):
    assert(len(kwargs) == 1)
    new_col = list(kwargs.keys())[0]
    group_col, mutate_fn = list(kwargs.values())[0]

    def mutate_fn_f(row, ix, df): 
        return p(
            df,
            filter(**{group_col: row[group_col]}),
            lambda f_df: mutate_fn(row, ix, f_df)
        )

    def ret(df):
        cols = df.columns
        mutate_input = {new_col: (cols, mutate_fn_f)}
        return p(
            df,
            mutate(**mutate_input)
        )
    
    return ret


def mutate(**kwargs):
    def ret(df):
        df_ret = df.copy()
        for col, rhs in kwargs.items():
            if callable(rhs):
                df_ret[col] = rhs(df_ret)
            elif type(rhs) is tuple:
                input_col, item_fn = rhs[0:2]

                are_multiple_cols = (
                    isinstance(input_col, Iterable) 
                    and not type(input_col) is str 
                )

                if not callable(item_fn):
                    assert not are_multiple_cols
                    
                    df_ret[col] = df_ret[input_col] == item_fn
                    continue

                sig = signature(item_fn)
                n_params = len(sig.parameters)

                def get_val(ix, item):
                    if n_params == 1:
                        return item_fn(item)
                    elif n_params == 2:
                        return item_fn(item, ix)
                    else:
                        return item_fn(item, ix, df)

                if are_multiple_cols:
                    source = df_ret[input_col].iterrows()
                else:
                    source = enumerate(df_ret[input_col])

                df_ret[col] = [get_val(*_) for _ in source]
            else:
                df_ret[col] = rhs

        return df_ret
    
    return ret


def rename(**kwargs):
    def ret_fn(df:pd.DataFrame):
        return df.rename(mapper = kwargs, axis = 1)
    
    return ret_fn


def summarize(**kwargs):
    return lambda _: _.agg(**kwargs)


def head(n = 5):
    return lambda _: _.head(n)


def tail(n = 5):
    return lambda _: _.tail(n)


def shape():
    return attr('shape')


def wide_to_long(id_vars=None, value_vars=None, var_name=None, value_name='value', 
                col_level=None):
    return curry(pd.melt, id_vars = id_vars, value_vars = value_vars,
                var_name = var_name, value_name = value_name, col_level = col_level)


def long_to_wide(index = None, columns = None, values = None):
    return curry(pd.pivot, index = index, columns = columns, values = values)

def sample(*args, **kwargs):
    return lambda _: _.sample(*args, **kwargs)


def append_row(fn):
    def ret(df):
        df.loc[len(df.index)] = fn(df)
        return df
    
    return ret


def if_else(condition_fn, val1, val2):
    if not callable(condition_fn):
        condition_fn = p_fn(
            pull(condition_fn),
            apply('flatten'),
        )

    def ret_fn(df):
        val1_ixs = condition_fn(df)
        ret = np.repeat(val2, len(val1_ixs))
        ret[val1_ixs] = val1
        return ret

    return ret_fn


class _Value():
    @staticmethod
    def __gt__(ref):
        return curry(o.gt, ref)
    
    @staticmethod
    def __lt__(ref):
        return curry(o.lt, ref)
    
    @staticmethod
    def __eq__(ref):
        return curry(o.eq, ref)
    
    @staticmethod
    def __le__(ref):
        return curry(o.le, ref)
    
    @staticmethod
    def __ge__(ref):
        return curry(o.ge, ref)
    
    @staticmethod
    def __ne__(ref):
        return curry(o.ne, ref)
    
    @staticmethod
    def __invert__():
        return lambda _: not _
    
    @staticmethod
    def __mul__(other):
        return curry(o.mul, other)
    
    @staticmethod
    def __truediv__(denom):
        return lambda num: num/denom
    
    @staticmethod
    def __add__(other):
        return lambda _ : _ + other
    
    @staticmethod
    def __sub__(right):
        return lambda _: _ - right
    
    @staticmethod #__contains__
    def In(items):
        return lambda _: _ in items

    @staticmethod
    def NotIn(items):
        return lambda _: _ not in items
    
    @staticmethod
    def Has(item):
        return lambda _: item in _
    
    @staticmethod
    def NotHave(item):
        return lambda _: item not in _

value = _Value()


class col():
    def __init__(self, _col):
        self._col = _col
    
    def __gt__(self, ref):
        return lambda df: df[self._col] > ref
    
    def __lt__(self, ref):
        return lambda df: df[self._col] < ref
    
    def __eq__(self, ref):
        return lambda df: df[self._col] == ref
    
    def __le__(self, ref):
        return lambda df: df[self._col] <= ref
    
    def __ge__(self, ref):
        return lambda df: df[self._col] >= ref
    
    def __ne__(self, ref):
        return lambda df: df[self._col] != ref 
    
    def __invert__(self):
        return lambda df: np.logical_not(df[self._col])
    
    def __mul__(self, other):
        return lambda df: df[self._col] * other
    
    def __truediv__(self, denom):
        return lambda df: df[self._col]/denom
    
    def __add__(self, other):
        return lambda df: df[self._col] + other
    
    def __sub__(self, right):
        return lambda df: df[self._col] - right


def mapCol(fn, col):
    return lambda df: [fn(_) for _ in df[col]] 

def iterize(_):
    return _ if _is_iterable(_) else [_]

def _is_iterable(_):
    return isinstance(_, Iterable) and not type(_) is str

def make_columns_df(df:pd.DataFrame):
    return p(
        df.columns,
        curry(pd.DataFrame, columns = ['name'])
    )


def to_set(col):
    if type(col) is list:
        assert len(col) == 1
        
    return p_fn(
        distinct(col),
        pull(col),
        apply('squeeze'),
        np.atleast_1d,
        set
    )

def to_lookup(key, values):
    keys = iterize(key)
    values = iterize(values)
    cols = keys + values

    key_fn = lambda item: (
        item[key] if not _is_iterable(key)
        else tuple(k for k in item[key])
    )
    
    return p_fn(
        distinct(cols),
        lambda df: {
            key_fn(item): {
                v: item[v]
                for v in values 
            } for _, item in df.iterrows()
        }
    )

def p_fn(*funcs):
    return lambda _: p(_, *funcs)


def get_item(k):
    return lambda _: _[k]


def branch(*args):
    return lambda _: tuple(
        arg(_) for arg in args
    )


side_effect = apply_return
