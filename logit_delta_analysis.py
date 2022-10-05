import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import tqdm

import dplyr_lite as d
import hyperparam_config as h_config
import local_datasets as lds
import models_pyt as mo
import note_cutoffs_config as ncc
import performance_script as ps
import utils as u
import red_flag_blue_flag as rfbf 


def get_ixs_below_delta_cutoff(outcome, run, cutoff, f_name = 'test'):
    zf:dict = rfbf.load_zf_dict(outcome, run, f_name)

    return [
        flr_ix for flr_ix, flr_info in zf.items()
        if flr_info['meta']['logit_delta'] < cutoff
    ]


def get_lowest_ix_delta(outcome, run,  f_name = 'test'):
    zf:dict = rfbf.load_zf_dict(outcome, run, f_name)

    log_ds = [
        flr_info['meta']['logit_delta']
        for _, flr_info in zf.items()
    ]

    ix = np.argmin(log_ds)
    return ix, log_ds[ix]


def get_lowest_ix_deltas(outcome, n_runs = 10):
    return d.p(
        [get_lowest_ix_delta(outcome, _) for _ in range(n_runs)],
        d.curry(pd.DataFrame, columns = ['flr', 'logit']),
        d.mutate(run = lambda _: _.index.values)
    )


def drop1_auc(outcome, run, subset_ix, y_label = None, progress_fn = d.identity()):
    lowest_ix, _ = get_lowest_ix_delta(outcome, run)
    gpc = get_performance_cutoff_gen(
        outcome, run, subset_ix, 
        y_label=y_label, progress_fn=progress_fn)[0]
    return gpc(cutoff_ixs = [lowest_ix])


def drop1_improvements(outcome, subset_ix, progress_fn=tqdm.tqdm):
    lowest_ix_deltas = get_lowest_ix_deltas(outcome)
    gpc_lo = {
        ix: get_performance_cutoff_gen(outcome, ix, subset_ix)
        for ix in progress_fn(range(len(lowest_ix_deltas)))
    }
    return d.p(
        lowest_ix_deltas,
        d.mutate(
            auc = ('run', lambda _: gpc_lo[_]()),
            auc_1 = (
                ['run', 'flr'], 
                lambda item: gpc_lo[item.run](cutoff_ixs = [item.flr])
            )
        ),
        d.mutate(auc_change = lambda _: _.auc_1 - _.auc)
    )


def get_performance_cutoff_gen(outcome, run, subset_ix, y_label = None,
                                progress_fn = d.identity()):
    if y_label is None:
        y_label = outcome

    num = rfbf.load_num(outcome)
    tokenizer = u.load_tokenizer()
    splits = ps.load_splits_list(outcome)[run]
    data = rfbf.load_df(outcome)
    datasets = lds.createDatasets(data, 'clean_txt', 'GRP', 
                    tokenizer=tokenizer, 
                    cutoff=ncc.note_cutoffs['OP NOTE'],
                    num=num, splits=splits)
    cnn_model = rfbf._load_cnn(outcome, run)

    subset = datasets.subset(subset_ix)

    logits = rfbf.get_dataset_logits(
        subset, 
        cnn_model, mo.CnnLogitExtracts(cnn_model), 
        num,
        progress_fn=progress_fn
    )

    y_true = np.array([_[1] == y_label for _ in progress_fn(subset)])

    all_ixs = set(range(len(logits)))

    def get_performance_cutoff(cutoff = None, cutoff_ixs = None, f_name = 'test'):
        if cutoff is not None and cutoff_ixs is None:
            cutoff_ixs = get_ixs_below_delta_cutoff(
                outcome, run, cutoff, f_name = f_name)
        
        if cutoff_ixs is not None:
            valid_ixs = d.p(
                cutoff_ixs,
                set,
                all_ixs.difference,
                list
            )
            
            logits_co = logits[valid_ixs, :]
        else:
            logits_co = logits
        
        return roc_auc_score(
            y_true,
            torch.tensor(logits_co.sum(axis = 0)).softmax(dim = -1)
        )
    
    return get_performance_cutoff, logits, y_true
