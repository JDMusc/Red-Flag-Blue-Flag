"""
Find patient name to measure overfit
"""
from toolz import pipe as p

import numpy as np
import torch
import tqdm


def extract_pat_name(tokenized_text, locs_flat, flr, n_terms):
    """
    return patient name ngram as single string for filter
    """
    tokens = tokenized_text[locs_flat[flr]:locs_flat[flr]+n_terms]

    has_pat_name = np.all(tokens[:2] == ['patient', 'name'])
    n_unk = sum(tokens == 'xxunk')
    n_mrn = 'mrn' in tokens
    n_not_name = n_unk + n_mrn + 2
    n_name = n_terms - n_not_name
    is_valid = has_pat_name and n_name > 1

    return '-*-'.join(tokens), is_valid


def sample_ix_to_pat_txt(sample_ix, dataset, num, cnn_logit, flr, n_terms):
    """
    return joined tokens for a patient
    """
    return p(
        sample_to_extracts(sample_ix, dataset = dataset, num = num, cnn_logit = cnn_logit),
        lambda _: extract_pat_name(_[0], _[1], flr = flr, n_terms = n_terms)
    )


def subset_to_pat_txts(dataset, num, cnn_logit, flr, n_terms,
                        progress_fn = tqdm.tqdm):
    """
    return joined tokens for all patients
    """
    txts = [
        sample_ix_to_pat_txt(_, dataset, num, cnn_logit, flr, n_terms)
        for _ in progress_fn(range(len(dataset)))
    ]
    pat_txts = [txt for txt, iv in txts if iv]

    return txts, pat_txts


def calc_ratio_in_train(train_txts_pat_set, dataset, num,
                        cnn_logit, flr, n_terms):
    """
    Calculate train overlap for samples in dataset
    """
    _, val_txts = subset_to_pat_txts(dataset, num, cnn_logit, flr, n_terms)
    overlap_txts = [v for v in val_txts if v in train_txts_pat_set]

    n_val = len(val_txts)
    n_overlap = len(overlap_txts)

    return overlap_txts, len(overlap_txts), len(val_txts), n_overlap/n_val


def create_overlap_splits(splits, n_overlap, dst_subset = 2, src_subset = 0):
    """
    simulate splits
    """
    splits_a = np.array([np.array(_) for _ in splits], dtype=object)

    dst_indexes = np.random.randint(0, len(splits[dst_subset]) - 1, n_overlap)
    src_indexes = np.random.randint(0, len(splits[src_subset]) - 1, n_overlap)

    splits_a[dst_subset][dst_indexes] = splits_a[src_subset][src_indexes]

    return [s for s in splits_a]


def sample_to_extracts(sample_ix, dataset, num, cnn_logit):
    """
    Extract locations that pass max pool for sample ix
    """
    sample = dataset[sample_ix][0]
    locs = cnn_logit(torch.tensor(sample))[0]
    numel = locs.numel()
    locs_flat = locs.transpose(1, 0).reshape(numel, 1).numpy().flatten()

    return np.array(num.vocab)[sample], locs_flat
