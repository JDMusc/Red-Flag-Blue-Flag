import os

import numpy as np
import pandas as pd
import torch
import tqdm

import dplyr_lite as d


def get_sample_activations_gen(scan_cnn, cnn_logit, dataset,
    ixs_to_tokens_fn):

    ixs_to_tokens_fn_arr = lambda _: d.p(_, ixs_to_tokens_fn, np.array)
    np_wghts = lambda _: (
        scan_cnn
        .linear
        .weight[_, :].detach().cpu().numpy()
        .reshape((-1, 1))
    )

    out_dim = scan_cnn.linear.weight.shape[0]

    if out_dim == 2:
        neg_weights = np_wghts(0)
        pos_weights = np_wghts(1)
        net_weights = pos_weights - neg_weights
    else:
        net_weights = np_wghts(0)

    get_sample = lambda ix: dataset[ix][0]

    def get_sample_activations(sample_ix, include_neg_pos = False, kernel_sizes = [1, 2, 3, 5]):
        sample = get_sample(sample_ix)
        locs, activs = cnn_logit(torch.tensor(sample))
        tokenized_txt = ixs_to_tokens_fn_arr(sample)

        numel = activs.numel()
        activs_flat = activs.transpose(1, 0).reshape(numel, 1).detach().cpu().numpy()
        locs_flat = locs.transpose(1, 0).reshape(numel, 1).detach().cpu().numpy()

        ngrams = []
        for kernel_size in kernel_sizes:
            ngrams = ngrams + pull_n_grams(tokenized_txt, kernel_size, locs)

        return d.p(
            activs_flat,
            d.curry(pd.DataFrame, columns = ['activ']),
            d.mutate(
                location = locs_flat,
                ngram = ngrams,
                net_weight = net_weights,
                net_logit = net_weights * activs_flat,
                k = [len(_) for _ in ngrams]
            ),
            (
                d.mutate(
                    neg_weight = neg_weights,
                    pos_weight = pos_weights,
                    neg_logit = neg_weights * activs_flat,
                    pos_logit = pos_weights * activs_flat,
                )
                if include_neg_pos
                else d.identity()
            ),
        ), create_loc_term_lo(tokenized_txt, locs.detach().numpy(), kernel_sizes=kernel_sizes)

    def check_logits(sample_ix, sample_activs_df = None):
        sample = get_sample(sample_ix)
        bias = scan_cnn.linear.bias.detach()

        if sample_activs_df is None:
            sample_activs_df = get_sample_activations(sample_ix)[0]
        mtd_score:float = d.p(
            sample_activs_df,
            d.select(['net_logit']),#['neg_logit', 'pos_logit']),
            d.apply('sum'),
            lambda _: _[0] + (
                bias[1] - bias[0] if out_dim == 2
                else bias[0]
            )
        )

        mtd_p1 = 1/(1 + np.exp(-mtd_score))

        nn_scores = scan_cnn(torch.tensor(sample)).detach().squeeze()
        denom = 1 if out_dim == 2 else 2
        nn_scores = nn_scores/denom

        return (
            mtd_score,
            mtd_p1,
            nn_scores,
            nn_scores[1] - nn_scores[0],
            torch.nn.Softmax(dim = 0)(nn_scores)
        )

    def _create_loc_term_lo_ngram(tokenized_txt, n_gram, locs):
        col_ix = n_gram -1 if n_gram < 5 else 3
        n_gram_locs = locs[:, col_ix]
        loc_to_term_lo = dict()
        for term_ix in range(0, n_gram):
            curr_locs = n_gram_locs + term_ix
            curr = dict(zip(curr_locs, tokenized_txt[curr_locs]))
            loc_to_term_lo = {**loc_to_term_lo, **curr}

        return loc_to_term_lo

    def create_loc_term_lo(tokenized_txt, locs, kernel_sizes = [1, 2, 3, 5]):
        loc_to_ngram_lo = lambda ng: _create_loc_term_lo_ngram(tokenized_txt, ng, locs)
        ret = dict()
        for kernel_size in kernel_sizes:
            ret.update(loc_to_ngram_lo(kernel_size))
        return ret

    return get_sample_activations, check_logits


def compose_mwcu(sample_activs_df, loc_to_term_lo):
    locations_df = d.p(
        sample_activs_df,
        d.mutate(locations = (
            ['location', 'k'],
            lambda item: {item.location + i for i in range(item.k)}
        ),
            engulfs = None,
            engulfed_by = None,
            is_mwcu = False,
            total_locations = lambda _: _.locations.values,
            total_logit = lambda _: _.net_logit.values,
            total_ngram = lambda _: _.ngram.values,
        ),
        d.arrange(['k', 'location']),
        d.apply('reset_index'),
        d.rename(index = 'orig_index'),
    )

    n_activs = len(sample_activs_df)

    location_states = {
        i: locations_df.iloc[i, :].to_dict()
        for i in range(n_activs)
    }

    for i in range(1, n_activs):
        start_state = location_states[i]
        for j in range(i-1, -1, -1):
            cand_state = location_states[j]
            cand_locs = cand_state['locations']
            if not cand_state['engulfed_by'] is None:
                continue

            has_overlap = len(start_state['locations'].intersection(cand_locs)) > 0

            if has_overlap:
                cand_state['engulfed_by'] = i

                engulfs = start_state['engulfs']
                if engulfs is None:
                    engulfs = set()
                engulfs.add(j)
                start_state['engulfs'] = engulfs

                start_state['total_logit'] = (
                    start_state['total_logit'] + cand_state['total_logit']
                )
                start_state['total_locations'] = start_state['total_locations'].union(
                    cand_state['total_locations']
                )

    for i in range(n_activs):
        curr_state = location_states[i]
        locations_df.engulfed_by.values[i] = curr_state['engulfed_by']
        locations_df.engulfs.values[i] = curr_state['engulfs']
        locations_df.is_mwcu.values[i] = curr_state['engulfed_by'] is None
        locations_df.total_logit.values[i] = curr_state['total_logit']
        locations_df.total_locations.values[i] = curr_state['total_locations']

    return d.p(
        locations_df,
        d.mutate(mwcu = (
            'total_locations',
            lambda tl: [loc_to_term_lo[list(tl)[i]] for i in range(len(tl))]
        ))
    ), location_states


def verify_mwcus(sample_activs_df, loc_to_term_lo):
    return (
        sample_activs_df.net_logit.sum(),
        d.p(
            compose_mwcu(sample_activs_df, loc_to_term_lo)[0],
            d.filter(is_mwcu = True),
            lambda _: _.total_logit.sum()
        )
    )


def pull_n_grams(tokenized_txt, n_gram, locs):
    col_ix = n_gram -1 if n_gram < 5 else 3
    n_gram_locs = locs[:, col_ix]
    n_grams = None
    for term_ix in range(0, n_gram):
        terms = [(_,) for _ in tokenized_txt[n_gram_locs + term_ix]]
        if n_grams is None:
            n_grams = terms
        else:
            n_grams = [ng + t for ng,t in zip(n_grams, terms)]
    return n_grams


def sample_to_mwl(sample_ix, simple_cnn, cnn_logit, dataset, ixs_to_tokens_fn):
    get_decom = \
        get_sample_activations_gen(simple_cnn, cnn_logit, dataset, ixs_to_tokens_fn)[0]

    sample_activs_df, loc_to_term_lo = get_decom(sample_ix)

    return compose_mwcu(sample_activs_df, loc_to_term_lo)[0]


def write_all_mwls(simple_cnn, cnn_logit, dataset, ixs_to_tokens_fn, dst_dir):
    n_samples = len(dataset)

    nm_to_pkl = lambda nm: os.path.join(dst_dir, f'{nm}.pkl')

    mwl_df = pd.DataFrame()
    for sample_ix in tqdm.trange(n_samples):
        sample_df = d.p(
            sample_to_mwl(sample_ix, simple_cnn, cnn_logit,
                    dataset, ixs_to_tokens_fn),
            d.mutate(sample_ix = sample_ix)
        )
        mwl_df = pd.concat([mwl_df, sample_df])

    mwl_df.to_pickle(nm_to_pkl('mwl'))


def mwl_to_ira(df, k = 10, kx = 10):
    cases = d.p(df,
        d.distinct('sample_ix'),
        d.attr('values'),
        d.apply('flatten')
    )

    ngram_to_rank = dict()
    for case in tqdm.trange(cases):
        c_df = d.p(
            df,
            d.filter(lambda _: _.sample_ix == case),
            d.select(['sample_ix', 'net_logit', 'ngram']),
            d.mutate(net_logit_abs = lambda _: _.net_logit.abs()),
            d.arrange('net_logit_abs', ascending = False),
            d.head(n = k),
            d.mutate(rank = lambda _: range(1, len(_) + 1)),
            d.filter(lambda _: _.net_logit > 0)
        )
        for tup in c_df.itertuples():
            if tup.ngram not in ngram_to_rank:
                ngram_to_rank[tup.ngram] = []

            ngram_to_rank[tup.ngram].append(tup.rank)

    return d.p(
        pd.DataFrame({
            ng: sum([1/v0 for v0 in v])/(len(v) + k*kx)
            for ng,v in ngram_to_rank.items()}.items(),
            columns = ['ngram', 'IRA']
        ),
        d.arrange('IRA', ascending = False)
    )
