"""
Calculates VGG dict
"""
from enum import Enum
from IPython.display import HTML
from toolz import pipe as p

import numpy as np
import pandas as pd
import seaborn as sbn
from sklearn.metrics import log_loss, roc_auc_score
import torch
import torch.nn.functional as F
import tqdm

import dplyr_lite as d
import ngram_analysis as nga
import models_pyt as mo


def plot_inner(model, layer_ix):
    """
    Plot correlations between weights of different layers
    """
    weights = model.convolutions[layer_ix].weight.squeeze().detach()

    is_n1 = weights.dim() == 2
    outer_dims = 1 if is_n1 else [1, 2]
    if is_n1:
        n_filters, embed_dim = weights.shape
        n_terms = 1
    else:
        n_filters, embed_dim, n_terms = weights.shape

    norms = torch.linalg.vector_norm(weights, dim = outer_dims)

    if is_n1:
        units = F.normalize(weights)
    else:
        units = weights/norms.repeat( (n_terms, embed_dim, 1) ).transpose(2, 0)

    units_2d = units.reshape(n_filters, embed_dim * n_terms)
    ret = p(
        units_2d,
        lambda _: torch.inner(_, _)
    )

    p(
        torch.inner(units_2d, units_2d).numpy(),
        pd.DataFrame,
        sbn.heatmap
    )

    return ret


default_flr_ix_to_layer = {
    i: (
        0 if i < 36
        else 1 if i < 72
        else 2 if i < 108
        else 3
    ) for i in range(144)
}

default_flr_ix_to_ngram = {
    i: (
        1 if i < 36
        else 2 if i < 72
        else 3 if i < 108
        else 5
    ) for i in range(144)
}


def get_flr_wghts(i, cnn_model,
                    flr_ix_to_layer = default_flr_ix_to_layer, flrs_per_layer = 36):
    """
    Return Filter Weights for CNN Model at a given filter
    """
    lyr = flr_ix_to_layer[i]
    offset = i - lyr * flrs_per_layer
    return cnn_model.convolutions[lyr].weight[offset].detach()


def get_flr_bias(i, cnn_model,
                flr_ix_to_layer = default_flr_ix_to_layer,
                flrs_per_layer = 36,
                ):
    """
    Return Filter Bias for CNN Model
    """
    lyr = flr_ix_to_layer[i]
    offset = i - lyr * flrs_per_layer
    if cnn_model.convolutions[lyr].bias is None:
        return 0
    return cnn_model.convolutions[lyr].bias[offset].detach()


def _initialize_vgg(cnn_model, n_show = 3, n_flrs = 144, flrs_per_layer = 36,
                    flr_ix_to_layer=default_flr_ix_to_layer):
    get_neg100 = lambda: np.repeat(-100., n_show)
    get_int0 = lambda: np.repeat(0, n_show)
    get_ngramtxt = lambda: ['ngram' for _ in range(n_show)]

    return {
        i: dict(
            meta = dict(
                fc = cnn_model.linear.weight.squeeze().detach()[i],
                w_norm = get_flr_wghts(i, cnn_model, flr_ix_to_layer, flrs_per_layer).norm(),
                w_bias = get_flr_bias(i, cnn_model, flr_ix_to_layer, flrs_per_layer),
            ),
            sample_info = dict(
                logit_mag = get_neg100(),
                net_logit = get_neg100(),
                ngram = get_ngramtxt(),
                x_norm = get_neg100(),
                angle = get_neg100(),
                e_dot_u = get_neg100(),
                logit_n = get_int0(),
                logit_n_controls = get_int0(),
                logit_n_patients = get_int0(),
            )
        )
        for i in range(n_flrs)
    }

def get_vgg_dict(cnn_model, dataset, numericize, ixs_to_tokens_fn,
                n_show = 3, remove_count = 0, n_flrs = 144,
                flr_ix_to_layer=default_flr_ix_to_layer,
                flrs_per_layer = 36, kernel_sizes = None,
                progress_fn = tqdm.tqdm,
                seq_len = 3000,
                patient_label = 1):
    """
    Return VGG Dictionary with a key for each filter
    """
    if kernel_sizes is None:
        kernel_sizes = [1, 2, 3, 5]

    cnn_xt_model = mo.CnnLogitExtracts(cnn_model, seq_length=seq_len)
    get_decomp_activ, _ = nga.get_sample_activations_gen(
        cnn_model, cnn_xt_model, dataset,
        ixs_to_tokens_fn=ixs_to_tokens_fn)

    ret = _initialize_vgg(cnn_model, n_show, n_flrs, flrs_per_layer)

    n_samples = len(dataset) - remove_count
    logit_vals = np.zeros( (n_flrs, n_samples) )
    join_tkn = '-*-'
    for sample_ix in progress_fn(range(n_samples)):
        activs, _ = get_decomp_activ(sample_ix, kernel_sizes=kernel_sizes)
        is_patient = dataset[sample_ix][1] == patient_label
        logit_n_key = f'logit_n_{"patients" if is_patient == 1 else "controls"}'

        for r_ix in range(len(activs)):
            net_logit = activs.net_logit[r_ix]
            logit_vals[r_ix, sample_ix] = net_logit
            logit_mag = np.abs(net_logit)
            ngram = activs.ngram[r_ix]
            ngram_s = join_tkn.join(ngram)
            s_i = ret[r_ix]['sample_info']

            if ngram_s in s_i['ngram']:
                loc = [_ for _, v in enumerate(s_i['ngram']) if v == ngram_s][0]
                s_i['logit_mag'][loc] = running_mean(
                    s_i['logit_n'][loc], s_i['logit_mag'][loc], logit_mag
                )
                s_i['net_logit'][loc] = running_mean(
                    s_i['logit_n'][loc], s_i['net_logit'][loc], net_logit
                )
                s_i['logit_n'][loc] = s_i['logit_n'][loc] + 1
                s_i[logit_n_key][loc] = s_i[logit_n_key][loc] + 1
                continue

            smaller_locs = np.nonzero(logit_mag > s_i['logit_mag'])[0]
            any_smaller = len(smaller_locs) > 0
            if any_smaller and not ngram_s in s_i['ngram']:
                loc = int(smaller_locs[0])
                s_i['logit_mag'][loc] = logit_mag
                s_i['net_logit'][loc] = net_logit
                s_i['ngram'][loc] = ngram_s
                s_i['logit_n'][loc] = 1
                s_i[logit_n_key][loc] = 1

    patient_ixs = [_[1] == patient_label for _ in dataset][:n_samples]

    for k in ret.keys():
        s_i = ret[k]['sample_info']
        for i in range(n_show):
            nums = numericize(s_i['ngram'][i].split(join_tkn))
            embed_weights = cnn_model.embed.weight[nums].detach().transpose(1,0)
            s_i['x_norm'][i] = torch.linalg.vector_norm(embed_weights)
            flr = get_flr_wghts(k, cnn_model, flr_ix_to_layer, flrs_per_layer)
            s_i['e_dot_u'][i] = cross_corr(
                embed_weights.reshape(-1), flr.reshape(-1)
            )
        s_i['angle'] = np.arccos(s_i['e_dot_u'])/np.pi * 180

        meta = ret[k]['meta']
        filter_logits = logit_vals[k, :]
        for (agg_fn_name, agg_fn) in [('median', np.median), ('mean', np.mean)]:
            plus_k = f'{agg_fn_name}_logit_+'
            minus_k = f'{agg_fn_name}_logit_-'
            meta[plus_k] = agg_fn(filter_logits[patient_ixs])
            meta[minus_k] =agg_fn(filter_logits[np.array(patient_ixs) == 0])
            meta[f'{agg_fn_name}_logit_delta'] = meta[plus_k] - meta[minus_k]

        meta['auc'] = roc_auc_score(patient_ixs, filter_logits)

    return ret


def get_dataset_logits(
                    dataset, cnn_model, cnn_xt_model, num,
                    remove_count = 0,
                    n_flrs = 144,
                    progress_fn = tqdm.tqdm):
    """
    Returns a matrix of number of filters by number of samples
    """
    get_decomp_activ, _ = nga.get_sample_activations_gen(
        cnn_model, cnn_xt_model, dataset, num)
    n_samples = len(dataset) - remove_count
    logit_vals = np.zeros( (n_flrs, n_samples) )
    for sample_ix in progress_fn(range(n_samples)):
        activs, _ = get_decomp_activ(sample_ix)

        net_logits = activs.net_logit.values
        logit_vals[:, sample_ix] = net_logits

    return logit_vals


def make_vgg_table(vgg_dict, rank_delta = False,
                    layer_lo = default_flr_ix_to_ngram.get, n_show = 3):
    """
    returns HTML table from a VGG dictionary
    """
    def get_median_logit(flr_ix):
        return np.median(vgg_dict[flr_ix]['sample_info']['net_logit'])

    def get_delta_logit(flr_ix):
        return vgg_dict[flr_ix]['meta']['logit_median_delta']

    n_filters = len(vgg_dict.keys())

    rank_fn = get_delta_logit if rank_delta else get_median_logit
    logit_mets = np.array([rank_fn(_) for _ in range(n_filters)])
    sort_args = np.argsort(logit_mets)
    half_flrs = int(n_filters/2)
    rankings = [
        (sort_args[_[0]], sort_args[_[1]])
        for _ in list(zip(range(n_filters - 1, half_flrs - 1, -1), range(half_flrs)))
    ]

    flr_ix_to_tbl = lambda flr_ix, rank: _vgg_item_to_tbl_row(flr_ix, rank, vgg_dict,
                                    layer_lo=layer_lo, n_show=n_show)

    return HTML(
        f"""
        <table>
            <th>
                <h3>+</h3>
            </th>
            <th>
                <h3>-</h3>
            </th>
            <tr>
            {''.join(
                [
                    f"<tr><td>{flr_ix_to_tbl(_[0], r)}</td><td>{flr_ix_to_tbl(_[1], r)}</td></tr>"
                    for r, _ in enumerate(rankings)
                ])
            }
            </tr>
        </table>
        """
    )


class RankMethod(Enum):
    """
    Options for ranking filter
    """
    MEAN_LOGIT_DELTA = 0
    MEDIAN_LOGIT_DELTA = 1
    AUC = 2
    MEDIAN_TOP_LOGITS = 3


class DisplayMetric(Enum):
    """
    Options for delta metric display
    """
    MEAN = 0
    MEDIAN = 1


class RankInput(Enum):
    """
    Options for VGG to rank
    """
    VGG_1 = 1
    VGG_2 = 2

def make_vggs_table(vgg_1, vgg_2,
                    title,
                    rank_method:RankMethod = RankMethod.MEDIAN_LOGIT_DELTA,
                    display_metric:DisplayMetric = DisplayMetric.MEDIAN,
                    flr = None,
                    layer_lo = default_flr_ix_to_ngram.get,
                    n_show = 3,
                    n_rows = None,
                    model1_name = 'Model 1',
                    model2_name = 'Model 2',
                    vgg_rank_input:RankInput = RankInput.VGG_1
                    ):
    """
    return HTML table of 2 columns for comparison of 2 VGG dicts
    """
    if flr is None:
        flr = lambda value1, value2, flr_ix: True

    def get_top_logits(vgg, flr_ix):
        return vgg[flr_ix]['sample_info']['net_logit'][:n_show]

    def get_median_top_logits(vgg, flr_ix):
        return np.median(get_top_logits(vgg, flr_ix))
    
    def get_meta_key_gen(k:str):
        def get_meta_key(vgg, flr_ix):
            return vgg[flr_ix]['meta'][k]
        return get_meta_key

    rank_by_vgg_1 = vgg_rank_input is RankInput.VGG_1
    rank_vgg = vgg_1 if rank_by_vgg_1 else vgg_2
    filter_ixs = d.p(
        rank_vgg,
        d.apply('keys'),
        list,
        np.array,
    )

    def get_rankings(vgg):
        if rank_method == RankMethod.MEDIAN_TOP_LOGITS:
            rank_fn = get_median_top_logits
        else:
            rank_fn = get_meta_key_gen(rank_method.name.lower())

        logit_mets = np.array([rank_fn(vgg, _) for _ in filter_ixs])

        return filter_ixs[np.flip(np.argsort(logit_mets))]

    rankings_1 = get_rankings(vgg_1)
    rankings_2 = get_rankings(vgg_2)
    rankings = (rankings_1 if rank_by_vgg_1 else rankings_2)
    if n_rows is not None:
        rankings = rankings[:n_rows]

    def flr_ix_to_cell(flr_ix, vgg_opt:int):
        if vgg_opt == 1:
            rank = np.argmax(rankings_1 == flr_ix)
            vgg = vgg_1
        elif vgg_opt == 2:
            rank = np.argmax(rankings_2 == flr_ix)
            vgg = vgg_2
        
        return _vgg_item_to_tbl_row(
            flr_ix, rank, vgg, layer_lo=layer_lo, n_show=n_show, display_metric=display_metric,
        )

    rank_ttl = (
        meta_key_to_ttl(rank_method.name) 
        if 'top' not in rank_method.name.lower()
        else f'Median of Top {n_show} Logits'
    )
    rank_title = f'Ranked by {rank_ttl} ({model1_name if rank_by_vgg_1 else model2_name} Set)'

    return HTML(
        f"""
        <table>
            <caption><h3>{title}, {rank_title}</h3></caption>
            <th>
                <h3 style="text-align:center">{model1_name}</h3>
            </th>
            <th>
                <h3 style="text-align:center">{model2_name}</h3>
            </th>
            <tr>
            {''.join(
                [
                    f"<tr><td>{flr_ix_to_cell(flr_ix, 1)}</td><td>{flr_ix_to_cell(flr_ix, 2)}</td></tr>"
                    for flr_ix in rankings if flr(vgg_1[flr_ix], vgg_2[flr_ix], flr_ix)
                ])
            }
            </tr>
        </table>
        """
    )

span_it = lambda txt, stl: f'<span style="{stl}">{txt}</span>'
BLUE = 'rgb(150, 75, 255)'
color_style = lambda c: f"color:{c}"
blue_style = color_style(BLUE)
red_style = color_style("red")
span_red = lambda txt: span_it(txt, red_style)
span_blue = lambda txt: span_it(txt, blue_style)

def _vgg_item_to_tbl_row(flr_ix, rank, vgg_dict,
                        layer_lo = default_flr_ix_to_ngram.get, n_show = 3,
                        display_metric=DisplayMetric.MEDIAN):
    s_i = vgg_dict[flr_ix]['sample_info']
    log_n_pats = lambda _: span_red(f'{s_i["logit_n_patients"][_]}x')
    log_n_cons = lambda _: span_blue(f'{s_i["logit_n_controls"][_]}x')
    sample_row = lambda _: f"""
        <tr><td>{s_i['ngram'][_]} {log_n_pats(_)}, {log_n_cons(_)}, 
        (
            {s_i['net_logit'][_].round(2)}, 
            {s_i['e_dot_u'][_].round(1)}, 
            {s_i['angle'][_].round(1)}°, 
            {s_i['x_norm'][_].round(1)}
        ) </td></tr>
    """

    meta = vgg_dict[flr_ix]['meta']
    imp_k = (meta['fc']*meta['w_norm'])
    sign = np.sign(meta['fc'])
    hdr_style= red_style if sign > 0 else blue_style

    imprt = span_it(f'Rank: {rank+1}, Imp.: {imp_k:.2f}, AUC: {meta["auc"]:.3f}', hdr_style)
    display_key = f'{display_metric.name.lower()}_logit'
    display_key_plus = f'{display_key}_+'
    display_key_minus = f'{display_key}_-'
    display_key_delta = f'{display_key}_delta'
    logit_plus = span_red(f"{meta[display_key_plus]:.2f}")
    logit_minus = span_blue(f"{meta[display_key_minus]:.2f}")

    if meta[display_key_minus] < 0:
        logit_minus = f'({logit_minus})'

    delta_sub = 'med' if 'median' in display_key else 'mean'
    meta_hdr = f"""
    <th>
        Ix: {flr_ix}, 
        {layer_lo(flr_ix)}g, 
        {imprt},
        Δ<sub>{delta_sub}</sub>={logit_plus}-{logit_minus}={meta[display_key_delta]:.3f},
        (Logit, •, Θ, ||x||)
    </th>
    """

    return f"""
    <table style="font-size: {12 + 12*np.abs(imp_k)}px">
        {meta_hdr}
        {''.join([sample_row(_) for _ in range(n_show)])}
    </table>
    """


def cross_corr(arr1, arr2):
    """
    Cross-Correlation
    """
    return F.normalize(arr1, dim = 0).dot(F.normalize(arr2, dim = 0))


def running_mean(old_count, old_mean, update):
    """
    Running Mean
    """
    return (old_count * old_mean + update)/(old_count + 1)


def sample_to_logits(sample_ix, dataset, cnn_logit):
    """
    Extract filter logit values for sample
    """
    sample = dataset[sample_ix][0]
    return make_logit_1d(cnn_logit(torch.tensor(sample))[1].detach())


def make_logit_1d(logs):
    """
    convert sample logits to flat array
    """
    return logs.transpose(1, 0).reshape(logs.numel(), 1).numpy().flatten()


def view_outliers(vgg_x, vgg_y, x_label = 'Train', y_label = 'Val', hue = None,
                    do_show_flr_ix = lambda x, y: x < 0 or y < 0,
                    meta_key = 'median_logit_delta',
                    meta_key_x = None,
                    meta_key_y = None,
                    ttl = None):
    """
    View Logit Delta plot
    """

    if meta_key_x is None: 
        meta_key_x = meta_key
    if meta_key_y is None: 
        meta_key_y = meta_key

    plot_df = d.p(
        vgg_x,
        d.curry(get_meta_key, meta_key_x),
        d.curry(pd.DataFrame, columns = [x_label]),
        d.mutate(**{y_label: get_meta_key(vgg_y, meta_key_y)}),
        d.mutate(x_lt0 = lambda _: _[x_label] < 0,
                y_lt0 = lambda _: _[y_label] <= 0),
    )
    plt = sbn.scatterplot(data = plot_df, x = x_label, y = y_label, hue = hue)

    x_values = plot_df[x_label].values
    y_values = plot_df[y_label].values
    for data_pt in range(len(plot_df)):
        if do_show_flr_ix(x_values[data_pt], y_values[data_pt]):
            plt.text(x_values[data_pt], y_values[data_pt], f'{data_pt}')
    
    if ttl is None:
        ttl = meta_key_to_ttl(meta_key)
    plt.set_title(ttl)

    return plt, d.p(plot_df, d.select([x_label, y_label]), d.apply('corr'))


def get_meta_key(vgg, key = 'median_logit_delta'):
    return [_['meta'][key] for _ in vgg.values()]


def get_preds(dataset, model, zero_ixs = None,
            progress_fn=tqdm.tqdm):
    """
    return predictions, can exclude (zero out) certain filters
    """
    n_samples = len(dataset)

    model.linear.weight.requires_grad = False

    if zero_ixs is not None:
        model.linear.weight[0][zero_ixs] = 0

    return [
        model(torch.tensor(dataset[_][0])).detach().softmax(dim = -1).flatten()[1]
        for _ in progress_fn(range(n_samples))
    ]


def dataset_to_metric(dataset, model, y_label = 1,
                        zero_ixs = None,
                        progress_fn=tqdm.tqdm,
                        metric_fn = log_loss):
    """
    Calculate metric for dataset
    """
    preds = get_preds(dataset, model,
                    zero_ixs=zero_ixs, progress_fn=progress_fn)
    
    y_labels = [_[1] == y_label for _ in dataset]

    return metric_fn(y_labels, torch.tensor(preds))

def meta_key_to_ttl(meta_key):
    return (
        meta_key.upper() if meta_key.lower() == 'auc' 
        else meta_key.replace('_', ' ').title()
    )