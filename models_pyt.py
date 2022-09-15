from toolz import pipe as p

import torch
from torch import nn

import dplyr_lite as d
import hyperparam_config as config


class ScanCnn(nn.Module):
    def __init__(self, sequence_length, n_terms:int,
                model_params: config.CnnModelParams = config.CnnModelParams(),
        ):
        super(ScanCnn, self).__init__()

        mp = model_params
        self.do_bias = mp.bias
        self.do_batch_norm = mp.do_batch_norm
        self.embed = nn.Embedding(n_terms, mp.embed_size)
        self.embed_size = mp.embed_size
        self.kernel_sizes = mp.kernel_sizes()
        self.n_filters = mp.n_filters
        self.n_kernels = len(self.kernel_sizes)

        def conv1d(ks):
            return nn.Conv1d(in_channels=self.embed_size,
                             out_channels=self.n_filters,
                             kernel_size=ks,
                             bias=self.do_bias)

        def max_pool1d(ks): return nn.MaxPool1d(
            sequence_length - ks + 1,
        )

        map_kernels = lambda fn, agg = nn.ModuleList: agg(
            [fn(ks) for ks in self.kernel_sizes]
        )

        self.convolutions = map_kernels(conv1d)
        self.pools = map_kernels(max_pool1d)
        
        if self.do_batch_norm:
            self.norms = map_kernels(
                lambda _: nn.BatchNorm1d(self.n_filters))

        self.linear = nn.Linear(self.n_filters*self.n_kernels, 2)

        self.dropout = nn.Dropout(p=mp.dropout)


    def forward(self, x, **_):
        return d.p(
            x,
            self.calc_pre_conv,
            self.calc_conv,
            self.calc_logits
        )


    def calc_pre_conv(self, x, **_):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        return p(
            x,
            self.embed,
            lambda _: _.swapaxes(2, 1),
        )
    

    def calc_conv(self, pre_conv, pool_lo = None):
        if pool_lo is None:
            pool_lo = self.pools

        conv_outputs = torch.cat(
            [
                self.apply_conv_max(pre_conv, i, 
                        pool = pool_lo[i]
                ) 
                for i in range(self.n_kernels)
            ],
            dim=1)

        return conv_outputs
    

    def apply_conv_max(self, pre_conv, i, pool):
        return p(
            pre_conv,
            self.convolutions[i],
            pool,
            self.norms[i] if self.do_batch_norm else d.identity()
        )


    def calc_logits(self, conv_outputs):
        return p(
            conv_outputs,
            lambda _: _.squeeze(),
            self.dropout,
            self.linear
        )


class ScanCnn1Logit(ScanCnn):
    def __init__(self, sequence_length, n_terms:int,
                model_params: config.CnnModelParams = config.CnnModelParams(),
        ):
        super(ScanCnn1Logit, self).__init__(
            sequence_length, model_params=model_params,
            n_terms=n_terms)

        self.linear = nn.Linear(self.n_filters*self.n_kernels, 1)
    
    def forward(self, x, **_):
        return self.mk_logit_pair(super(ScanCnn1Logit, self).forward(x))
    
    def mk_logit_pair(self, base_out):
        if base_out.dim() == 1:
            base_out = base_out.unsqueeze(0)
        return torch.cat([-base_out, base_out], dim = 1)


class CnnLogitExtracts(nn.Module):
    def __init__(self, cnn_model:ScanCnn,
                seq_length = 3000):
        super(CnnLogitExtracts, self).__init__()
        self.cnn_model = cnn_model
        self.cnn_model.train(mode = False)
        self.pools = [
            nn.MaxPool1d(kernel_size = seq_length - ks + 1,
                return_indices=True) 
            for ks in self.cnn_model.kernel_sizes
        ]
        self.train(mode = False)
    
    def forward(self, x, **_):
        conv_in = self.cnn_model.calc_pre_conv(x)
        loc_activs = [
            self.cnn_model.apply_conv_max(conv_in, i, self.pools[i])
            for i in range(self.cnn_model.n_kernels)
        ]
        def cat(ix):
            return torch.cat([
                _[ix].squeeze().unsqueeze(dim = 1)
                for _ in loc_activs
            ], dim = 1)
        locs = cat(1)
        activs = cat(0)

        return locs, activs

