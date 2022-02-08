#!python

# builtin
from typing import Union
import re

# external
import plotly.express as px
import pandas as pd
import numpy as np


def scale_pvals(
    x: Union[list, np.array],
) -> list:
    """
    Function to scale p-values that are already negative log10 transformed.

    Parameters
    ----------
    x : list or np.array of integers
        List (or any other iterable) of p-values that are already
        negative log10 transformed.

    Returns
    -------
    : list
        The lists of negative log10 transformed p-value bins.
    """
    steps = [1000, 100, 50, 10, 5, 2]
    r = []
    for xi in x:
        s_max = 0
        for s in steps:
            if xi >= s:
                if s > s_max:
                    s_max = s
        r.append('> '+str(s_max))
    return(r)


def plot_enrichment(
    data: pd.DataFrame,
    ptm_select: list = None,
    roi_select: list = None,
):
    """
    Plot the enrichment of PTMs in different protein regions.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with enrichment results from perform_enrichment_analysis.
    ptm_select: list
        List of PTMs to show.
        Default is None, which shows all PTMs in data.
    roi_select: list
        List of regions of interest (ROIs) to show.
        Default is None, which shows all ROIs in data.

    Returns
    -------
    : plot
        Figure showing enrichment of PTMs in different protein regions.
    """
    df = data.copy(deep=True)
    df['ptm'] = [re.sub('_', ' ', p) for p in df['ptm']]
    category_dict = {}
    if ptm_select:
        ptm_select = [re.sub('_', ' ', p) for p in ptm_select]
        df = df[df.ptm.isin(ptm_select)]
        category_dict['ptm'] = ptm_select
    if roi_select:
        df = df[df.roi.isin(roi_select)]
        category_dict['roi'] = roi_select
    df['log_odds_ratio'] = np.log(df['oddsr'])
    df['neg_log_adj_p'] = -np.log10(df.p_adj_bh)
    df['neg_log_adj_p_round'] = scale_pvals(df.neg_log_adj_p)
    category_dict['neg_log_adj_p_round'] = list(reversed([
        '> 1000', '> 100', '> 50', '> 10', '> 5', '> 2', '> 0']))
    color_dict = {'> 1000': 'rgb(120,0,0)',
                  '> 100': 'rgb(177, 63, 100)',
                  '> 50': 'rgb(221, 104, 108)',
                  '> 10': 'rgb(241, 156, 124)',
                  '> 5': 'rgb(245, 183, 142)',
                  '> 2': 'rgb(246, 210, 169)',
                  '> 0': 'grey'}
    fig = px.bar(df,
                 x='ptm',
                 y='log_odds_ratio',
                 labels=dict({'ptm': 'PTM',
                              'log_odds_ratio': 'log odds ratio',
                              'neg_log_adj_p_round': '-log10 (adj. p-value)'}),
                 color='neg_log_adj_p_round',
                 facet_col='roi',
                 hover_data=['oddsr', 'p_adj_bh'],
                 category_orders=category_dict,
                 color_discrete_map=color_dict,
                 template="simple_white",
                 )
    fig.update_layout(
        autosize=False,
        width=400+(len(df.ptm.unique())*20),
        height=500,
        margin=dict(
            autoexpand=False,
            l=100,
            r=150,
            b=150,
            t=50,
        ),
    )
    config = {'toImageButtonOptions': {
        'format': 'svg', 'filename': 'structure ptm enrichment'}}
    return(fig.show(config=config))


def plot_ptm_colocalization(
    df,
    name='Fraction of modified acceptor residues',
    context=None
):
    """
    Plot PTMs co-localization.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with results from evaluate_ptm_colocalization(.
    name: str
        Name of the resulting plot.
        Default is 'Fraction of modified acceptor residues'.
    context: str
        Either '3D', '1D' or None.
        Default is None, which shows both 1D and 3D results.

    Returns
    -------
    : plot
        Figure showing PTMs co-localization across distance bins.
    """
    if context in ['1D', '3D']:
        df = df[df.context == context]
        fig = px.scatter(
            df,
            x="cutoff",
            y="value",
            error_y="std_random_fraction",
            color="variable",
            facet_col="ptm_types",
            facet_col_spacing=0.05,
            labels={"value": "Fraction of modified acceptors",
                    "ptm_types": "",
                    "variable": ""},
            color_discrete_sequence=['rgb(177, 63, 100)', 'grey'])
        fig = fig.update_layout(width=1100, height=350)
        fig = fig.update_yaxes(matches=None, showticklabels=True, col=1)
        fig = fig.update_yaxes(matches=None, showticklabels=True, col=2)
        fig = fig.update_yaxes(matches=None, showticklabels=True, col=3)
        fig = fig.update_yaxes(matches=None, showticklabels=True, col=4)
        fig = fig.update_yaxes(matches=None, showticklabels=True, col=5)
    else:
        fig = px.scatter(
            df,
            x="cutoff",
            y="value",
            error_y="std_random_fraction",
            color="variable",
            facet_row="ptm_types",
            facet_col="context",
            labels={"value": "Fraction of modified acceptors",
                    "ptm_types": "",
                    "variable": ""},
            color_discrete_sequence=['rgb(177, 63, 100)', 'grey'])
        fig = fig.update_layout(width=1000, height=1800)
        fig = fig.update_yaxes(matches=None)
    fig = fig.update_layout(title=name,
                            template="simple_white")
    config = {'toImageButtonOptions': {'format': 'svg', 'filename': name}}
    return fig.show(config=config)