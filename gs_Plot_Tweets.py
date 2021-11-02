# encoding=utf-8
"""
gs_Plot_Tweets creates charts and maps to visualize social media datasets like Tweets.
galindosoft by Brian G. Herbert

"""

import copy
import io
from math import log, fabs, pow
import datetime as dt
from numpy.random import random
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from pandas.tseries.holiday import USFederalHolidayCalendar
from tweet_data_dict import GSC

pio.renderers.default = 'browser'
pio.templates.default = "plotly"
pd.options.plotting.backend = "plotly"
pd.options.display.precision = 3
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('max_columns', 12)
plt_cfg = {"displayModeBar": False, "showTips": False}

start_dt = '2021-04-18'
end_dt = '2021-04-23'

def convert_cloud_to_plotly(mpl_cld):
    """
    converts a matplotlib based word cloud to plotly figure
    :param mpl_cld: the mpl based wordcloud generated in gs_tweet_analysis.py
    :return: plotly figure for wordcloud
    """
    from plotly.tools import mpl_to_plotly
    cloud_fig = mpl_to_plotly(mpl_cld)

    return cloud_fig

def prep_scored_tweets(twdf, opt: str = "median"):
    """
    adjusts tweet score attributes to avoid problems with zero or negative values when
    applying scaling algorithms, then runs both standard and robust scaling on the data.

    :param twdf: pd.DataFrame with quoted/retweeted/reply and favorite counts plus sentiment
    :param opt: str choice of both scalings (default), just "mean" or just "median" scaling
    :return: twdf: PD.DataFrame with new adjusted, scaled columns for counts/sentiment
    """
    twdf['year'] = twdf['date'].apply(lambda x: str(x)[0:4])
    twdf['year'] = pd.to_numeric(twdf['year'], errors='coerce')
    twdf.drop(twdf.loc[twdf['year'] != 2021].index, inplace=True)

    twdf.drop('year', axis=1, inplace=True)

    for column in twdf.columns:
        if column in ["compound", "neg", "pos"]:
            colmin: float = twdf[column].min()
            if colmin < 0.0:
                twdf[column + '_adj'] = twdf[column] + fabs(colmin)

    sc1: int = 0
    if opt in ['both', 'median']:
        robustdf = do_scaling(twdf, "median")
        sc1 = 1
    if opt in ['both', 'mean']:
        scaledf = do_scaling(twdf, "mean")
        sc1 = sc1|2

    if sc1 == 3:
        return robustdf, scaledf
    elif sc1 == 2:
        return scaledf
    elif sc1 == 1:
        return robustdf
    else:
        print("prep_scored_tweets: no valid option! exiting without scaling df")
        return None

def do_scaling(df, typ: str="mean"):
    """
    scale_z creates z_scores, log values, and median-quartile based z_scores for
    Tweet metrics columns (currently the quoted/replied/retweeted count and favorite count)
    This is preparation for more understandable plots and multi-dimensional comparison.
    :param df: pd.DataFrame
    :param typ: string with the requested scaling type- "mean" or "median"
    :return: pd.DataFrame copy with attribute values as z-scores
    """
    df1 = df.copy()
    df1.drop(columns='rt_fave', inplace=True)
    df1['influence'] = df1['qrr'] + df1['fave']
    for column in ['qrr', 'fave', 'influence']:
        if df1[column].min() < 1:
            df1[column + '_log'] = df1[column].apply(lambda x: round(log(x + 1), ndigits=1))
        else:
            df1[column + '_log'] = df1[column].apply(lambda x: round(log(x), ndigits=1))

        stdv = df1[column + '_log'].std()
        cmean = df1[column + '_log'].median()
        df1[column + '_log'] = df1[column + '_log'].apply(lambda x: (cmean + (stdv * 5))
        if x > cmean + (stdv * 5) else x)

        if typ.startswith("mean"):
            df1[column + "_scl"] = round((df1[column] - df1[column].mean()) /
            (df1[column].std()), ndigits=1)
        elif typ.startswith("median"):
            df1[column + "_scl"] = round((df1[column] -
                                          df1[column].median()) / (df1[column].quantile(0.75) -
                                                                   df1[column].quantile(0.25)), ndigits=1)
        else:
            print("do_scaling did not receive either mean or median as scaling type")
            return 1

    return df1

def rscale_col(twdf, colname):
    """
    applies robust scaling to a column of a dataframe,
    currently used for value and volume cols of trading data, as opposed to applying
    scale_z or scale_robust to entire df, as is done with the tweet data.
    :param twdf: pd.DataFrame
    :param colname: name of a column in twdf
    """
    quan75 = twdf[colname].quantile(0.75)
    quan25 = twdf[colname].quantile(0.25)
    cmed = twdf[colname].median()
    twdf[colname + "_adj"] = round(twdf[colname].apply(lambda x:
                                                        (x - cmed) / (quan75 - quan25)), ndigits=2)

    return twdf

def create_layout():
    """
    plotly uses set of dictionaries to define layout for a graph_objects plot.
    this function allows a layout instance to be shared across plots in this app
        once instantiated, object properties can be set directly 'xaxis.title=',
        or creating/modifying objects via plotly Fxs like 'update_layout

    working towards standard typeface, sizes, colors, etc in my apps, such as:
        Helvetica Neue Thin for text, and Copperplate for legends

    :return: plotly layout
    """

    gs_lyt = go.Layout(height=900, width=1500,
        title={'font': {'size': 36, 'family': 'Helvetica Neue UltraLight',
                        'color': GSC['oblk']}},
        paper_bgcolor=GSC['ltgry'],
        font={'size': 18, 'family': 'Helvetica Neue UltraLight'},
        hovermode="closest",
        hoverdistance=10,
        # spikedistance=10,
        showlegend=True,

        legend={'title': {'font': {'size': 20, 'family': 'Copperplate Light'}},
                'font': {'size': 18, 'family': 'Copperplate Light', 'color': GSC["drkryl"]},
                'bgcolor': GSC['beig'], 'bordercolor': GSC['oblk'],
                'borderwidth': 2, 'itemsizing': "trace"
                },
        xaxis={'title': {'font': {'size': 24,'family': 'Helvetica Neue UltraLight'}},
               'linecolor': GSC['oblk'], 'showspikes': True, 'spikethickness': 1,
               },
        yaxis={'title': {'font': {'size': 24, 'family': 'Helvetica Neue UltraLight'}},
               'linecolor': GSC['oblk'], 'showspikes': True, 'spikethickness': 1,
               },
    )
    gs_lyt.template.data.scatter = [
        go.Scatter(marker=dict(symbol="diamond", size=10)),
        go.Scatter(marker=dict(symbol="circle", size=10)),
        go.Scatter(marker=dict(symbol="triangle-up", size=10)),
        go.Scatter(marker=dict(symbol="square", size=10))
    ]
    gs_lyt.coloraxis.colorscale = [
        [0, '#110099'],
        [0.1111111111111111, '#6600aa'],
        [0.2222222222222222, '#7201a8'],
        [0.3333333333333333, '#9c179e'],
        [0.4444444444444444, '#bd3786'],
        [0.5555555555555556, '#d8576b'],
        [0.6666666666666666, '#ed7953'],
        [0.7777777777777778, '#fb9f3a'],
        [0.8888888888888888, '#fdca26'],
        [1, '#f0f921']
    ]

    return gs_lyt

def config_rendering(gofig: go.Figure):
    """
    Some utility pieces to route plots to different types of platforms and displays,
    limit the options allowed when plots are shown,
        some plots should not be panned or manipulated in other ways,
        settings vary based on plot and app- this is a placeholder for plotly settings

        see https://plotly.com/python/renderers/ for more on renderers

    plotly.mimetype in renderers = display plots within the iPython context
    :param gofig: plotly.graph_objects.Figure object
    :return: None
    """
    print(pio.renderers)
    pio.renderers.default = 'chrome+browser+png+svg+json+sphinx_gallery'

    # pio.renderers.keys()
    # if following is set, will show plots on display in python
    # pio.renderers.render_on_display = True

    gs_rend = pio.renderers.default

    cfg: dict = {"displayModeBar": False, "showTips": False}

    gofig.show(renderer=gs_rend, config=cfg)

    return gofig

def prep_pandas(twdf: pd.DataFrame):
    """
    sorting and indexing prior to plotting with pandas
    :param twdf: tweet dataset
    :return: reformatted dataframes for both the above
    """
    dfcp: pd.DataFrame = twdf.copy()
    for colm in dfcp.columns:
        if colm in ['sent', 'datetime', 'dt']:
            dfcp[colm].astype('datetime64', copy=False, errors='ignore')
            dfcp.sort_values(by=[colm])
            dfcp.reset_index(drop=True, inplace=True)
            snt_col = dfcp.columns.get_loc(colm)
            dtstr: str = dt.datetime.strftime(dfcp.iat[0, snt_col], "%Y-%m-%d %H:%M")
            print("first tweet on %s" % dtstr)
            twlen = len(dfcp)
            dtstr = dt.datetime.strftime(dfcp.iat[twlen - 1, snt_col], "%Y-%m-%d %H:%M")
            print("    last tweet on %s" % dtstr)

    return dfcp

def histogram_metrics(df: pd.DataFrame, plyt: go.Layout = None, appd: str = "Superleague"):
    """
    plot either sentiment or influence metrics on bar histogram with overlays
    :param df: tweet dataframe with metrics columns
    :param plyt: go.Layout
    :param appd: name of project or dataset
    :return:
    """
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    lay.title.text = "Histogram Distribution of Sentiment for PRJX".replace("PRJX", appd)
    lay.xaxis.title.text = "Distribution and Frequency"
    lay.yaxis.title.text = "Frequency of Occurrence"

    fig = go.Figure(layout=lay)
    fig.add_trace(go.Histogram(x=df['compound'],
                               name="Compound sentiment",
                               marker_color=GSC["brnz"],
                               opacity=0.6,
                               xbins=dict(start=-1.0, end=1.0, size=0.1),
                               ))
    fig.add_trace(go.Histogram(x=df['neg'],
                               name="Negative sentiment",
                               marker_color=GSC["drkrd"],
                               opacity=0.8,
                               xbins=dict(start=0.0, end=1.0, size=0.1),
                               ))
    fig.add_trace(go.Histogram(x=df['neu'],
                               name="Neutral sentiment",
                               marker_color=GSC["dkblu"],
                               opacity=0.8,
                               xbins=dict(start=0.0, end=1.0, size=0.1),
                               ))
    fig.add_trace(go.Histogram(x=df['pos'],
                               name="Positive sentiment",
                               marker_color=GSC["dkgrn"],
                               opacity=0.8,
                               xbins=dict(start=0.0, end=1.0, size=0.1),
                               ))

    fig.update_layout(bargap=0.1, bargroupgap=0)
    fig.update_yaxes(type="log")

    fig.show(config=plt_cfg)

    return

def box_qrr_fave(twdf: pd.DataFrame, plyt: go.Layout = None, appd: str="Superleague"):
    """
    create box plots for compound, positive, and negative Vader sentiment scores
    show quartile box with std deviations as well as scatter distribution alongside

    :param twdf: pd.Dataframe with all score info
    :param plyt: plotly go.Layout object instance
    :param appd: project or dataset name
    :return: None
    """

    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    lay.title.text = "Q-R-R and Fave Metrics Distribution Plot for PRJX".replace("PRJX", appd)
    lay.xaxis.title.text = "Log Scale Distribution    (Dotted Lines- Mean and Std Dev)"
    lay.yaxis.title.text = "Score Type"
    lay.yaxis.tickfont.size = 18
    lay.xaxis.tickfont.size = 20
    lay.legend.title = "Tweet Influence Measures"
    lay.legend.title.font.size=20
    lay.legend.font.size = 18
    lay.margin.l = 140
    fig = go.Figure(layout=lay)

    fig.add_trace(go.Box(x=twdf['qrr'], quartilemethod="exclusive",
                         name="Q-R-R",
                         marker=dict(color=GSC["brnz"], outliercolor=GSC["mgnta"]),
                         alignmentgroup='tweet',
                         whiskerwidth=1,
                         boxmean='sd'))
    fig.add_trace(go.Box(x=twdf['fave'], quartilemethod="exclusive",
                         name="Likes (Faves)",
                         marker=dict(color=GSC["dkgrn"], outliercolor=GSC["mgnta"]),
                         alignmentgroup='tweet',
                         whiskerwidth=1,
                         boxmean='sd'))

    fig.add_trace(go.Box(x=twdf['rt_qrr'], quartilemethod="exclusive",
                         name="Retweet_Q-R-R",
                         marker=dict(color=GSC["drkrd"], outliercolor=GSC["mgnta"]),
                         whiskerwidth=1,
                         alignmentgroup='retweet',
                         boxmean='sd'))

    fig.add_trace(go.Box(x=twdf['rt_fave'], quartilemethod="exclusive",
                         name="Retweet_Likes",
                         marker=dict(color=GSC["dkryl"], outliercolor=GSC["mgnta"]),
                         whiskerwidth=1,
                         alignmentgroup='retweet',
                         boxmean='sd'))

    fig.update_traces(boxpoints='all', jitter=0.3)
    fig.update_xaxes(type="log")
    fig.show(config=plt_cfg)

    return fig

def box_sentiments(twdf: pd.DataFrame, plyt: go.Layout=None):
    """
    create box plots for compound, positive, and negative Vader sentiment scores
    show quartile box with std deviations as well as scatter distribution alongside

    :param twdf: pd.Dataframe with all score info
    :param plyt: plotly go.Layout object instance
    :return: None
    """
    cfg: dict = {"displayModeBar": False, "showTips": False}
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    lay.title.text = "Vader Sentiment: Compoound, Positive and Negative Distributions"
    lay.xaxis.title.text = "Includes Mean and Standard Deviation"
    lay.yaxis.title.text = "Score Type"
    lay.yaxis.tickfont.size = 20
    lay.xaxis.tickfont.size = 20
    lay.legend.title = "Vader Sentiment for Tweet"
    lay.legend.font.size = 14
    lay.margin.l= 80
    fig = go.Figure(layout=lay)

    fig.add_trace(go.Box(x=twdf['neg'], quartilemethod="linear",
                         name="negative",
                         marker=dict(color=GSC["brnz"], outliercolor=GSC["lgrn"]),
                         alignmentgroup='snt',
                         whiskerwidth=1,
                         boxmean='sd'))
    fig.add_trace(go.Box(x=twdf['pos'], quartilemethod="linear",
                         name="positive", marker_color=GSC["brnorg"],
                         alignmentgroup='snt',
                         boxmean='sd'))
    fig.add_trace(go.Box(x=twdf['compound'], quartilemethod="linear",
                         name="compound", marker=dict(color=GSC["gld"], outliercolor=GSC["lgrn"]),
                         whiskerwidth=1,
                         alignmentgroup='snt',
                         boxmean='sd'))
    fig.update_traces(boxpoints='all', jitter=0.3)
    fig.show(config=cfg)

    return

def bar_tags_mentions(hashes: dict, mentions: dict = None, plyt: go.Layout = None):
    """
    most frequent hashtags and user_mentions plotted with bar chart
    first- use utility in gs_tweet_analysis to sort descending plus filter out stoplists

    :param hashes: dict of hashtags with counts of occurrences
    :param mentions: dict of user mentions with counts of occurrences
    :param plyt: plotly layout instance, creates a copy so not to munge shared elements
    :return:
    """

    HSH_LIM: int = 16
    UMEN_LIM: int = 8
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()
    lay.title = 'Top Hashtags and User Mentions- Superleague Tweets (common STOPs removed)'
    lay.yaxis.title = 'Count: Original Tweets and Quoted Tweet Comments'
    lay.xaxis.title = ""
    lay.title.font.size = 28
    lay.title.font.color = "rgb( 102, 51, 51)"
    lay.xaxis.tickangle = -40
    lay.font.size = 24
    lay.legend.itemsizing = 'trace'
    lay.margin.b = 120
    fig = go.Figure(layout=lay)

    srt: list = sorted(hashes, key=lambda x: hashes[x], reverse=True)
    hashes: dict = {k: hashes[k] for k in srt}
    srt: list = sorted(mentions, key=lambda x: mentions[x], reverse=True)
    mentions: dict = {k: mentions[k] for k in srt}
    if hashes:
        # prep of plot data- first for hashtags then for mentions:
        h_x: list = []
        h_y: list = []
        h_c: list = []
        for h, ct in zip(hashes.items(), range(HSH_LIM)):
            h_x.append(h[0])
            h_y.append(h[1])
            h_c.append(round(random(), ndigits=2))
        if mentions:
            m_x: list = []
            m_y: list = []
            m_c: list = []
            for m, ct in zip(mentions.items(), range(UMEN_LIM)):
                m_x.append(m[0])
                m_y.append(m[1])
                m_c.append(round(random(), ndigits=2))

        fig.add_trace(go.Bar(name="hashtags (no leagues/clubs)", x=h_x, y=h_y, text=h_x,
                             marker=dict(line_width=1, color=h_c),
                             texttemplate="%{x}<br>count: %{y}",
                             textposition="inside",
                             textangle=-90,
                             textfont=dict(size=24)
                             ))

        fig.add_trace(go.Bar(name="Twitter User Mentions", x=m_x, y=m_y, text=m_x,
                             marker=dict(line_width=1, color=m_c),
                             texttemplate="%{x}<br>count: %{y}",
                             textposition="inside",
                             textangle=-90,
                             textfont=dict(size=24)
                             ))
    fig.show(config=plt_cfg)

    return fig

def ohlc_vol_and_sent(mdf: pd.DataFrame, twdf: pd.DataFrame, plyt: go.Layout, appd: str = "Gamestop"):
    """
    shows financial plot with open, high, low and close for stock,
    adds a subplot with trace for trading volume,
    and adds a trace to show tweet volume
    :param mdf: pd.DataFrame of stock trading data
    :param twdf: pd.DataFrame of Tweets
    :param plyt: my custom plotly layout
    :param appd: str with domain or project name
    :return: plotly Figure
    """

    trades: pd.DataFrame = mdf.copy(deep=True)
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    if 'date' in trades.columns:
        strtd = dt.date.strftime(trades['date'].min(), "%b-%d")
        endd = dt.date.strftime(trades['date'].max(), "%b-%d")
    else:
        strd = "Jan-11"
        endd = "Mar-26"

    lay.title.text = "PRJX Stock Price, Volume and Tweet Metadata".replace("PRJX", appd)
    lay.xaxis.tickmode = "linear"
    # lay.xaxis.autorange = False
    lay.xaxis.type = "date"
    lay.xaxis.tickformat = "%b-%d"
    lay.xaxis.tickangle = -75
    lay.xaxis.showticklabels = False
    lay.xaxis.rangeslider.visible = False

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        subplot_titles=('Open-High-Low-Close', 'Volume'),
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
                        row_width=[0.2, 0.7])

    fig.add_trace(go.Candlestick(
        x=trades['date'],
        xperiod=86400000,
        name=appd,
        xperiodalignment="middle",
        line=dict(width=2),
        open=trades['open'],
        high=trades['high'],
        low=trades['low'],
        close=trades['close']
    ), row=1, col=1, secondary_y=False
    )
    fig.update_layout(xaxis_rangeslider_visible=False)

    fig.add_trace(go.Scatter(x=twdf['date'], y=twdf['count'],
                             mode='markers',
                             name="Tweet dataset",
                             xperiod=86400000,
                             xperiodalignment="middle",
                             visible=True,
                             marker=dict(symbol="diamond", size=10,
                                         color=GSC['mgnta'],
                                         opacity=0.8)
                             ), row=1, col=1, secondary_y=True
                  )

    fig.add_trace(go.Bar(x=trades['date'],
                         y=trades['volume'],
                         name="daily trade volume",
                         xperiod=86400000,
                         xperiodalignment="middle",
                         ), row=2, col=1,
                  )

    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=["2021-02-15"])]
    )
    fig.update_layout(legend_title="Stock Market vs<br>Tweet Dataset")
    fig.update_layout(title="Gamestop Stock Performance and Tweets in Dataset")

    fig.show(config=plt_cfg)

    return fig

def plot_multiple(twdf: pd.DataFrame, plyt: go.Layout = None, ccol: str='compound', appd: str=" "):
    """
    uses plotly dual y-axis to plot both market data and tweet data
    :param twdf: pd.DataFrame with twitter data
    :param plyt: plotly go.Layout object instance
    :param ccol: name of column to use for marker color, default is 'compound'
    :param appd: str name of project
    :return: None
    """
    size_v: int = 16

    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    lay.title.text = "Superleague Tweets- Y=Sentiment, Size~=Influence"
    # next prop assignments for date and time format of x-axis, dtick in milliseconds
    lay.xaxis.autorange = False
    lay.xaxis.type = "date"
    lay.xaxis.range = ['2021-04-18 06:00', '2021-04-21 10:00']
    lay.xaxis.dtick = 21600000
    # lay.xaxis.tick0 = pbdf.iat[0, 0].strfrmt("%b-%d %H%M")
    lay.xaxis.tickmode = "linear"  # tickmode=linear turns on tick0 and dtick
    lay.xaxis.tickformat = "%b-%d  %H"
    lay.xaxis.tickangle = -60
    lay.margin.r = 60
    lay.margin.l = 70
    lay.margin.b = 100
    lay.margin.t = 100
    lay.showlegend = False
    lay.yaxis.title.font.size = 28

    siz_ref = round(twdf['influence_log'].max() / size_v, ndigits=1)
    twdf['size_scl'] = twdf['influence_log'].apply(lambda x: round(x / siz_ref, ndigits=1))

    rpt: bool = True
    if ccol in twdf.columns:
        while rpt:
            xmin: float = twdf[ccol].min()
            xmax: float = twdf[ccol].max()
            xmid: float = round(xmax - xmin, ndigits=2)
            if xmid < 1:
                rpt = True
                twdf[ccol] = twdf[ccol].apply(lambda x: pow((x + 0.5), 2.0))
            else:
                rpt = False
    else:
        print(" no column provided for color scaling")
        return 1

    fig = go.Figure(layout=lay)

    fig.add_trace(go.Scattergl(x=twdf['sent'], y=twdf['compound'],
                               mode='markers',
                               name="Influence<br>and Sentiment",
                               hovertemplate='<i><b>Tweet on %{x}</b></i>' +
                                             '<br><b>sentiment</b>: %{y:.2f}' +
                                             "<br><b>social influence</b>: %{meta}",
                             marker=dict(opacity=0.8,
                                         colorscale="thermal",
                                         color=twdf[ccol],
                                         cauto=False,
                                         cmin=xmin,
                                         cmax=xmax,
                                         cmid=xmid,
                                         sizemode='diameter',
                                         sizeref=siz_ref,
                                         sizemin=4,
                                         size=twdf['size_scl'],
                                         symbol="circle"
                                         ),
                             meta=twdf['influence'].apply(lambda x: '{:,}'.format(x)),
                             )
                  )

    fig.update_yaxes(title_text="Sentiment of Tweets, PRJX project".replace("PRJX", appd))
    fig.show(config=plt_cfg)

    return fig

def plot_esl_specific(pbdf: pd.DataFrame, twdf: pd.DataFrame, plyt: go.Layout = None):
    """
    uses plotly dual y-axis to plot both market data and tweet data
    :param pbdf: pd.DataFrame with timeline of public events
    :param twdf: pd.DataFrame with twitter data
    :param plyt: plotly go.Layout object instance
    :return: None
    """
    size_v: int = 24
    cfg: dict = {"displayModeBar": False, "showTips": False}
    sntdct: dict = twdf['compound'].value_counts().to_dict()
    sntdct = {y: sntdct[y] for y in sorted(sntdct, key=lambda x: fabs(x), reverse=True)}
    sntsum = sum(sntdct.values())

    def clr_scaling(scor: float, segments: int=3):
        """
        inner Fx to apply an rgb color to sentiment metrics passed to Fx as a float.
        :param scor: float (expect sentiment score from -1.0 to +1.0)
        :param segments: number of tiers or segments to use to color this attribute
        :return: str with rgb spec- as in 'rgb(102, 102, 102)'
        """
        sntslice: int = round(sntsum / segments, ndigits=0)
        aggr: int = 0
        for k, v in sntdct.items():
            aggr += v
            if fabs(k) < fabs(scor):
                if aggr < sntslice:
                    # return "rgb(204, 51, 51)"
                    return 0.9
                elif (aggr > sntslice) & (aggr < 2 * sntslice):
                    # return "rgb(153, 102, 51)"
                    return 0.5
                elif (aggr > 2 * sntslice) & (aggr < sntsum):
                    # return "rgb(102, 153, 153)"
                    return 0.1
                else:
                    # return "rgb(102, 102, 102)"
                    return 0.0

    yrang = sorted(pbdf['magnitude'].unique().tolist())
    pb_x: list = list(pbdf.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M")))
    pb_y: list = list(pbdf.magnitude.apply(lambda x: int(x)))
    pb_n: list = list(pbdf.description.apply(lambda x:
                                             "<b>" + str(x[:24] + "</b><br>" + x[24:48])))
    scl_adj = twdf['influence_scl'].min()

    twdf['influence_tmp'] = twdf['influence_scl'] + scl_adj + 1
    siz_ref = round(twdf['influence_tmp'].max() / size_v, ndigits=1)
    twdf['size_scl'] = twdf['influence_tmp'].apply(lambda x: round(x / siz_ref, ndigits=1))
    twdf['snt_clr'] = twdf['compound'].apply(lambda x: clr_scaling(x))

    fig = make_subplots(shared_xaxes=True, specs=[[{"secondary_y": True}]])
    fig.update_layout(plyt)
    fig.layout.xaxis.tick0 = twdf['sent'].min()
    fig.layout.xaxis.dtick = 21600000
    fig.layout.xaxis.tickformat = "%Y-%m-%d %H:%M"
    fig.layout.title = "European Superleague Launched...and Flushed -the Three Day Summary"
    fig.layout.legend.itemsizing = "constant"
    annot_rel = {
        'xref': 'paper', 'yref': 'paper',
        'x': 0.8, 'y': 1.0,
        'yanchor': 'top', 'xanchor': 'right',
        'text': "<b>Influence is measure of social impact of Tweets</b><br>" +
                "Marker size scaled to emphasize highest-impact Tweets<br>" +
                "Marker color is by magnitude of absolute value of Vader sentiment<br>" +
                "Tweets shown are top 10% on at least 1 influence measure<br>",
        'font': {'size': 18, 'color': "rgb(25, 25, 25)"}
    }
    fig.update_layout({'annotations': [annot_rel]})

    fig.add_trace(go.Scatter(x=twdf['sent'], y=twdf['compound'],
                             mode='markers', xaxis="x1", yaxis="y1",
                             name="Tweet influence<br>and sentiment",
                             hovertemplate="<i><b>Tweet on %{x}</b></i>" +
                                          "<br>y:<b>comp sentiment</b>: %{y:.2f}" +
                                          "<br><b>social influence</b>: %{meta}",
                             marker=dict(colorscale='Bluered',
                                         color=twdf['snt_clr'],
                                         sizemode='diameter',
                                         sizeref=siz_ref, sizemin=6,
                                         size=twdf['size_scl'], symbol="circle",
                                         ),
                             meta=twdf['influence_scl'],
                             ), secondary_y=False
                  )

    fig.add_trace(go.Scatter(x=pb_x, y=pb_y, mode='markers+text',
                             name="public events", xaxis="x1", yaxis="y2",
                             text=pb_n,
                             textposition="bottom right",
                             hovertemplate='<b>Event: %{x}</b>' +
                                           "<br><b>%{customdata}</b>",
                             marker=dict(size=14, symbol="diamond",
                                         color="rgb(0, 204, 102)",
                                         opacity=0.8,
                                         line=dict(color="rgb(51, 51, 51)",
                                                   width=2)
                                         ),
                             customdata=pb_n
                             ), secondary_y=True
                  )

    fig.update_yaxes(title_text="Vader Compound Sentiment", secondary_y=False)
    fig.update_yaxes(title_text="Superleague Key Events<br>rated by importance",
                     secondary_y=True)

    fig.show(config=cfg)

    return fig

def scatter_with_events(pbdf: pd.DataFrame, twdf: pd.DataFrame, plyt: go.Layout = None, appd: str = " "):
    """
    similar the the 'christmas tree' scatter with red-green
    :param pbdf: pd.DataFrame of public events
    :param twdf: pd.DataFrame of tweets with scaled metrics and sentiment
    :param plyt: plotly go.Layout object instance
    :param appd: str with project name or dataset used
    :return:
    """
    size_ref = 18
    dfcp = twdf.copy()

    def sent_color(varx: float):
        """
        inner fx to calc color based on sentiment stat distribution
        :param varx: field from dataframe
        :return: rgb color
        """
        # sent_stdv = dfcp['compound'].std()
        # sent_mean = dfcp['compound'].mean()
        sent_mean = 0.04
        # pcutoff = sent_mean + sent_stdv
        pcutoff = 0.46
        # ncutoff = sent_mean - sent_stdv
        ncutoff = -0.40

        if varx > pcutoff:
            return "rgb(51, 153, 102)"
            # return 0.9
        elif varx > sent_mean:
            return "rgb(102, 153, 153)"
            # return 0.6
        elif varx > ncutoff:
            return "rgb(153, 102, 102)"
            # return 0.3
        else:
            return "rgb(204, 51, 51)"
            # return 0.0
        return

    dfcp['clr_scl'] = dfcp['compound'].apply(lambda x: sent_color(x))

    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()
    pb_x: list = list(pbdf.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M")))
    pb_y: list = list(pbdf.magnitude.apply(lambda x: str(x)))
    pb_n: list = pbdf.description.to_list()

    dfrmt: str = "%Y-%m-%d %H:%M"
    if 'date' in pbdf.columns:
        end_rng: dt.datetime = pbdf['date'].max()
        end_str = end_rng.strftime(dfrmt)
        strt_rng: dt.datetime = pbdf['date'].min()
        strt_str = strt_rng.strftime(dfrmt)
    else:
        end_str = '2021-04-21 12:00'
        strt_str = '2021-04-18 12:00'

    lay.title.text = "Influential Tweets AND Key Events for PRJX".replace("PRJX", appd)
    lay.xaxis.autorange = False
    lay.xaxis.range = [strt_str, end_str]
    lay.xaxis.type = "date"
    lay.xaxis.dtick = 21600000
    lay.xaxis.tick0 = strt_str
    lay.xaxis.tickmode = "linear"  # tickmode=linear turns on tick0 and dtick
    lay.xaxis.tickformat = dfrmt
    lay.xaxis.tickangle = -60
    lay.legend.title = "Tweet Metric Type"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_yaxes(title_text="Scaled Influence Metric", secondary_y=False)
    fig.update_yaxes(title_text="PRJX Events and most influential Tweets".replace("PRJX", appd), secondary_y=True)
    fig.layout.xaxis.titlefont = dict(size=24, color="rgb(51, 51, 51)"
                                      )
    fig.update_layout(lay)

    fig.add_scatter(x=dfcp['sent'],
                    y=dfcp['qrr_log'], mode="markers",
                    name="Quoted-Retweeted-Replied",
                    hovertemplate='<b>Quoted-Retweeted-Replied</b>' +
                                  "<br>Influence Score: %{y}",
                    marker=dict(symbol="diamond", size=11,
                                color=dfcp['clr_scl'],
                                line=dict(width=1,
                                          color="rgb(51, 51, 51)")
                                ),
                    secondary_y=False)

    fig.add_scatter(x=dfcp['sent'],
                    y=dfcp['fave_log'], mode="markers",
                    name="Favorited",
                    hovertemplate='<b>Likes (Favorited) Score</b>' +
                                  "<br>Scaled Likes: %{y}",
                    marker=dict(symbol="triangle-up", size=11,
                                color=dfcp['clr_scl']),
                    secondary_y=False)

    fig.add_trace(go.Scatter(x=pb_x, y=pb_y, yaxis="y2",
                             mode='markers',
                             name="PRJX Tweets".replace("PRJX", appd),
                             marker=dict(symbol="square", size=14,
                                         color="rgb(204, 204, 102)",
                                         line=dict(color="rgb(51, 51, 51)",
                                                   width=1)
                                         ),
                             text=pb_n,
                             hovertemplate='<b>{name}</b>' +
                                           "<br>%{customdata}",
                             customdata=pb_n
                             ), secondary_y=True
                  )

    fig.show(config=plt_cfg)

    return fig

def scat_sntmnt_y_qrrf_mrkr_evts(pbdf: pd.DataFrame, twdf: pd.DataFrame, plyt: go.Layout = None):
    """
    reworking this plot asof oct 25, need a scatter with sentiment on first y, public events
    on second y, and influence metrics control marker size and color.
    similar to scatter with events but plot of tweets flips Y axis.

    :param pbdf: pd.DataFrame with timeline of public events
    :param twdf: pd.DataFrame with Tweets, influence metrics, and sentiment scores
    :param plyt: a plotly go Layout object
    :return: None
    """
    size_v: int = 20
    siz_ref = round(twdf['influence_log'].max() / size_v, ndigits=1)
    yrang: list = sorted(pbdf['magnitude'].unique().tolist())

    pb_x: list = list(pbdf.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M")))
    pb_y: list = list(pbdf.magnitude.apply(lambda x: str(x)))
    pb_n: list = pbdf.description.to_list()

    twdf['size_scl'] = twdf['influence_log'].apply(lambda x: round(x / siz_ref, ndigits=1))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.update_xaxes(go.layout.XAxis(autorange=False, constrain='domain',
                                     title=dict(text="Superleague Public Events and Social Media, April 2021"),
                                     tick0=twdf.sent.min(),
                                     overlaying="x domain", nticks=40,
                                     range=['Apr-18', 'Apr-21'], rangemode="normal",
                                     showticklabels=True, visible=True,
                                     tickangle=-90, type="date",
                                     tickformat="%b-%d %H", tickmode="linear",
                                     )
                     )
    fig.update_yaxes(go.layout.YAxis(title_text="Sentiment and Influence of ESL tweets",
                                     anchor="y domain",
                                     range=[min(twdf['compound']), max(twdf['compound'])],
                                     autorange=False,
                                     layer="below traces",
                                     rangemode="normal", side="left",
                                     tick0=min(twdf['compound']), dtick=0.1,
                                     type="linear",
                                     ), secondary_y=False
                     )
    fig.update_yaxes(go.layout.YAxis(title_text="Timeline of ESL Events",
                                              range=[min(yrang), max(yrang)],
                                              autorange=False,
                                              layer="below traces", side="right",
                                              rangemode="normal",
                                              tick0=min(yrang), dtick=0.5,
                                              type="linear"
                                     ), secondary_y=True,
                     )

    fig.add_trace(go.Scatter(x=twdf['sent'], y=twdf['compound'],
                             mode='markers', xaxis="x", yaxis="y",
                             name="Tweet influence<br>and sentiment",
                             hovertemplate="<i><b>Tweet on %{x}</b></i>" +
                                           "<br>y:<b>comp sentiment</b>: %{y:.2f}" +
                                           "<br><b>social influence</b>: %{meta}",
                             marker=dict(colorscale='Bluered',
                                         color=twdf['compound'].apply(lambda x: round(x, ndigits=1)),
                                         size=12,
                                         symbol="circle",
                                         ),
                             meta=twdf['influence_log'],
                             ), secondary_y=False
                  )
    fig.add_trace(go.Scatter(x=pb_x, y=pb_y, mode='markers',
                             name="public events", xaxis="x", yaxis="y2",
                             hovertemplate='<b>Superleague milestone: %{x}</b>' +
                                           "<br>%{customdata}",
                             marker=dict(size=14, symbol="diamond", color="rgb(51,51,51)"),
                             customdata=pb_n
                             ), secondary_y=True
                  )

    fig.update_layout()
    fig.show(config=plt_cfg)

    return fig

def scatter_with_template(twdf: pd.DataFrame, plyt: go.Layout=None, appd: str = " "):
    """
    plotly 3d chart with use of custom layout template
    :param twdf: pd.DataFrame
    :param plyt: plotly go.Layout object instance
    :param appd: str with name of domain or project
    :return:
    """
    dfcp = twdf.copy()
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    dtfrmt = "%b-%d"
    if 'sent' in dfcp.columns:
        # make sure first date is current year, then get month-day strings
        if dt.datetime.strftime(dfcp['sent'].min(), "%Y") == "2021":
            strtdt = dt.datetime.strftime(dfcp['sent'].min(), dtfrmt)
            enddt = dt.datetime.strftime(dfcp['sent'].max(), dtfrmt)
    elif 'dt' in dfcp.columns:
        if dt.datetime.strftime(dfcp['dt'].min(), "%Y") == "2021":
            strtdt = dt.datetime.strftime(dfcp['dt'].min(), dtfrmt)
            enddt = dt.datetime.strftime(dfcp['dt'].max(), dtfrmt)
    else:
        strtdt = "Apr-18"
        enddt = "Apr-21"

    lay.title.text = "PRJX Tweets- core Measures of Influence<br>Q-R-R and Faves".replace("PRJX", appd)
    lay.xaxis.autorange=False
    lay.xaxis.range = [strtdt, enddt]

    # date-time blocks in msec: 6-hour periods is  dtick= 21,600,000  604,800,000
    lay.xaxis.type = "date"
    lay.xaxis.tickmode = "linear"  # tickmode=linear turns on tick0 and dtick
    lay.xaxis.tick0 = strtdt
    lay.xaxis.dtick = 21600000
    lay.xaxis.tickformat = dtfrmt
    lay.xaxis.tickangle = -70
    lay.xaxis.ticks = "inside"
    lay.legend.title = "Tweet Influence Measure"
    lay.xaxis.title = "Log Scale adjusted Metric"
    fig = go.Figure(layout=lay)

    pd.options.display.float_format = "{:,.2f}".format
    fig.add_scatter(x=dfcp['sent'],
                    y=dfcp['qrr_log'],
                    mode="markers",
                    hovertemplate="<i><b>Tweet on %{x}</b></i>" +
                                  "<br>qrr total: %{meta:.2f}",
                    name="Quoted-Retweeted-Replied",
                    meta=dfcp['qrr'],
                    )

    fig.add_scatter(x=dfcp['sent'],
                    y=dfcp['fave_log'],
                    mode="markers",
                    hovertemplate="<i><b>Tweet on %{x}</b></i>" +
                                  "<br>qrr total: %{meta:.2f}",
                    name="Liked",
                    meta=dfcp['fave'])

    fig.add_annotation(text="One trace for each major influence metric:  Q-R-R, and Faves<br>" +
                            "ReTweets+QuotedTweets+Replies for originating Tweet<br>" +
                            "Favorite or Liked Count for originating Tweet<br>" +
                            "Top 10% on Metric Selected for Tweets on Topic in Dates",
                       xref="paper", yref="paper",
                       yanchor="top", xanchor="right",
                       x=1.0, y=1.0, showarrow=False)
    # following show parms remove floating menu and unnecessary dialog box
    fig.show(config=plt_cfg)

    return

def scatter3d_bydate(vlst, plyt: go.Layout=None, appd: str="", stype: str="compound"):
    """
    plot_3d_scatter uses 3 dimensions plus size and color to graphically represent tweet
    attributes such as date and time, sentiment, quote/retweet/reply count, favorite count,
    and counts for original tweet if item is a retweet.

    :param vlst: pd.DataFrame with normalized (scaled) features
    :param plyt: plotly go.Layout object instance
    :param appd: str to pass in the domain or topic of project
    :param stype: sentiment field to use for color coding, typicall 'compound' or 'neg'
    :return: None
    """
    cfg: dict = {"displayModeBar": False, "showTips": False}
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    def bin_color(cvar: float):
        """
        inner function to return an rgb color based on fractional value (i.e. sentiment)
        :param cvar: column value as float
        :return:
        """
        if fabs(cvar) > 0.6:
            return "rgb(204, 51, 102)"
        elif fabs(cvar) > 0.3:
            return "rgb(153, 52, 153)"
        else:
            return "rgb(51, 102, 102)"

    def sent_color(var: float):
        """
        inner fx to calc color based on type
        :param var: type field from df
        :return: rgb color
        """
        if var in ['qfs', 'fs', 'qs']:
            return "rgb(204, 51, 102)"
            # return 0.9
        elif var in ['s']:
            return "rgb(204, 52, 204)"
            # return 0.6
        elif var in ['q', 'f', 'qf']:
            return "rgb(153, 102, 102)"
            # return 0.3
        else:
            return "rgb(51, 51, 51)"
            # return 0.0
        return

    if 'type' in vlst.columns:
        vlst['s_color'] = vlst['type'].apply(lambda x: sent_color(x))
    else:
        vlst['s_color'] = vlst[stype].apply(lambda x: bin_color(x))

    lay.title.text = "Scatter3d by Date: Tweet influence and sentiment for PRJX".replace("PRJX", appd)
    lay.scene = {'xaxis': {'title': 'Project Date Range', 'spikethickness': 1},
                   'yaxis': {'title': 'scaled quote-retweet-reply', 'spikethickness': 1},
                   'zaxis': {'title': 'scaled favorites', 'spikethickness': 1},
                   'aspectmode': 'manual',
                   'aspectratio': {'x': 2, 'y': 1, 'z': 1}
                   }
    # aspectmode options are cube, manual, data, auto
    lay.xaxis.tickmode = "linear"
    lay.xaxis.dtick = 21600000
    lay.xaxis.tick0 = "2021-04-18 12:00"
    lay.margin.l = 100

    fig = go.Figure(data=go.Scatter3d(
        x=vlst['sent'],
        y=vlst['qrr_log'],
        z=vlst['fave_log'],
        hovertemplate='<i><b>Tweet on</b></i> %{meta}' +
                      '<br>y:<b>QRR Scale</b>: %{y:.2f}' +
                      '<br>z:<b>Favorite Scale</b>: %{z:.2f}' +
                      "<br><b>Compound Sentiment</b>: %{customdata:.2f}" +
                      "<br>%{text}",
        text=vlst['text'],
        name="Top 10% Tweets<br>Q-R-R or Fave count, " +
             "<br>High Sentiment in Magenta",
        mode='markers',
        showlegend=False,
        marker=dict(
            sizemode='diameter',
            sizeref=round(fabs(vlst[stype].max()) / 28, ndigits=1),
            sizemin=4,
            size=vlst[stype].apply(lambda x: round(fabs(x) * 20, ndigits=1)),
            opacity=1,
            color=vlst['s_color'],
        ),
        customdata=vlst[stype],
        meta=vlst['sent'].apply(lambda x: dt.datetime.strftime(x, "%B %d, %Y"))
    ), layout=lay
    )
    fig.update_layout(lay)

    fig.show(config=cfg)

    return fig

def plot_3d(rdf: pd.DataFrame, plyt: go.Layout=None, appd: str="", styp : str="compound"):
    """
    rewrite of scatter 3d to plot 6-hour blocks and scale marker size by influence
    :param rdf: pd.DataFrame with normalized features (use do_scaling function)
    :param plyt: plotly go.Layout object instance
    :return: None
    """
    twt: pd.DataFrame = rdf.copy(deep=True)
    siz_const: int = 15    # marker size: divide max value by this for sizeref
    sntdct: dict = twt[styp].value_counts().to_dict()
    sntdct = {y: sntdct[y] for y in sorted(sntdct, key=lambda x: fabs(x), reverse=True)}
    sntsum = sum(sntdct.values())

    def set_color(metric: float, segs: int=4, debug: bool=False):
        """
        passed a float field from a dataframe, returns an rgb color to use for markee
        :param metric: float
        :return: str in form "rbg(0, 0, 0)"
        """
        sntslice: int = round(sntsum / segs, ndigits=0)
        aggr: int = 0
        for k, v in sntdct.items():
            aggr += v
            if fabs(k) < fabs(metric):
                if aggr < sntslice:
                    # return "rgb(204, 51, 51)"
                    return 1.0
                elif (aggr > sntslice):
                    # return "rgb(153, 102, 51)"
                    return 0.7
                elif (aggr > 2 * sntslice):
                    # return "rgb(102, 153, 153)"
                    return 0.3
                else:

                    # return "rgb(102, 102, 102)"
                    return 0.0
        return

    def parse_lst(coly: list):
        """
        inner fx to parse a list in a column in a dataframe
        :param coly: a list of str
        return:
        """
        if isinstance(coly, list):
            if len(coly) > 0:
                # print(" %s is list with length > 0" % coly)
                lst_str: str = ""
                for hsh in coly:
                    hsh = str(hsh).lower()
                    if hsh.startswith("superleague"):
                        continue
                    tmp: str = " " + hsh
                    # print(" iter item is %s" % hsh)
                    lst_str: str = lst_str + tmp
                    lst_str = lst_str.strip()
                    # print("joined creation is %s" % lst_str)
                return lst_str
            else:
                # print(" length of hash list is 0")
                return None
        else:
            # print(" did not get list from hashtag field")
            return None

    def chk_typ(coly):
        """
        if this Tweet was on all three top lists (Q-R-R, Faves, and Sentiment) show
        it differently
        :param coly:
        :return:
        """
        if coly in ['qfs', 'qs', 'fs']:
            return 3
        elif coly in ['qf']:
            return 2
        elif coly in ['q', 'f']:
            return 1
        else:
            return 0

    cfg: dict = {"displayModeBar": False, "showTips": False}
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    twt['s_color'] = twt[styp].apply(lambda x: set_color(x))
    clrlst: list = twt['s_color'].to_list()

    # aspectmode options are cube, manual, data, auto
    lay.xaxis.tickmode = "linear"
    lay.xaxis.dtick = 43200000
    lay.xaxis.tick0 = "Apr-18 12p"
    lay.xaxis.tickformat = "%b-%d %H"
    lay.xaxis.tickfont = dict(size=12)
    lay.title.text = "PRJX Topic-dataset Twitter Project<br>QT, RT, Reply and Like metrics" \
                     "<br>Color-coded Sentiment".replace("PRJX", appd)
    lay.legend.title = "PRJX influential Tweets".replace("PRJX", appd)
    lay.scene = {'xaxis': {'spikethickness': 1, 'dtick': 21600000,
                           'showtickprefix': None, 'tickformat': "%b-%d %H:%M",
                           'type': 'date', 'tickfont': {'color': "rgb(51, 102, 153)",
                                                        'family': 'Helvetica Neue UltraLight',
                                                        'size': 14}
                           },
                 'yaxis': {'title': 'scaled Quote-ReTweet-Reply', 'spikethickness': 1,
                           'showtickprefix': None,
                           'tickfont': {'color': "rgb(51, 102, 153)",
                                        'family': 'Helvetica Neue UltraLight', 'size': 14}
                           },
                 'zaxis': {'title': 'scaled Liked count', 'spikethickness': 1,
                           'tickfont': {'color': "rgb(51, 102, 153)",
                                        'family': 'Helvetica Neue UltraLight', 'size': 14}
                           },
                 'aspectmode': 'manual', 'aspectratio': {'x': 2, 'y': 1, 'z': 1},
                 }
    lay.margin.l = 100

    annot_rel = {
        'xref': 'paper', 'yref': 'paper',
        'x': 0.1, 'y': 0.0, 'xanchor': 'left', 'yanchor':'bottom',
        'text': 'ESL is announced at about 12:30 BST Sunday 18th<br>' +
                'Online storm starts immediately<br>' +
                'Monday brings public fan protests',
        'bgcolor': "rgb(153, 153, 153)",
        'showarrow': False,
        'font': {'size': 14, 'color': "rgb(25, 25, 25)"}
    }
    lay.annotations = [annot_rel]

    xlst: list = twt.sent.to_list()
    qrlst: list = list(twt['qrr_log'].apply(lambda x: "{:.2f}".format(x)))
    fvlst: list = list(twt['fave_log'].apply(lambda x: "{:.2f}".format(x)))
    sntlst: list = list(twt[styp].apply(lambda x: "{: .2f}".format(x)))
    txtlst: list = list(twt['hashes'].apply(lambda x: parse_lst(x)))
    typlst: list = list(twt['type'].apply(lambda x: chk_typ(x)))

    mrglst: list = []
    for qm, fm, sm in zip(qrlst, fvlst, sntlst):
        mrglst.append((float(qm) + float(fm)) * pow(float(sm), 2))
    mrg_mean = sum([float(x) for x in mrglst]) / len(mrglst)
    mrg_max = max(mrglst)
    print("merged metric factor has mean of %.2f and max of %.2f " % (mrg_mean, mrg_max))
    sizlst: list = [round(pow((float(sntx) + 1.0), 2.0), ndigits=2) for sntx in sntlst]
    sizclc: float = float(max(sizlst)) / siz_const
    sizclc = round(sizclc, ndigits=2)
    siz2lst: list = []
    for siz, typ in zip(sizlst, typlst):
        siz2lst.append(siz + typ)

    fig = go.Figure(data=go.Scatter3d(x=xlst, y=qrlst, z=fvlst,
                                      hovertemplate='<b>Tweet on</b> %{x}' +
                                                    '<br>y: Scaled QRR-F: %{y:.2f}' +
                                                    "<br>Sentiment: %{customdata:.2f}" +
                                                    "<br>hashtags: %{text}",
                                      text=txtlst,
                                      name="QT-RT-Reply-Like, w/Sentiment",
                                      mode='markers', showlegend=True,
                                      textsrc="%{customdata:.1f}",
                                      marker=dict(sizemode='diameter',
                                                  sizeref=sizclc,
                                                  sizemin=4,
                                                  size=siz2lst,
                                                  opacity=0.8,
                                                  color=clrlst, colorscale='viridis'
                                                  ),
                                      customdata=sntlst
                                      ), layout=lay
                    )

    fig.update_layout(lay, margin=dict(l=30, r=50, b=30, t=60))
    fig.show(config=cfg)

    return fig

def p3d_new(rdf: pd.DataFrame, plyt: go.Layout = None, appd: str = "", styp: str = "compound"):
    """
    rewrite of scatter 3d to plot 6-hour blocks and scale marker size by influence
    :param rdf: pd.DataFrame with normalized features (use do_scaling function)
    :param plyt: plotly go.Layout object instance
    :return: None
    """
    twt: pd.DataFrame = rdf.copy(deep=True)
    siz_const: int = 22
    sntmean: float = twt[styp].mean()
    sntstdv: float = twt[styp].std()

    def set_color(metric: float):
        """
        passed a float field from a dataframe, returns an rgb color to use for markee
        :param metric: float
        :return: str in form "rbg(0, 0, 0)"
        """
        if metric > (sntmean + sntstdv):
            return GSC["dkgrn"]
            # return 1.0
        elif metric > sntmean:
            return GSC["gray"]
            # return 0.7
        elif metric > (sntmean - sntstdv):
            return GSC["drkrd"]
            # return 0.3
        else:
            return GSC["mgnta"]
            # return 0.0
        return

    def parse_lst(coly: list):
        """
        inner fx to parse a list in a column in a dataframe
        :param coly: a list of str
        return:
        """
        if isinstance(coly, list):
            if len(coly) > 0:
                # print(" %s is list with length > 0" % coly)
                lst_str: str = ""
                for hsh in coly:
                    hsh = str(hsh).lower()
                    if hsh.startswith("superleague"):
                        continue
                    tmp: str = " " + hsh
                    # print(" iter item is %s" % hsh)
                    lst_str: str = lst_str + tmp
                    lst_str = lst_str.strip()
                    # print("joined creation is %s" % lst_str)
                return lst_str
            else:
                # print(" length of hash list is 0")
                return None
        else:
            # print(" did not get list from hashtag field")
            return None

    def chk_typ(coly):
        """
        if this Tweet was on all three top lists (Q-R-R, Faves, and Sentiment) show
        it differently
        :param coly:
        :return:
        """
        if coly in ['qfs', 'qs', 'fs']:
            return 3
        elif coly in ['qf']:
            return 2
        elif coly in ['q', 'f']:
            return 1
        else:
            return 0

    cfg: dict = {"displayModeBar": False, "showTips": False}
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    twt['s_color'] = twt[styp].apply(lambda x: set_color(x))
    clrlst: list = twt['s_color'].to_list()

    # aspectmode options are cube, manual, data, auto
    lay.xaxis.tickmode = "linear"
    lay.xaxis.dtick = 43200000
    lay.xaxis.tick0 = "Apr-18"
    lay.xaxis.tickformat = "%b-%d"
    lay.title.text = "PRJX Topic-dataset Twitter Project<br>QT, RT, Reply and Like metrics"\
                     "<br>Color-coded Sentiment".replace("PRJX", appd)
    lay.legend.title = "PRJX influential Tweets".replace("PRJX", appd)
    lay.scene = {'xaxis': {'spikethickness': 1, 'dtick': 21600000,
                           'showtickprefix': None, 'tickformat': "%b-%d",
                           'type': 'date', 'tickfont': {'color': "rgb(51, 102, 153)",
                                                        'family': 'Helvetica Neue UltraLight',
                                                        'size': 14
                                                        }
                           },
                 'yaxis': {'title': 'Quote-ReTweet-Reply', 'spikethickness': 1,
                           'showtickprefix': None,
                           'tickfont': {'color': "rgb(51, 102, 153)",
                                        'family': 'Helvetica Neue UltraLight', 'size': 14
                                        }
                           },
                 'zaxis': {'title': 'Likes (scaled)', 'spikethickness': 1,
                           'tickfont': {'color': "rgb(51, 102, 153)",
                                        'family': 'Helvetica Neue UltraLight', 'size': 14
                                        }
                           },
                 'aspectmode': 'manual', 'aspectratio': {'x': 2, 'y': 1, 'z': 1},
                 }
    lay.margin.l = 120

    annot_rel = {
        'xref': 'paper', 'yref': 'paper',
        'x': 0.9, 'y': 0.0, 'xanchor': 'right', 'yanchor': 'bottom',
        'text': "ESL is announced 12:30 BST Sunday 18th<br>" +
                "Plot of only Tweets in 90+ percentile influence<br>" +
                "Color-banded for Sentiment, sized for Q-R-R-Fave<br>",
        'bgcolor': GSC["beig"],
        'showarrow': False,
        'font': {'size': 18, 'color': "rgb(25, 25, 25)"}
    }
    lay.annotations = [annot_rel]

    xlst: list = twt.sent.to_list()
    qrlst: list = list(twt['qrr'].apply(lambda x: round(log(x), ndigits=2)))
    fvlst: list = list(twt['fave'].apply(lambda x: round(log(x), ndigits=2)))
    sntlst: list = list(twt[styp].apply(lambda x: round((x), ndigits=2)))
    sntclr: list = list(twt[styp].apply(lambda x: set_color(x)))
    txtlst: list = list(twt['hashes'].apply(lambda x: parse_lst(x)))
    typlst: list = list(twt['type'].apply(lambda x: chk_typ(x)))

    sizlst: list = [x + y for x, y in zip(qrlst, fvlst)]
    sizclc: float = float(max(sizlst)) / siz_const
    sizclc = round(sizclc, ndigits=2)
    siz2lst: list = []
    for siz, typ in zip(sizlst, typlst):
        siz2lst.append(siz + typ)

    fig = go.Figure(data=go.Scatter3d(x=xlst, y=qrlst, z=fvlst,
                                      hovertemplate='<b>Tweet on</b> %{x}' +
                                                    '<br>y: Scaled QRR-F: %{y:.2f}' +
                                                    "<br>Sentiment: %{customdata:.2f}" +
                                                    "<br>hashtags: %{text}",
                                      text=txtlst,
                                      name="Tweet Influence and Sentiment",
                                      mode='markers', showlegend=False,
                                      textsrc="%{customdata:.1f}",
                                      marker=dict(sizemode='diameter',
                                                  sizeref=sizclc,
                                                  sizemin=6,
                                                  size=sizlst,
                                                  opacity=0.8,
                                                  color=sntclr,
                                                  ),
                                      customdata=sntlst
                                      ), layout=lay
                    )

    fig.update_layout(lay, margin=dict(l=30, r=50, b=30, t=60))
    fig.show(config=cfg)

    return fig

def do_cloud(batch_tw_wrds, opt_stops: str = None, maxwrd: int = 80):
    """
    wordcloud package options can be explored via '?wordcloud' (python- show docstring)
    background_color="white" - lighter background makes smaller words more legible,
    max_words= this can prevent over clutter, mask=shape the cloud to an image,
    stopwords=ad-hoc removal of unwanted words, contour_width=3,
    :param batch_tw_wrds: list of list of word tokens for tweets
    :param opt_stops: str var name for optional stop list
    :param maxwrd: int typically from 80 to 120 for total words to appear in cloud
    :return:
    """
    from wordcloud import WordCloud
    import io
    import matplotlib.pyplot as plt

    cloud_text = io.StringIO(newline="")
    for tok in batch_tw_wrds:
        if isinstance(tok, str):
            cloud_text.write(tok + " ")
        else:
            for a_tw in tok:
                if isinstance(a_tw, list):
                    cloud_text.write(" ".join([str(x) for x in a_tw]) + " ")
                if isinstance(a_tw, str):
                    # if simple list of text for each tweet:
                    cloud_text.write(a_tw + " ")

    wordcld = WordCloud(width=800, height=800, max_words=maxwrd,
                        background_color='white',
                        stopwords=opt_stops, min_word_length=4,
                        min_font_size=10).generate(cloud_text.getvalue())

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcld)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    return

def wordcloud_plotly(wlst: dict, lim: int = 50):
    """
    an implementation of word cloud in plotly, send a dict of word frequencies
    plus the max number of words to plot in the cloud
    For words, you would have your bag of words. colors and weights are random numbers here, but you can get them
    from the analysis you are doing.

    :param wlst: dict of words and frequency
    :return:
    """
    import plotly
    from plotly.offline import plot
    import random

    wlst = {x: wlst[x] for x, i in
            zip(sorted(wlst, key=lambda x: wlst.get(x), reverse=True), range(lim))}

    words: list = []
    weights: list = []
    colors: list = []
    for k, v in wlst.items():
        if v > 2:
            words.append(k)
            weights.append(v)
            colors.append(random.randrange(1, 10))

    data = go.Scatter(x=[random.random() for i in range(lim)],
                      y=[random.random() for i in range(lim)],
                      mode='text',
                      text=words,
                      marker={'opacity': 0.3},
                      textfont={'size': weights,
                                'color': colors
                                })
    layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}
                        })
    fig = go.Figure(data=[data], layout=layout)

    fig.show()

    return

def do_cloud(batch_tw_wrds, opt_stops: str = None, maxwrd: int = 80):
    """
    wordcloud package options can be explored via '?wordcloud' (python- show docstring)
    background_color="white" - lighter background makes smaller words more legible,
    max_words= this can prevent over clutter, mask=shape the cloud to an image,
    stopwords=ad-hoc removal of unwanted words, contour_width=3,
    :param batch_tw_wrds: list of list of word tokens for tweets
    :param opt_stops: str var name for optional stop list
    :return:
    """
    import matplotlib.pyplot as plt

    cloud_text = io.StringIO(newline="")
    for tok in batch_tw_wrds:
        if isinstance(tok, str):
            cloud_text.write(tok + " ")
        else:
            for a_tw in tok:
                if isinstance(a_tw, list):
                    cloud_text.write(" ".join([str(x) for x in a_tw]) + " ")
                if isinstance(a_tw, str):
                    # if simple list of text for each tweet:
                    cloud_text.write(a_tw + " ")

    wordcld = WordCloud(width=800, height=800, max_words=maxwrd,
                        background_color='white',
                        stopwords=opt_stops,
                        min_font_size=10).generate(cloud_text.getvalue())

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcld)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    return

def do_sent_classify(df: pd.DataFrame, clrcol: str = "compound"):
    """
    create classifications based on a sentiment score type
    :param df:
    :return: pd.DataFrame
    """
    if clrcol in df.columns:
        s_dev = round(float(df[clrcol].std()), ndigits=1)
        s_med = round(float(df[clrcol].median()), ndigits=1)
        upper = s_med + s_dev
        lower = s_med - s_dev

        def do_class(sntx: float):
            if sntx > upper:
                return "rgb(0, 102, 0)"
            elif sntx > lower:
                return "rgb(204, 204, 153)"
            else:
                return "rgb(255, 51, 153)"

        df['snt_clr'] = df[clrcol].apply(lambda x: do_class(x))

        return df
    else:
        print("Error applying sentiment classification")
        return None