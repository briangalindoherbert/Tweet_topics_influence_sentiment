# encoding=utf-8
"""
gs_Plot_Tweets creates charts and maps to visualize social media datasets like Tweets.
galindosoft by Brian G. Herbert

"""

from numpy.random import random
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas.tseries.holiday import USFederalHolidayCalendar
from math import log, fabs
import datetime as dt
import copy
from typing import OrderedDict

pio.renderers.default = 'browser'
pd.options.display.precision = 4
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('max_columns', 12)
pd.options.mode.use_inf_as_na = True
pd.options.plotting.backend = "plotly"
pio.templates.default = "plotly"

plt_config = {"displayModeBar": False, "showTips": False}
start_dt = '2021-01-07'
end_dt = '2021-03-26'
gsmap = {"gs_green": "rgb(0, 204, 102)",
         "gs_drkgrn": "rgb(0, 102, 0)",
         "gs_brown": "rgb( 102, 00, 51)",
         "gs_grey": "rgb( 153, 153, 153)",
         "gs_ltgry": "rgb(204, 204, 204)",
         "gs_drkblue": "rgb(51, 51, 204)",
         "gs_ltblue": "rgb(0, 153, 255)",
         "gs_orange": "rgb(255, 102, 51)",
         "gs_magenta": "rgb(255, 51, 255)",
         "gs_purple": "rgb(102, 0, 153)",
         "gs_offblk": "rgb(51, 51, 51)",
         "gs_beige": "rgb(204, 204, 153)",
         "gs_red": "rgb(255, 51, 51)"}
gs_seq_colors: list = [[0, 'rgb(102, 0, 153)'], [0.3, 'rgb(0, 51, 153)'], [0.5, 'rgb(0, 204, 102)'],
                       [0.7, 'rgb(255, 102, 51)'], [1.0, 'rgb(255, 51, 255)']]
gs_bincolor: list = [[0, "rgb(51, 51, 204)"], [0.5, "rbg(0, 153, 255)"],
                     [1.0, "rgb(218, 51, 255)"]]

def do_histo(twl, col: str=None):
    """
    produces plotly histogram for one column or var
    :param twl: list of values
    :param col: name of column to plot
    :return:
    """
    import plotly.express as px
    fig = px.histogram(twl, x=col)
    fig.show()

    return

def convert_cloud_to_plotly(mpl_cld):
    """
    converts a matplotlib based word cloud to plotly figure
    :param mpl_cld: the mpl based wordcloud generated in gs_tweet_analysis.py
    :return: plotly figure for wordcloud
    """
    from plotly.tools import mpl_to_plotly
    cloud_fig = mpl_to_plotly(mpl_cld)

    return cloud_fig

def prep_trading_data(trade_f):
    """
    reads csv file with stock market data for company.  expected layout is trading
    date (mm/dd/yyyy), open price, daily high, daily low, closing price, volume of shares.
    Also includes two derived elements:  daily gain(loss) and daily exchange
    value, which is closing price * volume (shares traded).
    TODO: look into using openpyxl to directly read and manipulate excel files as pd.DF's
    TODO: use pd.options.io.excel.xlsx.reader = openpyxl, from openpyxl import ...
    :param trade_f: csv file GME_jan_mar.csv with market trading data
    :return: trade_df: pd.DataFrame of trading data for company
    """

    trade_df = pd.read_csv(trade_f, parse_dates=True, dtype={'close': float})
    pd.options.display.float_format = '{:.2f}'.format

    trade_df['date'] = trade_df['date'].astype('datetime64[ns]')
    trade_df.sort_values(by=['date'], inplace=True, ignore_index=True)

    trade_df['gain_adj'] = trade_df['gain'].apply(lambda x: round(x / trade_df['gain'].max(), ndigits=1))
    trade_df = rscale_col(trade_df, 'value')
    trade_df = rscale_col(trade_df, 'volume')
    trade_df['vol_adj'] = trade_df['volume_adj'].apply(lambda x: round((x + 2) * 4, ndigits=0))

    return trade_df

def prep_scored_tweets(twdf, opt: str="median"):
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
        if column in ['neg', 'pos', 'compound']:
            twdf[column + '_adj'] = twdf[column] + 1
        if column in ['neu']:
            twdf.drop(column, axis=1, inplace=True)

    sc1: int = 0
    if opt in ['both', 'median']:
        robustdf = do_scaling(twdf, "median")
        sc1 = 1
    if opt in ['both', 'mean']:
        scaledf = do_scaling(twdf, "mean")
        sc1 = sc1 | 2

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

    gs_lyt = go.Layout(height=850, width=1400,
                       title={'font': {'size': 36, 'family': 'Helvetica Neue UltraLight',
                                       'color': "rgb(51, 51, 51)"
                                       }
                              },
                       paper_bgcolor="rgb(204, 204, 204)",
                       font={'size': 18, 'family': 'Helvetica Neue UltraLight'},
                       hovermode="closest",
                       hoverdistance=10,
                       spikedistance=10,
                       showlegend=True,
                       legend={'title': {'font': {'size': 20, 'family': 'Helvetica Neue UltraLight'}},
                               'font': {'size': 18, 'family': 'Copperplate Light',
                                        'color': "rgb(51, 102, 153)"},
                               'bgcolor': "rgb(204, 204, 153)", 'bordercolor': "rgb(51, 51, 51)",
                               'borderwidth': 2, 'itemsizing': "trace"
                               },
                       xaxis={'title': {'font': {'size': 18, 'family': 'Helvetica Neue UltraLight'}},
                              'linecolor': "rgb(51, 51, 51)", 'rangemode': "normal",
                              'showspikes': True, 'spikethickness': 1,
                              },
                       yaxis={'title': {'font': {'size': 18, 'family': 'Helvetica Neue UltraLight'}},
                              'linecolor': "rgb(51, 51, 51)",
                              'showspikes': True, 'spikethickness': 1,
                              },
                       margin=go.layout.Margin(autoexpand=True)
                       )
    gs_lyt.template.data.scatter = [
        go.Scatter(marker=dict(symbol="diamond", size=10)),
        go.Scatter(marker=dict(symbol="circle", size=10)),
        go.Scatter(marker=dict(symbol="triangle-up", size=10)),
        go.Scatter(marker=dict(symbol="hexagon", size=10))
    ]
    gs_lyt.coloraxis.colorscale = [
        [0, '#0d0887'],
        [0.1111111111111111, '#46039f'],
        [0.2222222222222222, '#7201a8'],
        [0.3333333333333333, '#9c179e'],
        [0.4444444444444444, '#bd3786'],
        [0.5555555555555556, '#d8576b'],
        [0.6666666666666666, '#ed7953'],
        [0.7777777777777778, '#fb9f3a'],
        [0.8888888888888888, '#fdca26'],
        [1, '#f0f921']
    ]
    gs_lyt.colorway = ['#636efa', '#EF553B', '#00cc96', '#ab63fa', '#FFA15A', '#19d3f3',
                       '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    return gs_lyt

def plot_market_data(m_df, plyt: go.Layout):
    """
    creates bubble chart of time series trading price and volume
    :param m_df: pd_DataFrame with company trading data by date
    :param plyt: plotly go.Layout object instance
    :return:
    """
    trades: pd.DataFrame = m_df.copy()
    cfg: dict = {"displayModeBar": False, "showTips": False}
    lay: go.Layout = plyt
    maxsize: int=20

    lay.title.text = "GameStop Share Price, Volume and Gain/Loss from Jan through Mar 2021"
    lay.xaxis.title.text = "Market (Business) Days, Jan 08 to Mar 26, 2021"
    lay.yaxis.title.text = "$GME Shares Daily Closing Price "

    fig = go.Figure(data=go.Scatter(
        x=trades['date'].dt.strftime("%Y-%m-%d"),
        y=round(trades['close'], ndigits=1),
        mode='markers+text',
        hovertemplate='<i><b>GME Shares on %{x}</b></i>' +
                      '<br><b>closing price</b>: %{y:.2f}' +
                      "<br><b>trading volume</b>: %{customdata}" +
                      "<br><b>daily gain/loss</b>: %{meta}",
        hoverlabel={'font': {'family': 'Copperplate Light', 'size': 16}},
        marker=dict(
            color=trades['gain_adj'],
            opacity=0.7,
            sizemode='diameter',
            sizeref=max(m_df['vol_adj'])/maxsize,
            sizemin=4,
            size=trades['vol_adj'].apply(lambda x: round(x, ndigits=1)),
            symbol="circle",
            colorscale='Bluered',
            colorbar_title='<b>scaled daily<br>gain or loss</b>',
            colorbar={'bgcolor': 'rgb(204, 204, 204)',
                      'bordercolor': 'rgb(102, 102, 102)', 'borderwidth': 1, 'len': 0.5,
                      'lenmode': 'fraction', 'x': 1.01, 'xpad': 5,
                      'title': {'font': {'color': 'rgb( 102, 00, 51)', 'family': 'Copperplate Light',
                                         'size': 14}, 'text': 'daily gain(loss)<br>color scale'},
                      'tickfont': {'color': 'rgb( 102, 00, 51)', 'family': 'Copperplate Light',
                                   'size': 14}
             }
        ), meta=trades['gain'].apply(lambda x: '${:.2f}'.format(x)),
        customdata=trades['volume'].apply(lambda x: '{:,}'.format(x))
    ), layout=lay)

    fig.add_annotation(text="<b>Marker Size</b> scaled by shares traded<br>" +
                       "<b>Marker Color</b> scaled to daily gain/loss",
                       xref="paper", yref="paper",
                       font=dict(
                           family="Copperplate Light",
                           size=16,
                           color=gsmap["gs_offblk"]
                       ),
                       x=0.9, y=0.9, showarrow=False)
    fig.update_layout(lay)

    fig.show(config=cfg)

    return fig

def plot_scores(twdf: pd.DataFrame, lyout: go.Layout):
    """
    show box plots for sentiment scores of tweets
    :param twdf: pd.Dataframe with all score info
    :param lyout: plotly go.Layout object instance
    :return: None
    """
    cfg: dict = {"displayModeBar": False, "showTips": False}
    lay: go.Layout = lyout

    lay.title.text = "Vader Sentiment Score Quartile Distribution"
    lay.xaxis.title.text = "Quartile Plot.  Dashed Lines for Mean and Standard Deviation"
    lay.yaxis.title.text = "Type of Sentiment"

    fig = go.Figure(layout=lay)

    fig.add_trace(go.Box(x=twdf['neg'], quartilemethod="inclusive",
                         name="negative", marker_color='indianred',
                         boxmean='sd'))
    fig.add_trace(go.Box(x=twdf['pos'], quartilemethod="inclusive",
                         name="positive", marker_color='lightseagreen',
                         boxmean='sd'))
    fig.add_trace(go.Box(x=twdf['compound'], quartilemethod="inclusive",
                         name="compound", marker_color='royalblue',
                         boxmean='sd'))
    fig.update_traces(boxpoints='all', jitter=0.3)
    fig.show(config=cfg)

    return

def scatter_with_template(twdf, lyout: go.Layout):
    """
    plotly 3d chart with use of custom layout template
    :param twdf: pd.DataFrame
    :param lyout: plotly go.Layout object instance
    :return:
    """
    dfcp: pd.DataFrame = twdf.copy()
    cfg: dict = {"displayModeBar": False, "showTips": False}
    lay: go.Layout = lyout

    sent_stdv = dfcp['compound'].std()
    sent_mean = dfcp['compound'].mean()

    def sent_color(varx: float):
        """
        inner fx to calc color based on sentiment stat distribution
        :param varx: field from dataframe
        :return: rgb color
        """
        # stdev = 0.42
        # mean  = 0.142
        sent_mean = 0.14
        pcutoff = sent_mean + sent_stdv
        # pcutoff = 0.56
        ncutoff = sent_mean - sent_stdv
        # ncutoff = -0.28

        if varx > pcutoff:
            return "rgb(51, 155, 102)"
            # return 0.9
        elif varx > sent_mean:
            return "rgb(102, 102, 102)"
            # return 0.6
        elif varx > ncutoff:
            return "rgb(204, 102, 102)"
            # return 0.3
        else:
            return "rgb(255, 51, 51)"
            # return 0.0
        return

    dfcp['clr_scl'] = dfcp['compound'].apply(lambda x: sent_color(x))

    lay.title.text = "Influential Gamestop Tweets, January through March 2021"
    lay.xaxis.title.text = "Tweets from January 8 to March 26, 2021"
    lay.yaxis.title.text = "Log of Count for Tweet"
    # holidays are ['2021-01-18', '2021-02-15'] but let's try to get them programmatically:
    mktclosed = USFederalHolidayCalendar().holidays(start=start_dt, end=end_dt).astype("str").values
    lay.xaxis.rangebreaks = [{'bounds': ["sat", "mon"]},
                             {'values': mktclosed}]
    # lay.xaxis dtick in msec, 21600000.0 is every 6 hours
    lay.xaxis.type = "date"
    # lay.xaxis.autorange=True
    lay.xaxis.rangemode = "normal"
    lay.xaxis.dtick=604800000.0
    lay.xaxis.tick0="2021-01-07"
    # if tickmode="linear" then tick0 and dtick sets
    # lay.xaxis.tickmode="linear"
    lay.xaxis.tickformat="%B-%d %H:%M"
    # lay.xaxis.ticks="outside"

    fig = go.Figure(layout=lay)

    fig.add_scatter(x=dfcp['datetime'],
                    y=dfcp['qrr_log'], mode="markers",
                    name="Quote-Retweet-Reply count",
                    hovertemplate='<b>Quoted-Retweeted-Replied</b>' +
                                  "<br>Influence Score: %{y}",
                    marker=dict(symbol="diamond", size=10,
                                color=dfcp['clr_scl'])
                    )

    fig.add_scatter(x=dfcp['datetime'],
                    y=dfcp['fave_log'], mode="markers",
                    name="Liked count",
                    hovertemplate='<b>Tweet Likes</b>' +
                                  "<br>Scaled score: %{y}",
                    marker=dict(symbol="triangle-up", size=12,
                                color=dfcp['clr_scl'])
                    )

    fig.add_annotation(text="Influence = aggregate Quote/Retweet/Reply count<br>"
                            "or Favorite count in top 10% for date range",
                       xref="paper", yref="paper",
                       x=0.9, y=0.9, showarrow=False)
    # following show parms remove floating menu and unnecessary dialog box
    fig.show(config=cfg)

    return

def scatter_annotate(pbdf: pd.DataFrame, twdf: pd.DataFrame, plyt: go.Layout = None, appds: str = " "):
    """
    similar to scatter_with_template but adds second y axis with public event data

    :param pbdf: pd.DataFrame of public events
    :param twdf: pd.DataFrame of tweets with scaled metrics and sentiment
    :param plyt: plotly go.Layout object instance
    :param appds: str pass the name of dataset or active topic to clarify domain of plot
    :return:
    """

    size_ref = 18
    dfcp = twdf.copy()

    if not 'sent' in dfcp.columns:
        if 'datetime' in dfcp.columns:
            dcol: str = 'datetime'
        elif 'dt' in dfcp.columns:
            dcol: str = 'dt'
    else:
        dcol: str = 'sent'

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

    cfg: dict = {"displayModeBar": False, "showTips": False}
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    pb_x: list = list(pbdf.date.apply(lambda x: x.strftime("%b-%d")))
    pb_y: list = list(pbdf.close.apply(lambda x: round(x, ndigits=2)))

    dfrmt: str = "%b-%d"
    if 'date' in pbdf.columns:
        end_rng: dt.datetime = pbdf['date'].max()
        end_str = end_rng.strftime(dfrmt)
        strt_rng: dt.datetime = pbdf['date'].min()
        strt_str = strt_rng.strftime(dfrmt)
    else:
        end_str = 'Mar-26'
        strt_str = 'Jan-08'

    trace1 = go.Scattergl(x=dfcp[dcol], y=dfcp['qrr_log'],
                          mode="markers", xaxis="x", yaxis="y",
                          name="Quote-Retweet-Reply",
                          hovertemplate='<b>Quoted-Retweeted-Replied</b>' +
                                        "<br>Influence Score: %{y}",
                          marker=dict(symbol="diamond", size=11,
                                      color=dfcp['clr_scl'],
                                      line=dict(width=1,
                                                color="rgb(51, 51, 51)")
                                      ),
                          )

    trace2 = go.Scattergl(x=dfcp[dcol], y=dfcp['fave_log'],
                          xaxis="x", yaxis="y",
                          mode="markers", name="Favorited",
                          hovertemplate="Likes (Favorited) Score<br>Scaled Likes: %{y}",
                          marker=dict(symbol="triangle-up",
                                      size=11, color=dfcp['clr_scl']
                                      )
                          )

    trace3 = go.Scattergl(x=pb_x, y=pb_y, mode='markers',
                          xaxis='x', yaxis="y2",
                          name="Gamestop shares",
                          marker=dict(symbol="square", size=14, color="rgb(204, 204, 102)",
                                      line=dict(color="rgb(51, 51, 51)", width=1)
                             )
                  )

    data = [trace1, trace2, trace3]

    layout = go.Layout( xaxis=dict(domain=[0.1, 0.9], range=[strt_str, end_str],
                                   tickmode="linear",type = "date",
                                   autorange=False, tick0=strt_str,
                                   dtick=86400000, showticklabels=False,
                                   tickformat=dfrmt, tickangle=-60,
                                ),
             yaxis= dict(autorange=False, range=[0, 15],
                         tick0=0, dtick=1,
                         title_text="Scaled Influence"),
             yaxis2= dict(range=[15, 350], tick0=0, dtick=15,
                          title_text="Daily Share Price",
                          showticklabels=False, tickformat=".3"),
    )

    fig = go.Figure(data=data, layout=layout)
    fig.layout.title = "Influential Tweet Traffic and " + appds + "Share Price"
    fig.layout.legend.title = "Aggregate Metric"
    fig.update_xaxes(titlefont = dict(size=24, color="rgb(51, 51, 51)"))
    fig.show(config=cfg)

    return fig

def plot_3d_scatter(vlst, lyout: go.Layout, appds: str = ""):
    """
    plot_3d_scatter uses 3 dimensions plus size and color to graphically represent tweet
    attributes such as date and time, sentiment, quote/retweet/reply count, favorite count,
    and counts for original tweet if item is a retweet.
    :param vlst: pd.DataFrame with normalized (scaled) features
    :param lyout: plotly go.Layout object instance
    :return: None
    """
    cfg: dict = {"displayModeBar": False, "showTips": False}
    lay: go.Layout = lyout

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
        vlst['seq_color'] = vlst['type'].apply(lambda x: sent_color(x))
    else:
        vlst['seq_color'] = vlst['snt_clr']

    lay.title.text = 'Influential Tweets by<b>QT-RT-Reply and Favorite</b> Counts' + \
        "<br> and Sentiment-marker color, for " + str(appds) + " dataset"
    lay.scene = {'xaxis': {'title': 'Sorted, Jan 08 to Mar 26', 'spikethickness': 1},
                   'yaxis': {'title': 'quote-retweet-reply count-scaled', 'spikethickness': 1},
                   'zaxis': {'title': 'favorite (likes) count-scaled', 'spikethickness': 1},
                   'aspectmode': 'manual',
                   'aspectratio': {'x': 2, 'y': 1, 'z': 1}
                   }

    lay.xaxis.title.standoff = 8
    # lay.yaxis.title.standoff = 8
    lay.xaxis.automargin = False
    # lay.yaxis.automargin = False

    fig = go.Figure(data=go.Scatter3d(
        x=list(range(len(vlst))),
        y=vlst['qrr_scl'],
        z=vlst['fave_scl'],
        hovertemplate='x:<i><b>TweetID</b></i>: %{x}' +
                      '<br>y:<b>QRR Scale</b>: %{y:.2f}' +
                      '<br>z:<b>Favorite Scale</b>: %{z:.2f}' +
                      "<br><b>Compound Sentiment</b>: %{customdata}" +
                      "<br>%{text}",
        text=vlst['text'],
        name="Plot Tweets on topic<br>top 10% by QRR or Fave count<br>or Sentiment score",
        mode='markers',
        showlegend=False,
        marker=dict(
            sizemode='diameter',
            sizeref=round(fabs(vlst['compound'].max()) / 28, ndigits=1),
            sizemin=4,
            size=vlst['compound'].apply(lambda x: round(fabs(x) * 20, ndigits=1)),
            opacity=1,
            color=vlst['seq_color'],
        ),
        customdata=vlst['compound'],
        meta=vlst['dt'].apply(lambda x: dt.datetime.strftime(x, "%b %d, %Y"))
    ), layout=lay
    )
    fig.update_layout(lay)

    fig.show(config=cfg)

    return fig

def plot_scatter3d_date(vlst, lyout: go.Layout, appds: str = ""):
    """
    plot_3d_scatter uses 3 dimensions plus size and color to graphically represent tweet
    attributes such as date and time, sentiment, quote/retweet/reply count, favorite count,
    and counts for original tweet if item is a retweet.
    :param vlst: pd.DataFrame with normalized (scaled) features
    :param lyout: plotly go.Layout object instance
    :return: None
    """
    cfg: dict = {"displayModeBar": False, "showTips": False}
    lay: go.Layout = lyout

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
        vlst['seq_color'] = vlst['type'].apply(lambda x: sent_color(x))
    else:
        vlst['seq_color'] = vlst['snt_clr']

    lay.title.text = 'Influential Tweets, color-coded Sentiment for ' + str(appds)
    lay.scene = {'xaxis': {'title': 'Sorted, Jan 08 to Mar 26', 'spikethickness': 1},
                   'yaxis': {'title': 'QuotedT -ReTweet -Reply ct -scaled', 'spikethickness': 1},
                   'zaxis': {'title': 'favorite (likes) -scaled', 'spikethickness': 1},
                   'aspectmode': 'manual',
                   'aspectratio': {'x': 2, 'y': 1, 'z': 1}
                   }
    # aspectmode options are cube, manual, data, auto
    lay.xaxis.automargin = False
    lay.legend.title = str(appds) + " dataset"

    fig = go.Figure(data=go.Scatter3d(
        x=vlst['dt'],
        y=vlst['qrr_log'],
        z=vlst['fave_log'],
        hovertemplate='<i><b>Tweet on</b></i> %{meta}' +
                      '<br>y:<b>QRR Scale</b>: %{y:.2f}' +
                      '<br>z:<b>Favorite Scale</b>: %{z:.2f}' +
                      "<br><b>Compound Sentiment</b>: %{customdata:.2f}" +
                      "<br>%{text}",
        text=vlst['text'],
        name="Top 10% Tweets<br>Q-R-R or Fave count,<br>High Sentiment in Magenta",
        mode='markers',
        showlegend=False,
        marker=dict(
            sizemode='diameter',
            sizeref=round(fabs(vlst['compound'].max()) / 28, ndigits=1),
            sizemin=4,
            size=vlst['compound'].apply(lambda x: round(fabs(x) * 20, ndigits=1)),
            opacity=1,
            color=vlst['seq_color'],
        ),
        customdata=vlst['compound'],
        meta=vlst['dt'].apply(lambda x: dt.date.strftime(x, "%B %d, %Y"))
    ), layout=lay
    )
    fig.update_layout(lay)

    fig.show(config=cfg)

    return fig

def plot_scatter3d_crop(vlst, lyout: go.Layout):
    """
    plot_3d_scatter uses 3 dimensions plus size and color to graphically represent tweet
    attributes such as date and time, sentiment, quote/retweet/reply count, favorite count,
    and counts for original tweet if item is a retweet.
    :param vlst: pd.DataFrame with normalized (scaled) features
    :param lyout: plotly go.Layout object instance
    :return: None
    """
    cfg: dict = {"displayModeBar": False, "showTips": False}
    lay: go.Layout = lyout

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
        vlst['seq_color'] = vlst['type'].apply(lambda x: sent_color(x))
    else:
        vlst['seq_color'] = vlst['snt_clr']

    lay.title.text = 'Influential Tweets coded for Sentiment'
    lay.scene = {'xaxis': {'title': 'Sorted, Jan 08 to Mar 26', 'spikethickness': 1},
                 'yaxis': {'title': 'scaled quote-retweet-reply', 'spikethickness': 1},
                 'zaxis': {'title': 'scaled likes', 'spikethickness': 1},
                 'aspectmode': 'manual',
                 'aspectratio': {'x': 2, 'y': 1, 'z': 1}
                 }
    # aspectmode options are cube, manual, data, auto
    lay.xaxis.title.standoff = 8
    lay.yaxis.title.standoff = 8
    lay.xaxis.automargin = False
    lay.yaxis.automargin = False

    fig = go.Figure(data=go.Scatter3d(
        x=vlst['datetime'],
        y=vlst['qrr_log'],
        z=vlst['fave_log'],
        hovertemplate='<i><b>Tweet on</b></i> %{meta}' +
                      '<br>y:<b>QRR Scale</b>: %{y:.2f}' +
                      '<br>z:<b>Favorite Scale</b>: %{z:.2f}' +
                      "<br><b>Compound Sentiment</b>: %{customdata:.2f}" +
                      "<br>%{text}",
        text=vlst['text'],
        name="Top 10% Tweets<br>Q-R-R or Fave count,<br>High Sentiment in Magenta",
        mode='markers',
        showlegend=False,
        marker=dict(
            sizemode='diameter',
            sizeref=round(fabs(vlst['compound'].max()) / 28, ndigits=1),
            sizemin=4,
            size=vlst['compound'].apply(lambda x: round(fabs(x) * 20, ndigits=1)),
            opacity=1,
            color=vlst['seq_color'],
        ),
        customdata=vlst['compound'],
        meta=vlst['datetime'].apply(lambda x: dt.date.strftime(x, "%B %d, %Y"))
    ), layout=lay
    )
    fig.update_layout(lay)

    fig.show(config=cfg)

    return fig

def plot_multiple(m_df, tw_df, lyout: go.Layout):
    """
    uses plotly dual y-axis to plot both market data and tweet data
    :param m_df: pd.DataFrame with market data
    :param tw_df: pd.DataFrame with twitter data
    :param lyout: plotly go.Layout object instance
    :return: None
    """
    cfg: dict = {"displayModeBar": False, "showTips": False}
    lay: go.Layout = lyout

    def gain_clr(varx: float):
        """
        inner fx to return color based on attribute value
        :param varx: field from dataframe
        :return: rgb color
        """
        hi_gain = 10.00
        hi_loss = -10.00

        if varx > hi_gain:
            return "rgb(51, 153, 102)"
            # return 0.9
        elif varx > 0.00:
            return "rgb(204, 204, 153)"
            # return 0.6
        elif varx > hi_loss:
            return "rgb(153, 102, 102)"
            # return 0.3
        else:
            return "rgb(255, 51, 51)"
            # return 0.0
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    lay.margin = {'l': 50, 'r': 50, 't': 50, 'b': 50}
    lay.title.text = '$GME Closing Price/Share along with Influential Tweets with #GameStop or #GME'
    fig.update_yaxes(title_text="$GME Daily Price/Share at Market Close", secondary_y=False)
    fig.update_yaxes(title={'text': "Vader Compound Sentiment and Tweet influence",
                            'font': {'size': 28, 'family': 'Helvetica Neue UltraLight', 'color': gsmap["gs_drkblue"]}
                            }, secondary_y=True)

    m_df['gain_color'] = m_df['gain'].apply(lambda x: gain_clr(x))

    fig.add_trace(go.Scatter(
        x=m_df['date'],
        y=round(m_df['close'], ndigits=2),
        name="Gamestop (GME)<br>share price",
        text="$GME daily closing price",
        mode='markers',
        hovertemplate='<i><b>Stock Market for %{customdata}</b></i>' +
                      '<br>y:<b>closing price/share</b>: %{y:.2f}' +
                      "<br><b>trading volume</b>: %{meta}",
        hoverlabel={'font': {'family': 'Copperplate Light', 'size': 14}},
        marker=dict(
            color=m_df['gain_color'],
            sizemode="diameter",
            opacity=0.8,
            line=dict(color="rgb(51, 51, 51)",
                      width=1),
            sizemin=6,
            sizeref=round(m_df['vol_adj'].max()/24, ndigits=1),
            size=round(m_df['vol_adj'], ndigits=0),
            symbol="diamond"
        ), customdata=m_df['date'].apply(lambda x: dt.datetime.strftime(x, "%B %d")),
        meta=m_df['volume'].apply(lambda x: '{:,}'.format(x))
    ),
        secondary_y=False
    )

    fig.add_trace(go.Scatter(
        x=tw_df['dt'],
        y=tw_df['compound'],
        name="Vader<br>sentiment",
        mode='markers',
        hovertemplate='<i><b>Tweet on %{customdata}</b></i>' +
                      '<br>y:<b>comp sentiment</b>: %{y:.2f}' +
                      "<br><b>social influence</b>: %{meta}",
        hoverlabel={'font': {'family': 'Copperplate Light', 'size': 14}},
        marker=dict(
            color=gsmap["gs_ltblue"],
            sizemode='diameter',
            opacity=0.6,
            sizeref=round(tw_df['influence_log'].max()/24, ndigits=0),
            sizemin=4,
            size=tw_df['influence_log'].apply(lambda x: round(x, ndigits=0)),
            symbol="circle"
        ), customdata=tw_df['dt'].apply(lambda x: dt.datetime.strftime(x, "%B %d, %H:%M")),
        meta=tw_df['influence'].apply(lambda x: '{:,}'.format(x))
    ),
        secondary_y=True
    )

    fig.update_layout(lay, overwrite=False)
    fig.show(config=cfg)

    return None

def plot_ohlc(mdf, lyout: go.Layout):
    """
    preps and shows a financial plot with open, high, low and close data for stock
    :param m_df: market pd.DataFrame
    :param lyout: my custom plotly layout
    :return: plotly Figure
    """
    trades: pd.DataFrame = mdf.copy()
    cfg: dict = {"displayModeBar": False, "showTips": False}
    if lyout:
        lay: go.Layout = copy.copy(lyout)
    else:
        lay: go.Layout = create_layout()

    lay.title.text = "GameStop Daily Open, High, Low, and Closing Price"
    lay.xaxis.title.text = "Market (Business) Days, Jan 08 to Mar 26, 2021"
    lay.yaxis.title.text = "$GME Share Price"
    lay.xaxis.tickformat = "%Y-%b-%d"
    lay.xaxis.type = "date"
    lay.xaxis.autorange = False
    lay.xaxis.tickmode = "linear"
    lay.xaxis.tickangle = -75

    annot_rel = {
        'xref': 'paper', 'yref': 'paper',
        'x': 0.8, 'y': 0.9, 'xanchor': 'right', 'yanchor': 'top',
        'text': 'Gamestop shares daily open, high, low, and close<br>' +
                'Rally began week of January 11-15<br>' +
                'By March, GME was no longer giving back gains weekly',
        'bgcolor': "rgb(153, 153, 153)",
        'showarrow': False,
        'font': {'size': 14, 'color': "rgb(25, 25, 25)"}
    }
    lay.annotations = [annot_rel]
    fig3 = go.Figure(layout=lay)

    fig3.add_trace(go.Ohlc(
        x=trades['date'],
        xperiod=86400000,
        xperiodalignment="middle",
        xperiod0="2021-Jan-08",
        visible=True,
        open=trades['open'],
        high=trades['high'],
        low=trades['low'],
        close=trades['close']
    ))

    fig3.show(config=cfg)

    return fig3

def plot_tags(hashes, mentions):
    """
    shows the most frequent hashtags and user mentions in the dataset
    :param hashes: dict of hashtags with counts of occurrences
    :param mentions: dict of user mentions with counts of occurrences
    :return:
    """
    srt: list = sorted(hashes, key=lambda x: hashes[x], reverse=True)
    hashes: dict = {k: hashes[k] for k in srt}

    srt: list = sorted(mentions, key=lambda x: mentions[x], reverse=True)
    mentions: dict = {k: mentions[k] for k in srt}

    h_x: list = []
    h_y: list = []
    for k, v in hashes.items():
        h_x.append(k)
        h_y.append(v)
    h_x = h_x[:10]
    h_y = h_y[:10]

    m_x: list = []
    m_y: list = []
    for k, v in mentions.items():
        m_x.append(k)
        m_y.append(v)
    m_x = m_x[:10]
    m_y = m_y[:10]

    fighist = go.Figure()

    fighist.add_trace(go.Bar(name='hashtags', x=h_x, y=h_y, text=h_x,
                         marker=dict(color=h_y,
                                     coloraxis='coloraxis',
                                     line=dict(color='#333333', width=1)
                                     )
                         ))

    # fighist.add_histogram(x=m_x, y=m_y, text=m_x)
    fighist.update_layout(title_text='Top Hashtags in Twitter-Gamestop Data')
    fighist.show()

    return

def show_figure(gofig: go.Figure):
    """
    -displays plotly figure object constructed in this module's functions.
    -control of config and renderer settings for Plotly's fig.show()
    - config of run-time preferences for visualization or save to image or disk.
    - default renderers: 'json', 'png', 'svg', 'chrome', 'browser', 'sphinx_gallery'

    -also: plotly.mimetype to render on display in an iPython context
    :param gofig: plotly.graph_objects.Figure object
    :return: None
    """
    print(pio.renderers)
    pio.renderers.default = 'chrome+browser+png+svg'
    # pio.renderers.keys()
    # if following is set, will show plots on display in python
    # pio.renderers.render_on_display = True

    gs_rend = pio.renderers.default

    cfg: dict = {"displayModeBar": False, "showTips": False}

    gofig.show(renderer=gs_rend, config=cfg)

    return gofig

def bar_tags_mentions(hashes: dict, mentions: dict = None, plyt: go.Layout = None):
    """
    most frequent hashtags and user_mentions plotted with bar chart
    first- use utility in gs_tweet_analysis to sort descending plus filter out stoplists

    :param hashes: dict of hashtags with counts of occurrences
    :param mentions: dict of user mentions with counts of occurrences
    :param plyt: plotly layout instance, creates a copy so not to munge shared elements
    :return:
    """

    hsh_limit: int = 16
    u_mnt_limit: int = 8
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()
    lay.title = 'Top Hashtags and User Mentions<br>GameStop Tweets (common STOPs removed)'
    lay.yaxis.title = 'Count from Original Tweets and Quoted Tweet Comments'
    lay.xaxis.title = ""
    lay.title.font.size = 24
    lay.title.font.color = "rgb( 102, 51, 51)"
    lay.xaxis.tickangle = -40
    lay.font.size = 20
    lay.legend.itemsizing = 'trace'
    lay.legend.title = "Hahstags and User Mentions"
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
        for h, ct in zip(hashes.items(), range(hsh_limit)):
            h_x.append(h[0])
            h_y.append(h[1])
            h_c.append(round(float(random(size=1)), ndigits=2))
        if mentions:
            m_x: list = []
            m_y: list = []
            m_c: list = []
            for m, ct in zip(mentions.items(), range(u_mnt_limit)):
                m_x.append(m[0])
                m_y.append(m[1])
                m_c.append(round(float(random(size=1)), ndigits=2))

        fig.add_trace(go.Bar(name="hashtags (not GME or Gamestop)", x=h_x, y=h_y, text=h_x,
                             marker=dict(line_width=1, color=h_c, colorscale="viridis"),
                             texttemplate="%{x}<br>count: %{y}",
                             textposition="inside",
                             textangle=-90,
                             textfont=dict(size=20)
                             ))

        fig.add_trace(go.Bar(name="Twitter User Mentions", x=m_x, y=m_y, text=m_x,
                             marker=dict(line_width=2, color=m_c, colorscale="bluered"),
                             texttemplate="%{x}<br>count: %{y}",
                             textposition="inside",
                             textangle=-90,
                             textfont=dict(size=20)
                             ))
    fig.show(config=plt_config)

    return fig

def do_cloud(batch_tw_wrds, opt_stops: str = None, maxwrd: int=80):
    """
    wordcloud package options can be explored via '?wordcloud' (python- show docstring)
    background_color="white" - lighter background makes smaller words more legible,
    max_words= this can prevent over clutter, mask=shape the cloud to an image,
    stopwords=ad-hoc removal of unwanted words, contour_width=3,
    :param batch_tw_wrds: list of list of word tokens for tweets
    :param opt_stops: str var name for optional stop list
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
    # may be able to put this in plotly with the following
    # cld_img = wc.to_array()
    # go.Figure()
    # go.imshow(cld_img, interpolation="bilinear")
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcld)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    return

def do_sent_classify(df: pd.DataFrame, clrcol: str= "compound"):
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