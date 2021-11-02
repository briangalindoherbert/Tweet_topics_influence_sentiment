"""
import, clean, and filter Twitter data (tweets plus metadata) on specific topics.
builds and reports metrics on 'Influence'- social approval and sharing of tweets
it also calculates and reports on Sentiment of tweets, using nltk-Vader.
import either stock market data or public event data for the topic and produce a range
of plots in plotly to show the topic dataset of Tweets and the event data.

I've tried to isolate changes necessary to run this app for a new/different dataset:
    tweet_data_dict.py holds many variable definitions for 'constants' like folder
    locations. also this main script controls the flow and constants to use.
"""
import datetime as dt
import pandas as pd

from tweet_data_dict import ESLDIR, STOP_ADD, STOP_TWEET, STOPS_ESL, STOP_NONALPHA, GS_STOP
import gs_nlp_util as gsutil
import gs_tweet_analysis as gsta
import gs_Tweet2Insight as t2i
import gs_Plot_Tweets as gsPT
import timeit
# import gs_Vector_models as gsvm

# BEGIN SCRIPT: set boolean vars for script control
clean_raw_dataset: bool = True
crop_split_dates: bool = True
run_word_token: bool = True
run_sentiment: bool = True
run_scaling: bool = True
plot_prep: bool = True
# visualization components:
show_plots: bool = True
run_tfidf: bool = False
run_cloud: bool = False
run_tfidf_bydate: bool = False

# point ESLDIR to Tweet .json folder, also: specify project-specific STOP lists
ttopic: str = "Superleague Launch April18-21"
events_f: str = ESLDIR + "ESL_apr.csv"
archive_qualifier: str = "ESL_Octrun"
module_path = "/Users/bgh/dev/pydev/superleague/models"
pd.options.plotting.backend = "plotly"

gsutil.box_prn("building Tweet analytics dataset: XXX, importing data".replace("XXX", ttopic))
inputfiles: list = gsutil.get_file_list(ESLDIR)
tw_raw: list = gsutil.get_tweets_from_json(inputfiles, ESLDIR)
tweets_pre: list = gsutil.get_fields_simple(tw_raw, debug=False)
gsutil.save_recs(tweets_pre, archive_qualifier)
pub_df = gsutil.prep_public_data(events_f)

if clean_raw_dataset:
    # clean_text and remove_replace_text scrub Tweets- tricky process to not lose meaning
    gsutil.box_prn("cleaning, filtering and wrangling raw Tweet dataset")
    tweets_cln = gsutil.clean_text(tweets_pre)
    tweets_post = gsutil.remove_replace_text(tweets_cln,stop1=GS_STOP, stop2=STOP_TWEET)
    tweets_flat = gsutil.flatten_twdict(tweets_post)

    # Step 1: filter ReTweets and remove off-topic Tweets - return filtered dataset
    # Step 2: look for originating tweets for all 'shared' content (RT, QT, Reply)
    # 2nd step identifies missing tweets: find_original_ids replaced id_originating_tweets
    tw_rtfilt, rt_count, qt_count, qt_merge = gsta.filter_rt_topic(tweets_post)
    print("length of dataset after RT filter is %d" % len(tw_rtfilt))
    missing, not_missing = gsta.find_original_ids(tw_rtfilt, rt_count, qt_count)

    onert_cln = gsutil.remove_replace_text(tw_rtfilt, stop1=STOP_ADD, stop2=STOPS_ESL)
    hash_list, mention_list = t2i.get_top_tags(tw_rtfilt)
    filtrdates: list = []
    for tw in onert_cln:
        if tw['date'] in ['2021-04-18', '2021-04-19', '2021-04-20', '2021-04-21']:
            filtrdates.append(tw)
    print("filtrdates: list of dict of tweets cropped to dates has %d records" % len(filtrdates))

    # date_name is the datetime column to use as date
    date_name: str = "sent"
    tw_days = gsutil.get_dates_in_batch(filtrdates)
    tweetdf: pd.DataFrame = gsta.create_dataframe(filtrdates)
    tweetdf.sort_values(date_name, inplace=True, ignore_index=True)
    tweetdf.reset_index(inplace=True, drop=True)
    tweet_4d = tweetdf.copy(deep=True)
    tweet_3d = gsta.crop_df_to_date(tweetdf, "2021-04-18 06:00", "2021-04-20 23:59")
    t4d_lst: list = tweet_4d.to_dict("records")
    print("    4-day dataset has %d records" % len(tweet_4d))
    t3d_lst: list = tweet_3d.to_dict("records")
    print("    3-day dataset has %d records" % len(tweet_3d))

if run_word_token:
    flat_t4d = gsutil.flatten_twdict(t4d_lst)
    words_t4d = gsutil.do_wrd_tok(flat_t4d)

if run_sentiment:
    # Uses nltk-Vader for sentiment. summarize_vader displays sentiment by type
    gsutil.box_prn("\nRunning Sentiment Cale for %d Tweets in %s dataset" % (len(t4d_lst), ttopic))
    tsnt_4d = gsta.apply_vader(t4d_lst)
    gsta.summarize_vader(tsnt_4d)

    tsnt_3d = gsta.apply_vader(t3d_lst)
    gsta.summarize_vader(tsnt_3d, top_lim=10)

if run_scaling:
    top_qrr = gsta.get_pctle_qrr(tsnt_4d, ptile=90)
    open_qrr = gsta.get_pctle_qrr(tsnt_4d, ptile=80)
    d3_qrr = gsta.get_pctle_qrr(tsnt_3d, ptile=80)

    top_fave = gsta.get_pctle_fave(tsnt_4d, ptile=90)
    open_fave = gsta.get_pctle_fave(tsnt_4d, ptile=80)
    d3_fave = gsta.get_pctle_fave(tsnt_3d, ptile=80)

    top_snt4 = gsta.get_pctle_sentiment(tsnt_4d, ptile=65)
    top_snt3 = gsta.get_pctle_sentiment(tsnt_3d, ptile=65)
    top_neg3 = gsta.get_neg_sentiment(tsnt_3d, cutoff=0.2)

    neg_combo3 = t2i.get_combined_toplist(d3_qrr, d3_fave, top_neg3)
    top_combo3 = t2i.get_combined_toplist(d3_qrr, d3_fave, top_snt3)
    top_combo4 = t2i.get_combined_toplist(top_qrr, top_fave, top_snt4)

    neg3topdf = gsta.create_dataframe(neg_combo3)
    negscordf = gsPT.prep_scored_tweets(neg3topdf, opt="median")

    combo3topdf = gsta.create_dataframe(top_combo3)
    comb3_df = gsPT.prep_scored_tweets(combo3topdf, opt="median")

    combo4topdf = gsta.create_dataframe(top_combo4)
    comb4_rbdf, combo_scdf = gsPT.prep_scored_tweets(combo4topdf, opt="both")

    pub_3day = gsta.crop_df_to_date(pub_df, strtd="2021-04-18 11:00", endd="2021-04-20 23:59")

    tsnt3d_df = gsta.create_dataframe(tsnt_3d)
    # can further refine with final toplist if more filtering wanted
    # final_combined = t2i.final_toplist(top_qrr, top_fave)
    # final_topdf = gsta.create_dataframe(final_combined)
    # final_rbdf, final_scdf = gsPT.prep_scored_tweets(final_topdf, opt="both")

if plot_prep:
    # prep functions for various plots:
    plt_lay = gsPT.create_layout()
    domain_stops: list = ['acmilan', 'afc', 'arsenal', 'cfc', 'championsleague',
                         'chelsea', 'efl', 'elt', 'epl', 'esl',
                         'europeansuperleague', 'europesuperleague',
                         'fifa', 'football', 'juventus', 'lfc', 'liverpool',
                         'mancity', 'manutd', 'mcfc', 'mufc', 'nufc', 'pl',
                         'premierleague', 'psg', 'realmadrid', 'seriea',
                         'spurs', 'superleague', 'superlega', 'superliga',
                         'tefl', 'thfc', 'ucl', 'uefa']
    hash_stops: list = ['learnenglish', 'vocabulary', 'uk', 'vocab',
                        'speakenglish', 'english', 'europe']
    hash_cln: dict = gsta.clean_hash_and_mentions(hash_list, domain_stops)
    hash_cln: dict = gsta.clean_hash_and_mentions(hash_cln, hash_stops)
    # buckets = gsta.bucket_lst(top_combo4)

    final4_df = gsPT.prep_pandas(comb4_rbdf)
    final3_df = gsPT.prep_pandas(comb3_df)
    finneg_df = gsPT.prep_pandas(negscordf)

if show_plots:
    # gsPT.box_sentiments(combo_rbdf, plt_lay)
    # barfig = gsPT.bar_tags_mentions(hash_cln, mention_list, plt_lay)

    # gsPT.scatter_with_template(finneg_df, plt_lay)
    # gsPT.scatter_with_template(final4_df, plt_lay)

    # mlt2_fig = gsPT.plot_esl_specific(pub_df, final4_df, plt_lay)
    # mlt2n_fig = gsPT.plot_esl_specific(pub_df, final4_df, plt_lay)

    new_lyt = gsPT.create_layout()
    # bubble chart with yaxis sentiment, marker size=influence, intuitive
    # fin_mult = gsPT.plot_multiple(final3_df, new_lyt, ccol="compound")

    # scatter_annotate: 2d, two metric traces- diamond and triangle, red-green
    # sctneg_fig = gsPT.scatter_with_events(pub_3day, finneg_df, plyt=new_lyt, appd="Superleague NEG Sentiment")
    # sctant3_fig = gsPT.scatter_with_events(pub_df, final3_df, plyt=new_lyt, appd="Superleague 3-day")
    sctant4_fig = gsPT.scatter_with_events(pub_3day, final3_df, plyt=new_lyt, appd="Euro-Superleague")

    gsPT.histogram_metrics(combo3topdf, plyt=plt_lay, appd="Superleague Tweets")

    gsPT.box_qrr_fave(tweetdf, plyt=plt_lay, appd="Superleague Tweets")
    # use this to switch axes put sentiment on first y
    fig_3d = gsPT.scat_sntmnt_y_qrrf_mrkr_evts(pub_3day, comb3_df, plyt=new_lyt)

    fig_scatd = gsPT.scatter3d_bydate(comb3_df, plt_lay, appd="Superleague")
    # fig_neg = gsPT.scatter3d_bydate(finneg_df, plt_lay, appd="ESL neg_snt 3 day", stype="neg")
    # fig_neg = gsPT.scatter3d_bydate(final3_df, plt_lay, appd="3 day esl", stype="compound")

    # styp_df, qf_df, qorf_df = gsta.split_toplist_bytyp(combo_rbdf)
    comb_3d = gsPT.p3d_new(comb3_df, plyt=plt_lay, appd="Superleague")
    # fin_3d = gsPT.plot_3d(final4_df, plt_lay, appd=ttopic)
    # f3d = gsPT.plot_3d(finneg_df, plt_lay, appd="ESL Top Neg Sentiment Only", styp="compound")

if run_tfidf:
    gsutil.box_prn("TF*IDF using oneRT corpus...determining importance of words")
    # FIRST, clear any additional stop words found in reviewing word tokens
    # build this manually from stop words seen during tasks:
    ad_hoc = ['after', 'been', 'before', 'next']
    limrt_cln = gsutil.remove_replace_text(words_t4d, stop2=STOP_ADD)
    limrt_cln = gsutil.do_stops(limrt_cln, stop1=STOPS_ESL, stop2=STOP_TWEET)
    limrt_cln = gsutil.do_stops(limrt_cln, stop1=GS_STOP, stop2=ad_hoc)
    inp_tfidf = gsutil.do_start_stops(limrt_cln)

    words_rtclean = gsutil.do_wrd_tok(inp_tfidf)
    wrd_freq = t2i.calc_tf(words_rtclean, word_tokens=True, calctyp="TOP")
    tws_per_wrd = t2i.count_tweets_for_word(wrd_freq)
    idf_by_tw = t2i.calc_idf(wrd_freq, tws_per_wrd)
    tf_idf = t2i.calc_tf_idf(wrd_freq, idf_by_tw)
    tfi_final: dict[str, float] = t2i.calc_single_tfidf(tf_idf, calctyp="SUM")
    TFI_STOP = t2i.do_tfidf_stops(tfi_final)
    gsutil.box_prn("tfidf added %d stop words" % len(TFI_STOP))
    # can build stop list of 'junk' words with following:
    STOP_NONALPHA: list = []
    for x in tfi_final.keys():
        if not str(x).isalpha():
            STOP_NONALPHA.append(x)
    tweet_tfistop = gsutil.do_stops(inp_tfidf, stop1=TFI_STOP, stop2=STOP_NONALPHA)

if run_cloud:

    flat_tfistop = gsutil.flatten_twdict(tweet_tfistop)
    oneRT_flat = gsutil.do_stops(flat_tfistop, stop1=TFI_STOP, stop2=STOP_NONALPHA)
    oneRT_cloud = t2i.scrub_cloud(oneRT_flat)
    # handy to see what we have after pre-processing and filtering
    word_freq = gsta.get_word_freq(oneRT_cloud)
    word_freq = {k: word_freq[k] for k in sorted(word_freq, key=lambda x: word_freq[x], reverse=True)}
    print("word_freq is STOPPED and sorted by descending word frequency")
    gsPT.do_cloud(oneRT_cloud, opt_stops=STOPS_ESL, maxwrd=125)

if run_tfidf_bydate:
    # alternate corpus by_date: aggregate Tweets by calendar day, optionally filter RT's
    Tweets_bydt, RT_cnt_bydt, qrr_bydt = gsutil.groupby_date(tweets_post, one_rt=True)
    Tweets_bydt_check = gsutil.remove_replace_text(Tweets_bydt, stop2=STOP_ADD)
    flat_bydt = gsutil.flatten_twdict(Tweets_bydt)
    words_bydate = gsutil.do_wrd_tok(Tweets_bydt)

    gsutil.box_prn("TF*IDF applying to Tweets aggregated by day sent")
    freq_bydate = t2i.calc_tf(Tweets_bydt, word_tokens=False, calctyp="UNIQ")
    tw_to_wrd_date = t2i.count_tweets_for_word(freq_bydate)
    idf_bydate = t2i.calc_idf(freq_bydate, tw_to_wrd_date)
    tfidf_bydate = t2i.calc_tf_idf(freq_bydate, idf_bydate)
    tfifinal_bydate: dict[str, float] = t2i.calc_single_tfidf(tfidf_bydate)
    STOP_NORT = t2i.do_tfidf_stops(tfifinal_bydate)
    gsutil.box_prn("tfidf on noretweet by date: %d stop words" % len(STOP_NORT))
    wrds_bydt_final = gsutil.do_stops(words_bydate, stop1=STOP_NORT, stop2=None)
