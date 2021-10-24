"""
gs_Tweet2Vec applies nlp models to analyze twitter feeds I've collected on specific topics
using my twitter dev api access.  I mainly use tools from NLTK and google word2vec.
see also my gs_Sentiment app for custom nlp tools and analysis
"""
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from tweet_data_dict import GMEDIR, STOP_ADD, STOP_TWEET, STOPS_ESL, STOP_NONALPHA
import gs_nlp_util as gsutil
import gs_tweet_analysis as gsta
import gs_Tweet2Insight as t2i
import gs_Plot_Tweets as gsPT

# BEGIN SCRIPT: set boolean vars for script control
run_tokenization: bool = True
run_limit_retweets: bool = True
run_word_token: bool = True
run_sentiment: bool = True
run_scaling: bool = True
plot_multi: bool = True

run_tfidf: bool = True
run_cloud: bool = True
run_tfbydate: bool = False

# point ESLDIR to Tweet .json folder, also: specify project-specific STOP lists
mkt_data_fil = GMEDIR + "GME_jan_mar.csv"
archive_qualifier: str = "GME_gstweet"
module_path = "/Users/bgh/dev/pydev/gsTweet/models"
inputfiles: list = gsutil.get_file_list(GMEDIR)
tw_raw: list = gsutil.get_tweets_from_json(inputfiles, GMEDIR)
tweets_pre: list = gsutil.get_fields_simple(tw_raw, debug=False)
gsutil.save_recs(tweets_pre, archive_qualifier)

if run_tokenization:
    # clean_text and remove_replace_text scrub Tweets- tricky process to not lose meaning
    gsutil.box_prn("gsTweet Pre-Processing and Tokenization")
    tweets_cln = gsutil.clean_text(tweets_pre)
    tweets_post = gsutil.remove_replace_text(tweets_cln, stop2=STOP_ADD)
    tweets_flat = gsutil.flatten_twdict(tweets_post)
    # wrds = gsutil.do_wrd_tok(tweets_flat)

    if run_limit_retweets:
        # creates corpus with only one copy of each ReTweet
        tweets_nodupe, RT_count = gsta.limit_retweets(tweets_post)
        missing_tweets = gsta.id_originating_tweets(tweets_post, RT_count, tweets_nodupe)
        oneRT_check = gsutil.remove_replace_text(tweets_nodupe, stop2=STOP_ADD)
        tw_days = gsutil.get_dates_in_batch(tweets_nodupe)
        tweetdf: pd.DataFrame = gsta.create_dataframe(oneRT_check)
        hash_list, mention_list = t2i.get_top_tags(oneRT_check)

        filt_df = gsta.crop_df_to_date(tweetdf, "2021-01-11 06:00", "2021-03-26 18:00")
        filt_lst: list = filt_df.to_dict("records")

        if run_word_token:
            flat_oneRT = gsutil.flatten_twdict(oneRT_check)
            words_oneRT = gsutil.do_wrd_tok(oneRT_check)

    if run_sentiment:
        # Uses nltk-Vader for sentiment. summarize_vader displays sentiment by type
        print("\n")
        gsutil.box_prn("gsTweet Sentiment Scoring and Analysis Section")
        tw_sentiment = gsta.apply_vader(filt_lst)
        gsta.summarize_vader(tw_sentiment)

        if run_scaling:
            top_qrr = gsta.get_pctle_qrr(tw_sentiment, ptile=85)
            open_qrr = gsta.get_pctle_qrr(tw_sentiment, ptile=80)

            top_fave = gsta.get_pctle_fave(tw_sentiment, ptile=85)
            open_fave = gsta.get_pctle_fave(tw_sentiment, ptile=80)

            top_sent = gsta.get_pctle_sentiment(tw_sentiment, ptile=67)
            neg_sent = gsta.get_neg_sentiment(tw_sentiment, cutoff=0.2)

            neg_combined = t2i.get_combined_toplist(open_qrr, open_fave, neg_sent)
            neg_topdf = gsta.create_dataframe(neg_combined)
            negscaledf = gsPT.prep_scored_tweets(neg_topdf, opt="median")

            top_combined = t2i.get_combined_toplist(top_qrr, top_fave, top_sent)
            combo_topdf = gsta.create_dataframe(top_combined)
            combo_rbdf, combo_scdf = gsPT.prep_scored_tweets(combo_topdf, opt="both")

            # final_combine = t2i.final_toplist(top_qrr, top_fave)
            # final_topdf = gsta.create_dataframe(final_combine)

            neg_len: int = len(negscaledf)
            print("len of neg sent df is %d" % neg_len)
            negscaledf = negscaledf[negscaledf['qrr'] != 0]
            print("    %d rows with qrr=0 removed" % (neg_len - len(negscaledf)))
            neg_len = len(negscaledf)
            negscaledf = negscaledf[negscaledf['fave'] != 0]
            print("    %d rows with fave=0 removed" % (neg_len - len(negscaledf)))
            neg_cropdf = gsta.crop_df_to_date(negscaledf, "2021-01-11 06:00", "2021-02-12 23:00")

            trade_df = gsPT.prep_trading_data(mkt_data_fil)
            mylayout = gsPT.create_layout()

            combo_jan = gsta.crop_df_to_date(tweetdf, "2021-01-11 06:00", "2021-02-12 23:00")
            combo_jan.sort_values('datetime')
            combo_jan.reset_index(drop=True, inplace=True)
            jan_lst = combo_jan.to_dict("records")
            sntmt_jan = gsta.apply_vader(jan_lst)
            jan_pctl = gsta.get_pctle_qrr(sntmt_jan, ptile=75)
            sntjan_df = gsta.create_dataframe(jan_pctl)
            robjandf = gsPT.prep_scored_tweets(sntjan_df, opt="median")
            robjandf = gsPT.do_sent_classify(robjandf)

            if plot_multi:
                # fig_md = gsPT.plot_market_data(trade_df, mylayout)
                # gsPT.plot_scores(robustdf, mylayout)
                # gsPT.plot_tags(hash_list, mention_list)
                gsPT.plot_ohlc(trade_df, mylayout)

                # scatter-template has red-green diamond-triangle scatter like
                # gsPT.scatter_with_template(neg_cropdf, mylayout)
                # gsPT.scatter_with_template(robustdf, mylayout)
                gsPT.scatter_with_template(combo_rbdf, mylayout)

                sc_anotfig = gsPT.scatter_annotate(trade_df, combo_rbdf, plyt=mylayout, appds=" Gamestop")

                # fig_3d = gsPT.plot_3d_scatter(robjandf, mylayout, appds=" Gamestop Jan")
                neg_3d = gsPT.plot_3d_scatter(negscaledf, mylayout, appds=" Neg sentiment mash-up")


                # rb_scat3 = gsPT.plot_scatter3d_date(combo_rbdf, mylayout, appds="Gamestop")


                # fig_3d = gsPT.plot_scatter3d_date(negscaledf, mylayout, appds=" neg sentmnt GME")
                fig_3d = gsPT.plot_scatter3d_date(robjandf, mylayout, appds=" GME jan 2021")

                # gsPT.plot_multiple(trade_df, combo_rbdf, mylayout)

        if run_tfidf:
            gsutil.box_prn("TF*IDF using oneRT corpus...determining importance of words")
            # FIRST, clear any additional stop words found in reviewing word tokens
            oneRT_filter = gsutil.remove_replace_text(tweets_nodupe, stop2=STOP_ADD)
            oneRT_clean = gsutil.do_stops(oneRT_filter, stop1=STOP_TWEET, stop2=STOPS_ESL)
            oneRT_clean2 = gsutil.do_start_stops(oneRT_clean)

            words_rtclean = gsutil.do_wrd_tok(oneRT_clean)
            wrd_freq = t2i.calc_tf(words_rtclean, word_tokens=True, calctyp="COUNT")
            tws_per_wrd = t2i.count_tweets_for_word(wrd_freq)
            idf_by_tw = t2i.calc_idf(wrd_freq, tws_per_wrd)
            tf_idf = t2i.calc_tf_idf(wrd_freq, idf_by_tw)
            tfi_final: dict[str, float] = t2i.calc_single_tfidf(tf_idf, calctyp="SUM")
            TFI_STOP = t2i.do_tfidf_stops(tfi_final)
            gsutil.box_prn("tfidf added %d stop words" % len(TFI_STOP))
            # can build stop list of 'junk' words with following:
            # STOP_NONALPHA: list = []
            # for x in tfi_final.keys():
            #    if not str(x).isalpha():
            #        STOP_NONALPHA.append(x)
            tweet_tfistop = gsutil.do_stops(oneRT_clean2, stop1=TFI_STOP, stop2=STOP_NONALPHA)

        if run_cloud:
            # handy to see what we have after pre-processing and filtering
            STOP_CLD = ['you', 'gme', 'GME', 'new', 'one', 'when', 'stock', 'day',
                        'out', 'who', 'more']
            words_cld = gsutil.do_stops(words_rtclean, stop1=STOP_CLD)
            words_cld = t2i.cleanup_for_cloud(words_cld)
            gsPT.do_cloud(words_cld, opt_stops=TFI_STOP, maxwrd=100)
            word_freq = gsta.get_word_freq(words_rtclean)
            # gsta.do_cloud(words_rtclean)

        if run_tfbydate:
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
