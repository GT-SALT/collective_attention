"""
Helper methods for models.
"""
from __future__ import division
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_helpers import compute_distances, geo_lookup, normalize_str, query_norm
# Python 2
# from itertools import izip
izip = zip
import numpy as np
import pandas as pd
from geopy.distance import great_circle
# import torch
# python 2
# from torch import Tensor, LongTensor
# from torch.autograd import Variable
# from vae_lab.gpsvae import approx_log_px
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics.ranking import auc
from scipy.special import logsumexp
from scipy.stats import norm, chi
from scipy.stats.distributions import chi2
from scipy.sparse import diags, csr_matrix, linalg
# from allennlp.modules.elmo import batch_to_ids, Elmo
import logging
# from mpl_toolkits.basemap import Basemap
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import Binomial
from statsmodels.genmod.families.links import logit
from statsmodels.genmod.families.links import Logit
from sklearn.model_selection import KFold
import re
from IPython.display import Markdown, display

## disambiguation
def disambiguate_mention(topo_mention, city_mentions, geo_data, geo_lexicon, candidate_k=100, output_k=10):
    """
    Return toponyms in geo data with closest average distance to
    all cities referenced.
    
    Parameters:
    -----------
    topo_mention : str
    city_mentions : str
    geo_data : pandas.DataFrame
    Combined data from OSM and Geonames files.
    geo_lexicon : list
    candidate_k : int
    output_k : int
    
    Returns:
    --------
    topo_data : pandas.DataFrame
    Data frame with toponym name, ID, lat, lon.
    """
    # get candidate toponyms
    candidate_topo_mentions = geo_lookup(topo_mention, geo_lexicon, k=candidate_k, word_char=False)
    # compute dist
    topo_data = pd.DataFrame()
    for c in candidate_topo_mentions:
        topo_data_c = compute_distances(c, city_mentions, geo_data)
        if(topo_data.shape[0]==0):
            topo_data = topo_data_c.copy()
        else:
            topo_data = topo_data.append(topo_data_c)
    # return top-k
    topo_data.sort_values('dist', inplace=True, ascending=True)
    topo_data = topo_data.iloc[0:output_k, :]
    return topo_data

def get_at_k(topo_data_list, gold_labels):
    """
    Compute precision at K over list of toponym candidates and gold labels.
    
    Parameters:
    -----------
    topo_data_list : [pandas.DataFrame]
    gold_labels : [int]
    
    Returns:
    --------
    precision : float
    """
    true_positive = 0
    N = len(gold_labels)
    for topo_data, gold_label in izip(topo_data_list, gold_labels):
        match = int(gold_label in topo_data.loc[:, 'id'])
        true_positive += match
    precision = true_positive / N
    return precision

def get_mean_dist(topo_data_list, gold_locs):
    """
    Compute mean distance between the top toponym's location
    and the gold location.
    
    Parameters:
    -----------
    topo_data_list : [pandas.Series]
    gold_locs : [(float, float)]

    Returns:
    --------
    mean_dist : float
    std_dist : float
    """
    dists = []
    for topo_data, gold_loc in izip(topo_data_list, gold_locs):
        t_loc = topo_data.loc[['lat', 'lon']].values.tolist()
        dist = great_circle(t_loc, gold_loc).miles
        dists.append(dist)
    mean_dist = np.mean(dists)
    std_dist = np.std(dists)
    return mean_dist, std_dist

def great_circle_error(p1, p2):
    """
    Compute error between points in miles.
    
    Parameters:
    -----------
    p1 : list
    p2 : list
    
    Returns:
    --------
    error : float
    """
    error = great_circle(p1, p2).miles
    return error

def generate_score_candidates(query_data, candidate_generator, scorers, feature_name, k_dists=None, k_ranks=None):
    """
    Generate and score candidates from queries.
    
    Parameters:
    -----------
    query_data : pandas.DataFrame
    candidate_generator : CandidateGenerator
    scorers : list
    List of scoring modules.
    dist_scorer : CandidateScorer
    Score top candidates based on precision-at-k miles.
    rank_scorer : CandidateScorerRanked
    Score all candidates based on precision-at-k rank.
    mrr_scorer : CandidateScorerMeanRank
    Score all candidates based on mean reciprocal rank.
    feature_name : str
    Name of feature to use in scoring.
    k_dists : list
    Precision-at-k miles.
    k_ranks : list
    Precision-at-k ranks.
    
    Returns:
    --------
    error_data : pandas.Series
    All error data from scorers.
    top_candidate_score_data : pandas.DataFrame
    For all top candidates, data on distance error and feature score.
    gold_topo_missing_data : pandas.DataFrame
    For the ranking score: for all candidate sets 
    which did not include the gold toponym, 
    return the rank and score for each candidate.
    """
    top_k = max(k_ranks) # top candidates from ranked list
    top_candidates = []
    all_candidates = []
    gold_topo_ids = query_data.loc[:, 'geoNameId'].values.tolist()
    if(len(gold_topo_ids) == 0):
        print('error with gold topo IDs in data:\n %s'%(query_data.head()))
    gold_topo_lat_lons = query_data.loc[:, ['latitude', 'longitude']].values.tolist()
    for q, q_row in query_data.iterrows():
        q_query = q_row.loc['text']
        q_lat_lon = q_row.loc[['latitude', 'longitude']]
        q_candidates = candidate_generator.generate_candidates(q_query)
        q_candidates.sort_candidates(feature_name)
        q_top_candidates = q_candidates.get_top_k_candidates(k=top_k)
        q_top_candidate = q_top_candidates[0]
        top_candidates.append(q_top_candidate)
        all_candidates.append(q_top_candidates)
        if(len(all_candidates) % 100 == 0):
            print('processed %d toponyms'%(len(all_candidates)))

    ## score candidates!
    error_data = []
    # add feature data
    error_data.append(feature_name)
    for scorer in scorers:
        if(scorer.__class__ is CandidateScorer.CandidateScorer and k_dists is not None):    
            # distance precision
            scorer.compute_errors(top_candidates, gold_topo_lat_lons)
            dist_k_scores = []
            for k_dist in k_dists:
                score = scorer.score(k=k_dist)
                dist_k_scores.append(score)
            error_data += dist_k_scores
        elif(scorer.__class__ is CandidateScorer.CandidateScorerRanked and k_ranks is not None):            
            # precision-at-k 
            rank_k_scores = []
            for k_rank in k_ranks:
                score = scorer.score(all_candidates, gold_topo_ids, k=k_rank)
                rank_k_scores.append(score)
            error_data += rank_k_scores
        elif(scorer.__class__ is CandidateScorer.CandidateScorerMeanRank):        
            # mean reciprocal rank
            mrr_score = scorer.score(all_candidates, gold_topo_ids)
            error_data.append(mrr_score)
        elif(scorer.__class__ is CandidateScorer.CandidateScorerMedianDist):
            median_dist = scorer.score(all_candidates, gold_topo_lat_lons, k=1)
            error_data.append(median_dist)
    # keep track of error values
#     error_data = pd.Series([feature_name] + dist_k_scores + rank_k_scores + [mrr_score])
    
    # track errors on top candidates for later inspection
    top_candidate_ids = pd.np.array([t.get_id for t in top_candidates])
    top_candidate_scores = pd.np.array([t.get_attr(feature_name) for t in top_candidates])
    candidate_lat_lons = map(lambda x: [x.get_lat(), x.get_lon()], top_candidates)
    top_candidate_errors = np.array(map(lambda x: great_circle_error(x[0], x[1]), izip(candidate_lat_lons, gold_topo_lat_lons)))
    N = len(gold_topo_ids)
    feat_name_vector = np.repeat(feature_name, N)
    top_candidate_score_data = pd.np.vstack([top_candidate_ids, gold_topo_ids, top_candidate_scores, top_candidate_errors, feat_name_vector]).transpose()
    # for all candidate sets (at k=10) that don't have the gold id:
    # track each candidate, its associated query and the score
    error_k_rank = k_ranks[0]
    gold_topo_missing_data = pd.DataFrame()
    all_queries = query_data.loc[:, 'text']
    for q_query, q_top_candidates, gold_topo_id in izip(all_queries, all_candidates, gold_topo_ids):
        if(gold_topo_id not in q_top_candidates):
            q_top_candidate_ids = pd.np.array([t.get_id() for t in q_top_candidates])
            q_top_candidates_scores = pd.np.array([t.get_attr(feature_name) for t in q_top_candidates])
            q_missing_data = pd.DataFrame(pd.np.vstack([q_top_candidate_ids, q_top_candidates_scores, pd.np.repeat(gold_topo_id, len(q_top_candidates)), pd.np.repeat(q_query, len(q_top_candidates))]).transpose())
            gold_topo_missing_data = gold_topo_missing_data.append(q_missing_data)
    gold_topo_missing_data.columns = ['candidate', 'score', 'gold_topo', 'query']
    
    return error_data, top_candidate_score_data, gold_topo_missing_data

def generate_candidates_simple(topo_data, gazetteer_data, topo_name_col='entity_string', gazetteer_name_col='name_regex', topo_doc_col='docid', verbose=False, meta_cols=['context']):
    """
    For each toponym, look up all possible 
    candidates in the gazetteer as well as 
    associated metadata.

    :param topo_data: Toponym data frame
    :param gazetteer_data: Gazetteer data
    :param topo_name_col: Toponym name column
    :param gazetteer_name_col: Gazetteer name column (containing regex)
    :param topo_doc_col: Toponym document column
    :param verbose: debugging
    :param meta_cols: metadata columns to include from processed data
    :return topo_candidate_data:: Candidate data with 1 candidate per row.
    """
    unique_topo_names = topo_data.loc[:, topo_name_col].apply(query_norm).unique()
    print('processing %d unique toponyms'%(len(unique_topo_names)))
    topo_names_match_pairs = list(map(lambda x: (x, gazetteer_data[gazetteer_data.loc[:, gazetteer_name_col].apply(lambda y: y.match(x) is not None)]), unique_topo_names))
    topo_names_match_pairs = dict(topo_names_match_pairs)
    topo_names_match_pairs_with_candidates = {k : v for k,v in topo_names_match_pairs.items() if v.shape[0] > 0}
    topo_candidate_data = []
    date_cols = list(filter(lambda x: x.startswith('date'), topo_data.columns))
    lat_mu = topo_data.loc[:, 'lat'].mean()
    lon_mu = topo_data.loc[:, 'lon'].mean()
    lat_sd = topo_data.loc[:, 'lat'].std()
    lon_sd = topo_data.loc[:, 'lon'].std()
    topo_ctr = 0
    for (topo_name, topo_id), topo_data_i in topo_data.groupby([topo_name_col, topo_doc_col]):
        # lowercase and de-accent
#         topo_name_norm = normalize_str(topo_data_i.loc[:, topo_name_col].values[0])
        # de-camel case, lowercase and de-accent
        topo_name_norm = query_norm(topo_data_i.loc[:, topo_name_col].values[0])
#         print('topo name %s => "%s"'%(topo_name, topo_name_norm))
        if(topo_name_norm in topo_names_match_pairs_with_candidates):
            topo_names_match_pairs_i = topo_names_match_pairs[topo_name_norm].copy()
            topo_names_match_pairs_i.index = np.arange(topo_names_match_pairs_i.shape[0])
            topo_names_match_pairs_i = topo_names_match_pairs_i.assign(**{topo_name_col:topo_name})
            topo_names_match_pairs_i = topo_names_match_pairs_i.assign(**{'query':topo_name_norm})
            topo_names_match_pairs_i = topo_names_match_pairs_i.assign(**{topo_doc_col:topo_id})
            for date_col in date_cols:
                if(date_col == 'date_periodic'):
                    N_i = topo_names_match_pairs_i.shape[0]
                    topo_data_date_periodic_i = np.repeat(topo_data_i.loc[:, 'date_periodic'].iloc[0].reshape(1,-1), N_i, axis=0)
#                     topo_names_match_pairs_i = topo_names_match_pairs_i.assign(**{'date_periodic':topo_data_date_periodic_i})
                    topo_names_match_pairs_i['date_periodic'] = topo_data_date_periodic_i.tolist()
                else:
                    topo_names_match_pairs_i = topo_names_match_pairs_i.assign(**{date_col:topo_data.loc[:, date_col].iloc[0]})
            # gold data
            gold_coords_i = topo_data_i.loc[:, ['lat', 'lon']].values[0]
            topo_names_match_pairs_i = topo_names_match_pairs_i.assign(gold_lat=gold_coords_i[0], gold_lon=gold_coords_i[1])
            gold_error_i = topo_names_match_pairs_i.loc[:, ['latitude', 'longitude']].apply(lambda x: great_circle(x.values, gold_coords_i).kilometers, axis=1)
            # compute error as great circle distance
            topo_names_match_pairs_i = topo_names_match_pairs_i.assign(error=gold_error_i)
            # assign gold based on nearest candidate
            gold_label_i = topo_names_match_pairs_i.loc[:, 'error'].apply(lambda x: int(x==topo_names_match_pairs_i.loc[:, 'error'].min()))
            topo_names_match_pairs_i = topo_names_match_pairs_i.assign(gold=gold_label_i)
            # in case of duplicate gold locations: keep the one with higher population
            topo_names_match_pairs_i_gold = topo_names_match_pairs_i[topo_names_match_pairs_i.loc[:, 'gold']==1]
            topo_names_match_pairs_i_non_gold = topo_names_match_pairs_i[topo_names_match_pairs_i.loc[:, 'gold']==0]
            if(topo_names_match_pairs_i_gold.shape[0] > 1):
                topo_names_match_pairs_i_gold = topo_names_match_pairs_i_gold.sort_values('population', inplace=False, ascending=False).iloc[[0], :]
            topo_names_match_pairs_i = pd.concat([topo_names_match_pairs_i_gold, topo_names_match_pairs_i_non_gold], axis=0)
            
            # get normed coords
            lat_norm_i = (topo_names_match_pairs_i.loc[:, 'latitude'] - lat_mu) / lat_sd
            lon_norm_i = (topo_names_match_pairs_i.loc[:, 'longitude'] - lon_mu) / lon_sd
            topo_names_match_pairs_i = topo_names_match_pairs_i.assign(latitude_norm=lat_norm_i, longitude_norm=lon_norm_i)
            # add meta cols too
            if(len(meta_cols) > 0):
                for c in meta_cols:
                    c_data = topo_data_i.loc[:, c].values[0]
                    # to set list for a whole column, need to duplicate value and concat
                    if(type(c_data) is list or type(c_data) is np.array):
                        N_i = topo_names_match_pairs_i.shape[0]
                        c_data = pd.DataFrame(np.repeat(np.array(c_data).reshape(1,-1), N_i, axis=0))
#                         print('c_data shape %s'%(str(c_data.shape)))
#                         print('topo data shape %s'%(str(topo_names_match_pairs_i.shape)))
                        tmp_data = pd.concat([topo_names_match_pairs_i, c_data], axis=1).fillna('')
                        tmp_data.index = np.arange(tmp_data.shape[0])
                        tmp_cols = list(range(c_data.shape[1]))
#                         print('tmp cols %s'%(str(tmp_cols)))
#                         print('tmp data')
#                         print(tmp_data.loc[:, tmp_cols])
#                         print('tupled data')
#                         print(tmp_data.loc[:, tmp_cols].apply(lambda x: tuple(x), axis=1))
                        tmp_data.loc[:, c] = pd.Series([x.values for i,x in tmp_data.loc[:, tmp_cols].iterrows()])
#                         tmp_data.to_csv('tmp_data.tsv', sep='\t', index=False)
#                         tmp_data.loc[:, c] = tmp_data.loc[:, tmp_cols].apply(lambda x: list(x), axis=1)
#                         print('context data %s'%(tmp_data.loc[:, c]))
                        tmp_data.drop(tmp_cols, axis=1, inplace=True)
                        topo_names_match_pairs_i = tmp_data.copy()
                    else:
                        topo_names_match_pairs_i.loc[:, c] = c_data
#                 topo_names_match_pairs_i = topo_names_match_pairs_i.assign(**{c : topo_data_i.loc[:, c] for c in meta_cols})
            topo_candidate_data.append(topo_names_match_pairs_i)
        topo_ctr += 1
        if(verbose and topo_ctr % 100 == 0):
            print('processed %d unique topos = %d rows'%(topo_ctr, len(topo_candidate_data)))
#             print(topo_candidate_data[0])
    topo_candidate_data = pd.concat(topo_candidate_data, axis=0)
    # drop extra cols
    extra_geo_cols = ['name_regex', 'feature_class', 'feature_code', 'country']
    for c in extra_geo_cols:
        topo_candidate_data.drop(c, axis=1, inplace=True)
    return topo_candidate_data

def load_elmo(weight_file='/hg190/corpora/ELMO/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5', options_file='/hg190/corpora/ELMO/elmo_2x1024_128_2048cnn_1xhighway_options.json'):
    """
    Load ELMO model from file.
    
    :param weight_file: weight file
    :param options_file: options file
    :return model: ELMO model
    """
    elmo_model = Elmo(options_file, weight_file, 1, dropout=0, requires_grad=False)
    return elmo_model

def compute_elmo_embed_all_data(data, elmo_model, doc_col='docid_vol_context', method="avg"):
    """
    Compute ELMO embedding over all data.
    
    :param data: toponym DataFrame
    :param elmo_model: pre-loaded ELMO model
    :param doc_col: document ID column
    :returns data: updated toponym DataFrame
    """
    # only compute ELMO over unique contexts!! don't want to duplicate effort ;_;
    data_unique_contexts = data.loc[:, ['context', doc_col, 'entity_start_context', 'entity_string', 'entity_end_context']]
    data_unique_contexts.drop_duplicates(doc_col, inplace=True)
    data_unique_contexts.index = np.arange(len(data_unique_contexts))
    data_unique_contexts_elmo = pd.Series([compute_elmo_embed(x, elmo_model, method=method) for _,x in data_unique_contexts.iterrows()])
    data_unique_contexts_elmo.index = data_unique_contexts.index
    data_unique_contexts.loc[:, 'context_ELMO'] = data_unique_contexts_elmo
    # rejoin
    data = pd.merge(data, data_unique_contexts.loc[:, ['context_ELMO', doc_col]], on=doc_col, how='inner')
# convert to char embeddings
#     elmo_embeds = data.apply(lambda x: compute_elmo_embed(x, elmo_model), axis=1)
#     data.loc[:, 'context_ELMO'] = elmo_embeds
    return data

def compute_elmo_embed(x, elmo, x_combined_context=None, method="avg", micro_win=3):
    """
    Compute entity context from ELMO embedding.
    
    :param x: Row of data
    :param elmo: ELMO model
    :param x_combined_context: Combined string context
    :returns x_entity_embed:: entity context embedding
    """
    if(x_combined_context is None):
        x_combined_context = x.loc['entity_start_context'] + [x.loc['entity_string']] + x.loc['entity_end_context']
    x_embed = elmo(batch_to_ids([x_combined_context]))['elmo_representations'][0][0]
    x_entity_start_idx = len(x.loc['entity_start_context'])
    x_entity_end_idx = len(x.loc['entity_start_context']) + len(x.loc['entity_string'].split(' '))
    if method == "avg": # this is wrong
        x_entity_embed = x_embed[max(x_entity_start_idx - micro_win, 0):min(x_entity_start_idx + micro_win, len(x_combined_context)), :].mean(dim=0).detach().numpy()
    elif method == "max": # also wrong
        x_entity_embed = x_embed[max(x_entity_start_idx - micro_win, 0):min(x_entity_start_idx + micro_win, len(x_combined_context)), :].max(dim=0)[0].detach().numpy()
    elif method == "correct":
        x_entity_embed = x_embed[x_entity_start_idx].detach().numpy()
        
#     if(x_combined_context is None):
#         x_combined_context = x.loc['entity_start_context'] + x.loc['entity_string'].split(' ') + x.loc['entity_end_context']
#     x_embed = elmo(batch_to_ids([x_combined_context]))['elmo_representations'][0][0]
#     x_entity_start_idx = len(x.loc['entity_start_context'])
#     x_entity_end_idx = len(x.loc['entity_start_context']) + len(x.loc['entity_string'].split(' '))
#     if method == "avg": # this is wrong
#         x_entity_embed = x_embed[x_entity_start_idx:x_entity_end_idx, :].mean(dim=0).detach().numpy()
#     elif method == "max": # also wrong
#         x_entity_embed = x_embed[x_entity_start_idx:x_entity_end_idx, :].max(dim=0)[0].detach().numpy()
#     elif method == "correct":
#         x_entity_embed = x_embed[x_entity_start_idx].detach().numpy()
        
#     elif method == "concat":
#         try:
#             x_entity_embed = torch.cat([x_embed[x_entity_start_idx], x_embed[x_entity_end_idx - 1]]).detach().numpy()
#         except Exception as e:
#             print(x.loc['entity_string'])
#             print(x_combined_context)
#             print(x_embed[x_entity_start_idx])
#             print(x_embed[x_entity_start_idx].shape)
#             print(x_embed[x_entity_end_idx - 1])
#             print(x_embed[x_entity_end_idx - 1].shape)
#             print(e)
    return x_entity_embed

## train/test code

def train_and_test(train_data, dev_data, data_name, learning_rate, kl, num_epochs, h_size, z_size, enc_dec_depth, out_dir='vae_lab/checkpoints/', doc_col='docid_vol_context', num_processes=10, x_cols=['latitude_norm', 'longitude_norm'], cond_features=['context_ELMO'], gold_col='gold_pop', cuda=False, logger=None):
    """
    Train a ConditionalVAE with the given hyperparameters
    on the train data and compute accuracy on the
    dev data. 
    
    :param train_data: training data
    :param dev_data: dev data
    :param data_name: data name
    :param learning_rate: learning rate
    :param kl: KL factor weight
    :param ep: epochs
    :param h_size: encoder/decoder layer size
    :param z_size: hidden layer size
    :param enc_dec_depth: encoder/decoder depth
    :param cuda: use GPU for training boost (doesn't work?)
    :returns model_stats: model validation metrics, in DataFrame
    """
    gradient_minibatch_size = 32
    warmup_pct = 0.25
    N_warmup = int(warmup_pct * num_epochs)
    result_filename='hyperparam_search_' + data_name + '.txt'
    cond_size = np.hstack(train_data.loc[:, cond_features].iloc[0]).shape[0]
    model = ConditionalVAE(in_size=2, cond_size=cond_size, h_size=h_size, z_size=z_size, enc_depth=enc_dec_depth, dec_depth=enc_dec_depth, kld_factor=kl, batch_norm=False, cuda=cuda)
    checkpoint_dir = os.path.join(out_dir, data_name + '_EM_checkpoints')
    ## train model
    loss, ll = train(model, train_data, num_epochs,
                     x_cols=x_cols, cond_features=cond_features,
                     doc_col=doc_col, checkpointdir=checkpoint_dir, num_processes=num_processes,
                     N_warmup=N_warmup, learning_rate=learning_rate, 
                     calc_likelihood=False, m_step_epochs=1, 
                     gradient_minibatch_size=gradient_minibatch_size, 
                     logger=logger)
    model_file_final = os.path.join(checkpoint_dir, 'best.checkpoint.model')
    model_final = torch.load(model_file_final)
    
    model_final.eval()
    
    ## E-step on dev set
    pool = Pool(num_processes)
    chunk_size = int(ceil(dev_data.shape[0] / num_processes))
    dev_data_split = [dev_data.iloc[(i*chunk_size):((i+1)*chunk_size)] for i in range(num_processes)]
    # test
    test_scores = assign_scores(dev_data, model_final, 'COND_VAE', cond_features, x_cols)
    results = pool.starmap(assign_scores, zip(dev_data_split, [model_final,]*num_processes, ['COND_VAE',]*num_processes, [cond_features,]*num_processes, [x_cols,]*num_processes))
    pool.close()
    pool.terminate()
    dev_data = pd.concat(results, axis=0)
    
    ## compute internal loss on dev data
    uniform_prior_test = False # use weighted loss
    test_loss = 0
    x_test = [y.loc[:, x_cols].values for x,y in dev_data.groupby(doc_col)]
    cond_test = [np.vstack(np.hstack(y.loc[:, cond_features].values)) for x,y in dev_data.groupby(doc_col)]
    if uniform_prior_test:
        NULL_PROB = -1.
        prob_test = [pd.Series([NULL_PROB,]*len(x_test[i])) for i in range(len(x_test))]
    else:
        prob_test = [y.loc[:, 'COND_VAE'].values for x,y in dev_data.groupby(doc_col)]
    N = len(x_test)
    N_rep = 5
    for i in range(N):
        loss_i = []
        x_coords_i = x_test[i]
        prob_test_i = prob_test[i] - logsumexp(prob_test[i]) # normalize
#         print('prob train \n%s'%(prob_train_i))
        for j in range(len(x_coords_i)):
            x_coords_i_j = x_coords_i[j]
            if(cond_test is not None):
                cond_test_i_j = cond_test[i][j]
            else:
                cond_test_i_j = None
            prob_test_i_j = prob_test_i[j]
            loss_j = loss_candidates(x_coords_i_j, model_final, cond_data=cond_test_i_j, N_rep=N_rep, kld_factor=kl)
            tensor_prob_test_i_j = Tensor([np.exp(prob_test_i_j)])
            if model_final.cuda:
                tensor_prob_test_i_j = tensor_prob_test_i_j.cuda()
            loss_j_prob = loss_j.mul(tensor_prob_test_i_j)
            loss_i.append(loss_j_prob)
        loss_i = sum(loss_i)
        if(np.isnan(loss_i.detach().numpy())):
            print('TESTING: nan loss at group %d'%(i))
        test_loss += loss_i.data[0]
    
    # compute resolution score
    sort_col = 'COND_VAE'
    acc_scores, rec_scores, med_dist, err_auc = score_candidates(dev_data, sort_col=sort_col, query_col='entity_string', doc_col=doc_col, gold_col=gold_col)
    
    model_stats = pd.concat([pd.Series([test_loss.numpy()], index=['dev_loss']), 
                             acc_scores, rec_scores, 
                             pd.Series([med_dist, err_auc], index=['MedDist', 'AUC'])])
    return model_stats

# EARTH_CIRCUMFERENCE=24901.461 # miles
EARTH_CIRCUMFERENCE=40075.017 # kilometers
EARTH_RADIUS=EARTH_CIRCUMFERENCE/(np.pi*2)
MAX_ERROR=EARTH_CIRCUMFERENCE/2

def haversine(x):
    """
    Calculate the Haversine distance between two points 
    on the earth
    
    :param x: lat/lon Tensor => lat1,lon1,lat2,lon2 (n x 4 dimensions)
    :returns dist: dist Tensor (n x 1 dimensions)
    """
    # convert decimal degrees to radians 
#     lat1, lon1, lat2, lon2 = x
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    x = x * np.pi / 180.
    
    # haversine formula 
    dlon = x[:, 3] - x[:, 1]
    dlat = x[:, 2] - x[:, 0]
    a = torch.sin((dlat/2))**2 + torch.cos(x[:, 2]) * torch.cos(x[:, 0]) * torch.sin((dlon/2))**2
    c = 2 * torch.asin((a**.5))
#     dlon = lon2 - lon1 
#     dlat = lat2 - lat1 
#     a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    
#     c = 2 * asin(sqrt(a))
#     r = 6378 # Radius of earth in kilometers. Use 3956 for miles
    dist = c * EARTH_RADIUS
    return dist

def score_candidates(candidate_data, sort_col='population', error_col='error', query_col='query_name', doc_col='wotr_idx', gold_col='gold'):
    """
    Compute error metrics:
    1. accuracy@k miles (10,20,50,100,161)
    2. recall@k ranks (1,5,10,20,50)
    3. median distance [0,inf]
    4. AUC [0,1] => lower is better
    
    :param candidate_data: Candidate DataFrame
    :param sort_col: Column by which to sort for relevance
    :param error_col: Column containing error
    :param query_col: Column containing toponym query name
    :param doc_col: Column containing document of toponym
    :param gold_col: Column containing gold label
    :returns accuracy_at_k:: Accuracy metrics
    :return recall_at_k:: Recall metrics
    :return median_distance:: Median distance of top-ranked toponym
    """
    accuracy_at_k = [10, 20, 50, 100, 161]
    recall_at_k = [1, 5, 10, 20, 50]
    accuracy_at_k_vals = []
    recall_at_k_vals = []
    error_vals = []
    N = 0
    for g_name, g_group in candidate_data.groupby([query_col, doc_col]):
#         print(type(g_group.loc[:, error_col].iloc[0]))
        if(g_group.shape[0] > 1 and type(g_group.loc[:, error_col].iloc[0]) is not str):
            if(g_group.loc[:, gold_col].sum() > 1):
                print('too many gold candidates in group %s,%s'%(g_name))
                print(g_group[g_group.loc[:, gold_col]==1])
            else:
                N += 1
                g_group.sort_values(sort_col, inplace=True, ascending=False)
                a_k_g = []
                r_k_g = []
                for a_k in accuracy_at_k:
                    a_k_val = int(g_group.iloc[0, :].loc[error_col] <= a_k)
                    a_k_g.append(a_k_val)
                        # some candidates have error = "" ???
#                     except Exception as e:
#                         print('error %s'%(e))
#                         print('error dist data %s'%(g_group.loc[:, ['query', 'name', error_col]]))
                for r_k in recall_at_k:
    #                 else:
                    r_k_val = g_group.iloc[:r_k, :].loc[:, gold_col].sum()
                    r_k_g.append(r_k_val)
                top_err = g_group.iloc[0, :].loc['error']
                accuracy_at_k_vals.append(a_k_g)
                recall_at_k_vals.append(r_k_g)
                error_vals.append(top_err)
            if(N % 10 == 0):
                print('processed %d candidate groups'%(N))
    # compute
    accuracy_at_k_vals = np.array(accuracy_at_k_vals)
    recall_at_k_vals = np.array(recall_at_k_vals)
    accuracy_at_k_vals = accuracy_at_k_vals.mean(axis=0)
    recall_at_k_vals = recall_at_k_vals.mean(axis=0)
    # convert to series
    accuracy_at_k_vals = pd.Series(accuracy_at_k_vals, index=['A@%d'%(x) for x in accuracy_at_k])
    recall_at_k_vals = pd.Series(recall_at_k_vals, index=['R@%d'%(x) for x in recall_at_k])
    # remove nan errors
    error_vals = [x for x in error_vals if not np.isnan(x)]
    median_dist = np.median(error_vals)
    # compute auc (code from https://github.com/milangritta/Geocoding-with-Map-Vector/blob/master/preprocessing.py)
    error_log = np.log(np.array(error_vals) + 1)
    error_auc = np.trapz(error_log) / (np.log(MAX_ERROR) * (len(error_vals) - 1))
    return accuracy_at_k_vals, recall_at_k_vals, median_dist, error_auc

def get_cond_data(data, cond_features):
    cond_data = []
    for cond_feature in cond_features:
        if(cond_feature == 'date_bin'):
            cond = pd.get_dummies(data.loc[:, cond_feature]).values
        else:
            cond = np.array(data.loc[:, cond_feature].values)
        cond_data.append(cond)
    if(len(cond_data) > 1):
#         print('combining cond data %s'%(str(cond_data[0].shape)))
        cond_data = np.vstack(cond_data).transpose()
#         print('combined cond data %s'%(str(cond_data.shape)))
    else:
        cond_data = cond_data[0]
    if(len(cond_data.shape) == 1):
        cond_data = np.vstack([x for x in cond_data])
    return cond_data

def assign_scores(candidate_data, model, model_type, cond_features=None, x_cols=['lat_norm', 'lon_norm'], debug=False):
    """
    Assign probability scores to all candidates using model and (optional) conditional variables.
    
    :param candidate_data: Candidate toponym data
    :param model: Probability model
    :param model_type: Model type
    :param cond_features: if conditional model, the conditions for assigning probability
    :param x_cols: Columns in candidate data containing x data
    :param debug: print debug statements
    :return candidate_data_scores:: Update candidate toponym data
    """
    candidate_data_scores = candidate_data.copy()
    coords_tensor = Tensor(candidate_data.loc[:, x_cols].values)
    if(debug):
        print('coords %s'%(coords_tensor))
    if model.cuda:
        coords_tensor = coords_tensor.cuda()
    N_samp = 10
    N_rep = 5
    if(model_type == 'VAE'):
        from vae_lab.gpsvae import approx_log_px
        coords_recon, _, _ = model(Variable(coords_tensor))
        var_px_z = Variable((coords_recon.data.sub(coords_tensor)).pow(2).mean(dim=0))
        candidate_data_scores.loc[:, model_type] = candidate_data_scores.loc[:, x_cols].apply(lambda x: approx_log_px(Variable(Tensor(np.repeat(x.values.reshape(1,-1), N_rep, axis=0))), model, var_px_z, N_samp=N_samp).mean(), axis=1)
    elif(model_type == 'COND_VAE'):
        from vae_lab.gpsvae_conditional import approx_log_px
        if('context' not in cond_features):
            cond_data = Tensor(get_cond_data(candidate_data, cond_features))
        else:
            cond_data = [x.values for _, x in candidate_data.loc[:, cond_features].iterrows()]
        if model.cuda:
            cond_data = cond_data.cuda()
        coords_recon, _, _ = model(Variable(coords_tensor), cond_data)
        var_px_z = Variable((coords_recon.data.sub(coords_tensor)).pow(2).mean(dim=0))
#         if(debug):
#             x = candidate_data_scores.iloc[0, :]
#             p_x = approx_log_px(Tensor(np.repeat(x.loc[x_cols].values.reshape(1,-1).astype(float), N_rep, axis=0)), Tensor(np.repeat(get_cond_data(pd.DataFrame(x).transpose(), cond_features), N_rep, axis=0)), model, var_px_z, N_samp=N_samp, debug=True)[0]
#             print('p(x) %s'%(p_x))
        candidate_data_scores.loc[:, model_type] = candidate_data_scores.apply(lambda x: approx_log_px(Tensor(np.repeat(x.loc[x_cols].values.reshape(1,-1).astype(float), N_rep, axis=0)), Tensor(np.repeat(get_cond_data(pd.DataFrame(x).transpose(), cond_features), N_rep, axis=0)), model, var_px_z, N_samp=N_samp)[0], axis=1)
        if(debug):
            print(candidate_data_scores.loc[:, model_type].head())
    return candidate_data_scores

def assign_joint_scores(data, model_types, doc_col, entity_col='entity_string', model_weights=None, log_transform=False):
    """
    Assign joint score for data candidates as
    product of normalized probabilities across model types.

    :param data: candidate data
    :param model_types: list of model types with probabilities/scores computed
    :param doc_col: column with document ID
    :param entity_col: column with entity string
    :param model_weights: optional weights for different models, w \in (0,1)
    :returns score_data:: scored data
    """
    score_data = []
    joint_model_name = '+'.join(model_types)
    for (entity_i, doc_i), data_i in data.groupby([entity_col, doc_col]):
        # normalize => sum to 1
        for model_type in model_types:
            # normalize count variables
            if(log_transform and (model_type == 'population' or model_type == 'alternate_name_count')):
                data_i.loc[:, model_type] += 1.
                data_i.loc[:, model_type] = np.log(data_i.loc[:, model_type] / data_i.loc[:, model_type].sum())
            data_i.loc[:, model_type] = data_i.loc[:, model_type] - logsumexp(data_i.loc[:, model_type])
        # get joint => add probabilities
        # optional model weights for different probability mass
        if(model_weights is not None):
            data_i.loc[:, joint_model_name] = np.log(np.exp(data_i.loc[:, model_types]) * model_weights).sum(axis=1)
        else:
            data_i.loc[:, joint_model_name] = data_i.loc[:, model_types].sum(axis=1)
        score_data.append(data_i)
    score_data = pd.concat(score_data, axis=0)
    return score_data

# TODO error analysis
# def view_predictor_errors()

def compute_P_prob(lat, lon, P, N_samp=10):
    """
    Compute probability of a single lat/lon point.
    
    Parameters:
    -----------
    lat : float
    lon : float
    P : VAE
    N_samp : int
    
    Returns:
    --------
    prob : float
    """
    point = torch.Tensor([[lat, lon]])
    point_var = torch.autograd.Variable(point)
    point_recon, _, _ = P(point_var)
    prob_var = torch.autograd.Variable((point_recon.data.sub(point)).pow(2).mean(dim=0))
    prob = approx_log_px(point_var, P, prob_var, N_samp=N_samp)
    return prob

def plot_approx_prob_map(x_bounds, y_bounds, coords, vae, geo_map, depth=40, log_scale=False, scale_power=1, plot_type='contour'):
    mapper = cm.Blues
    all_logp = []
    x = pd.np.linspace(x_bounds[0], x_bounds[1], depth)
    y = pd.np.linspace(y_bounds[0], y_bounds[1], depth)
    xx, yy = pd.np.meshgrid(x, y)
    
    coords_tensor = torch.Tensor(coords)
    coords_recon, _, _ = vae(torch.autograd.Variable(coords_tensor))
    var_px_z = torch.autograd.Variable((coords_recon.data.sub(coords_tensor)).pow(2).mean(dim=0))
    prob = []
    for x_i in x:
        x_coords = np.tile(x_i,depth)
        points = torch.Tensor(np.vstack([y, x_coords]).T)
        log_px = approx_log_px(torch.autograd.Variable(points),
                               vae,
                               var_px_z,
                               N_samp=10) 
        # log-scale as needed
        if(log_scale):
            log_px = -1 * np.log10(-1*log_px)
        # power as needed
        log_px = log_px ** scale_power
        prob += [list(log_px)]
    prob = pd.np.array(prob)
    geo_map.drawcoastlines()
    if(plot_type=='contour'):
        geo_map.contourf(xx, yy, prob, latlon=True, cmap=mapper)
    elif(plot_type=='hexbin'):
        hex_size = 25
        xx_flat = pd.np.hstack(xx)
        yy_flat = pd.np.hstack(yy)
        # correct xx, yy to new scale
        xx_flat, yy_flat = geo_map(xx_flat, yy_flat)
        prob_flat = pd.np.hstack(prob)
        geo_map.hexbin(xx_flat, yy_flat, C=prob_flat, cmap=mapper, gridsize=hex_size, reduce_C_function=pd.np.mean, mincnt=None)
    cb = geo_map.colorbar(location='bottom')
    cb.ax.tick_params(labelsize=18)
    cb.set_label(label='P(Y)', fontsize=24)

## compute and plot prob distribution

def compute_prob_dist(model, x_bounds, y_bounds,
                      coord_mean, coord_sd,
                      var_px_z,
                      data=None,
                      grid_size=1., cond_col='context_ELMO', N_samp=10,
                      debug=True):
    """
    Compute conditional probability distribution over geographic grid.
    
    :param model: CVAE model
    :param x_bounds: longitude bounds
    :param y_bounds: latitude bounds
    :param coord_mean: mean of coordinates (for scaling)
    :param coord_sd: standard deviation of coordinates (for scaling)
    :param var_px_z: variance P(X | Z)
    :param grid_size: coordinate grid size
    :param data: data for a single candidate (Series) => need for conditional model
    :param cond_col: conditional variable columns
    :param N_samp: number of samples to draw for probability calculation
    :param debug: print debug statements
    :returns prob_data:: 2-D matrix of coordinate-binned probability
    """
    from vae_lab.gpsvae_conditional import approx_log_px
    if(cond_col is not None):
        cond_size = len(data.loc[cond_col])
        data_cond = Tensor(np.reshape(data.loc[cond_col], (1, cond_size)))
    x_bin_count = int((x_bounds[1] - x_bounds[0]+1) / grid_size)
    y_bin_count = int((y_bounds[1] - y_bounds[0]+1) / grid_size)
    if(debug):
        print('%d x bins, %d y bins'%(x_bin_count, y_bin_count))
    x_bins = np.linspace(x_bounds[0], x_bounds[1], x_bin_count)
    y_bins = np.linspace(y_bounds[0], y_bounds[1], y_bin_count)
    N = x_bin_count * y_bin_count
    x_flat = np.reshape(np.repeat(x_bins, y_bin_count), (N, ))
    y_flat = np.reshape(np.repeat(np.reshape(y_bins, (1, y_bin_count)), x_bin_count, axis=0), (N, ))
#     data_context = data.loc['context']
#     if(debug):
#         print('processing context %s'%(' '.join(data_context)))
    prob_data = np.zeros((x_bin_count, y_bin_count))
    ## compute mean probability for each bin
    ctr = 0
    for i, x_bin in enumerate(x_bins):
        for j, y_bin in enumerate(y_bins):
            xy_coord = Tensor(np.reshape((np.array([y_bin, x_bin]) - coord_mean) / coord_sd, (1, 2)))
            # compute V(P_X_Z)??
            if(cond_col is not None):
                from vae_lab.gpsvae_conditional import approx_log_px
#                 xy_hat = model(xy_coord, data_cond)
#                 var_px_z = Variable(Tensor(((X_hat - X)**2).mean(axis=0)))
                xy_prob = approx_log_px(xy_coord, data_cond, model, var_px_z, N_samp=N_samp)
            else:
                from vae_lab.gpsvae import approx_log_px
#                 xy_hat = model(xy_coord)
#                 var_px_z = Variable(Tensor(((X_hat - X)**2).mean(axis=0)))
                xy_prob = approx_log_px(xy_coord, model, var_px_z, N_samp=N_samp)
#             if(np.isnan(xy_prob)):
#                 print('nan prob at %.3f,%.3f'%(x_bin, y_bin))
            prob_data[i, j] = xy_prob
            ctr += 1
            if(debug and ctr % 100 == 0):
                print('computed %d/%d probs'%(ctr, N))
    # remove nan values!!
    prob_data[np.isnan(prob_data)] = prob_data[~np.isnan(prob_data)].min()
    # normalize
    prob_data = prob_data - logsumexp(prob_data)
    if(debug):
        print('probs\n%s'%(prob_data))
    return prob_data

def plot_prob_dist(prob_data, x_bounds, y_bounds, 
                   cmap=cm.Blues, grid_size=20, 
                   prob_cutoff_pct=None, geo_map=None,
                   vmin=None, vmax=None):
    """
    Plot probability distribution on a map.
    
    :param prob_data: 2-D matrix of coordinate-binned probability
    :param x_bounds: longitude boundaries
    :param y_bounds: latitude boundaries
    :param cmap: color map
    :param grid_size: grid size for hex bins (bigger value => smaller bins)
    :param prob_cutoff_pct: optional cutoff percentile (round all prob X to max(X, pct))
    :param vmin: min prob val to include in color bar
    :param vmax: max prob val to include in color bar
    :param geo_map: optional Basemap
    """
    x_bin_count, y_bin_count = prob_data.shape
    N = x_bin_count * y_bin_count
    x_bins = np.linspace(x_bounds[0], x_bounds[1], x_bin_count)
    y_bins = np.linspace(y_bounds[0], y_bounds[1], y_bin_count)
    x_flat = np.reshape(np.repeat(x_bins, y_bin_count), (N, ))
    y_flat = np.reshape(np.repeat(np.reshape(y_bins, (1, y_bin_count)), x_bin_count, axis=0), (N, ))
    prob_data = np.reshape(prob_data, (N, ))
    if(prob_cutoff_pct is not None):
        prob_cutoff = np.percentile(prob_data, prob_cutoff_pct)
        prob_data[prob_data < prob_cutoff] = prob_cutoff
    
    if(geo_map is None):
        from mpl_toolkits.basemap import Basemap
        geo_map = Basemap(projection='merc', llcrnrlon=x_bounds[0], llcrnrlat=y_bounds[0], urcrnrlon=x_bounds[1], urcrnrlat=y_bounds[1])
    x_flat, y_flat = geo_map(x_flat, y_flat)
    geo_map.drawcoastlines()
    # scatterplot...looks bad man
    #         geo_map.scatter(x_flat, y_flat, c=prob_colors, s=10.)
    hex_plot = geo_map.hexbin(x_flat, y_flat, C=prob_data, cmap=cmap, gridsize=grid_size, vmin=vmin, vmax=vmax)
    geo_map.colorbar(hex_plot, location='right')

def get_z_sample(model, sample_size, cond_data=None):
    """
    Sample points from Z layer (hidden).
    
    :param model: VAE/CVAE from which to sample
    :param sample_size: sample size
    :param cond_data: conditional data to restrict samples
    :return x_hat: sampled X
    """
    z_samp = Variable(torch.randn(sample_size, model.z_size))
    # forward pass
    if(cond_data is not None):
        cond_data_fixed = Variable(Tensor(np.repeat(np.reshape(cond_data, (1, len(cond_data))), sample_size, axis=0)))
        z_samp = torch.cat([z_samp, cond_data_fixed], dim=1)
#     print(z_samp.shape)
    x_hat = model.decode(z_samp)
    x_hat = x_hat.data.numpy()
    return x_hat
    
def plot_samples(model, data, coord_mean, coord_sd, sample_size=1000, x_bounds=[-125, -65], y_bounds=[25, 50], cond_col='context_ELMO', geo_map=None):
    """
    Generate random samples from Z layer 
    of CVAE and plot on map.
    
    :param model: CVAE from which to sample
    :param data: DataFrame to use for context (each row = unique context to sample)
    :param coord_mean: mean coordinate for rescaling
    :param coord_sd: standard dev coordinate for rescaling
    :param sample_size: sample size
    :param x_bounds: longitude boundaries
    :param y_bounds: latitude boundaries
    :param cond_col: conditional variable column
    :param geo_map: Basemap to plot on
    """
    if(geo_map is None):
        geo_map = Basemap(llcrnrlon=x_bounds[0], llcrnrlat=y_bounds[0], 
                          urcrnrlon=x_bounds[1], urcrnrlat=y_bounds[1], 
                          projection='merc')
    for i, r in data.iterrows():
        if(cond_col is not None):
            cond_data_i = r.loc[cond_col]
            x_hat = get_z_sample(model, sample_size, cond_data_i)
        else:
            x_hat = get_z_sample(model, sample_size)
        # unnormalize
        x_hat = x_hat * np.repeat(coord_sd, sample_size, axis=0) + coord_mean
        ## show context!!
        r_context = ' '.join(r.loc['context'])
#         r_context = ' '.join(r.loc['entity_start_context'] + ['**%s**'%(r.loc['entity_string'])] + r.loc['entity_end_context'])
        print('idx = %d, context = %s'%(i, r_context))
        topo_i = r.loc['entity_string']
    
        ## plot samples
        x, y = geo_map(x_hat[:, 1], x_hat[:, 0])
        geo_map.drawcoastlines()
        geo_map.scatter(x, y, alpha=0.5, s=3., c='r')
        plt.title('Samples for "%s" mention'%(topo_i))
        plt.show()

def plot_recon_samples_prob(data, model, x_cols=['lat_norm', 'lon_norm'], coord_cols=['lat', 'lon'], cond_col=None, coord_mean=None, coord_sd=None, sample_size=1000, x_bounds=[-125, -65], y_bounds=[25, 50], prob_grid_size=1., hex_grid_size=15, prob_cutoff_pct=0.):
    """
    Plot (1) reconstructed data (2) sampled Z data (3) probability distribution
    for a given VAE/CVAE.
    
    :param data: candidate DataFrame
    :param model: VAE/CVAE
    :param x_cols: data columns
    :param coord_cols: coordinate columns
    :param cond_col: optional conditional column
    :param coord_mean: mean of coordinates
    :param coord_sd: standard deviation of coordinates
    :param sample_size: sample size
    :param x_bounds: longitude bounds
    :param y_bounds: latitude bounds
    :param prob_grid_size: probability map grid size
    :param hex_grid_size: probability hex grid size
    :param prob_cutoff_pct: lower percentile of probability at which to cut off 
    """
    if(coord_mean is None):
        coord_mean = np.reshape(data.loc[:, coord_cols].mean(axis=0).values, (1,2))
    if(coord_sd is None):
        coord_sd = np.reshape(data.loc[:, coord_cols].std(axis=0).values, (1,2))
    
    height = 4
    width = 10
    cols = 3
    rows = 1
    fig, axs = plt.subplots(rows, cols, figsize=(width*cols, height*rows))
    ## recon
    ax_recon = axs[0]
    geo_map = Basemap(projection='merc', llcrnrlon=x_bounds[0], llcrnrlat=y_bounds[0], urcrnrlon=x_bounds[1], urcrnrlat=y_bounds[1], ax=ax_recon)
    X = data.loc[:, x_cols].values
    X_gold = data.loc[:, coord_cols].values
    x_gold, y_gold = geo_map(X_gold[:, 1], X_gold[:, 0])
    if(cond_col is not None):
        X_cond = np.vstack(data.loc[:, cond_col].values)
        X_hat, _, _ = model(Tensor(X), Tensor(X_cond))
    else:
        X_hat, _, _ = model(Tensor(X))
    X_hat = X_hat.detach().numpy()
    # rescale
    X_hat = X_hat * np.repeat(coord_sd, X_hat.shape[0], axis=0) + coord_mean
    x_hat, y_hat = geo_map(X_hat[:, 1], X_hat[:, 0])
    ## debug: compute error
#     recon_err = X_gold - X_hat
#     recon_err_mean = recon_err.mean(axis=0)
#     recon_err_sd = recon_err.std(axis=0)
#     print('recon err = %s +/- %s'%(recon_err_mean, recon_err_sd))
    geo_map.drawcoastlines()
    geo_map.scatter(x_gold, y_gold, c='r', s=1., alpha=0.5, label='gold')
    geo_map.scatter(x_hat, y_hat, c='b', s=1., alpha=0.5, label='recon')
    ax_recon.legend(loc='lower right')
    ax_recon.set_title('recon')
    
    ## samples
    ax_samp = axs[1]
    geo_map = Basemap(projection='merc', llcrnrlon=x_bounds[0], llcrnrlat=y_bounds[0], urcrnrlon=x_bounds[1], urcrnrlat=y_bounds[1], ax=ax_samp)
    if(cond_col is not None):
        # use conditional data from first toponym sample...because it's simpler
        cond_data_i = data.loc[:, cond_col].iloc[0]
        X_samp = get_z_sample(model, sample_size, cond_data_i)
    else:
        X_samp = get_z_sample(model, sample_size)
    X_samp = X_samp * np.repeat(coord_sd, sample_size, axis=0) + coord_mean
    x_samp, y_samp = geo_map(X_samp[:, 1], X_samp[:, 0])
    geo_map.drawcoastlines()
    geo_map.scatter(x_samp, y_samp, c='r', s=1., alpha=0.5)
    ax_samp.set_title('samples')
    
    ## prob map
    ax_prob = axs[2]
    geo_map = Basemap(projection='merc', llcrnrlon=x_bounds[0], llcrnrlat=y_bounds[0], urcrnrlon=x_bounds[1], urcrnrlat=y_bounds[1], ax=ax_prob)
    # compute V(P_X_Z)
    var_px_z = Variable(Tensor(((X_hat - X)**2).mean(axis=0)))
#     var_px_z = None
    # compute dist
    N_samp = 10
    if(cond_col is not None):
        data_i = data.iloc[0, :]
    else:
        data_i = None
    prob_dist = compute_prob_dist(model, x_bounds, y_bounds,
                                  coord_mean, coord_sd, var_px_z, 
                                  N_samp=N_samp,
                                  grid_size=prob_grid_size, 
                                  data=data_i, cond_col=cond_col, 
                                  debug=False)
    plot_prob_dist(prob_dist, x_bounds, y_bounds, grid_size=hex_grid_size, prob_cutoff_pct=prob_cutoff_pct, geo_map=geo_map)
    ax_prob.set_title('prob')
        
def set_logger(log_file):
    """
    Set up basic file logger.
    
    :param log_file: log file name
    :returns logger:: logger
    """
    logger = logging.getLogger('basicLogger')
    log_handler = logging.FileHandler(log_file)
    log_formatter = logging.Formatter('%(asctime)s %(message)s')
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    return logger

def pandas_df_to_markdown_table(df):
    """
    Convert pandas.DataFrame to markdown table.
    
    :param df: DataFrame
    """
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    df_formatted = df_formatted.to_csv(sep="|", index=False)
    return(df_formatted)

def print_results_coeffs(results, coeffs, verbose=False):
    ## get mean/deviance
    float_cols = ['acc_mean', 'deviance']
    if(verbose):
        print('printing stats for %s'%(','.join(float_cols)))
    # format cols
    for x in float_cols:
        if(type(results.loc[:, x].iloc[0]) is not str):
            results.loc[:, x] = results.loc[:, x].apply(lambda x: '%.3f'%(x))
    results_str = pandas_df_to_markdown_table(results)
    print(results_str)
    ## get coefficients, p-vals for time and importance vars
    coeff_float_cols = ['coeff', 'p_val']
    if(verbose):
        print('printing stats for %s'%(','.join(coeff_float_cols)))
    # format cols
    for x in coeff_float_cols:
    #     if(type(joint_data_coeffs.loc[:, x].iloc[0]) is not str):
        coeffs.loc[:, x] = coeffs.loc[:, x].apply(lambda x: '%.3f'%(float(x)))
    coeffs_str = pandas_df_to_markdown_table(coeffs)
    print(coeffs_str)
    
def clean_var_name(x):
    return x.replace('-','_').replace('/', '_').replace("'", '_')

def fit_evaluate_lr_model(data, ind_vars, dep_var, test=0.1, k=10, balance=False):
    """
    Fit and evaluate LR model based on ability
    to predict dep_var. 
    We are interested in (1) predictive power and (2) deviance from null model.
    
    :param data: prediction data
    :param ind_vars: independent vars
    :param dep_var: dependent var
    :param test: test percent
    :param k: k_fold classification count
    :param balance: 
    """
    np.random.seed(123)
    formula = '%s ~ %s'%(dep_var, ' + '.join(ind_vars))
    print('formula: %s'%(formula))
#     print(data.loc[:, 'NE_fixed'].head())
    ## regular fit/statistics
    model = glm(formula=formula, data=data, family=Binomial())
    model_results = model.fit()
    print(model_results.summary())
    if(balance):
        data.loc[:, dep_var] = data.loc[:, dep_var].astype(int)
        dep_var_counts = data.loc[:, dep_var].value_counts()
        N_min_class = dep_var_counts.iloc[-1]
        data_balanced = pd.concat([data_c.loc[np.random.choice(data_c.index, N_min_class, replace=False), :] for c, data_c in data.groupby(dep_var)], axis=0)
        data = data_balanced.copy()
#     print(data.loc[:, 'NE_fixed'].head())
    
    ## k-fold cross validation
    # convert categorical vars to usable format
    reg_data = data.copy()
    cat_var_matcher = re.compile('C\((.+)\)')
    ind_vars_cat = [cat_var_matcher.search(x).group(1) for x in ind_vars if cat_var_matcher.search(x) is not None]
    if(len(ind_vars_cat) > 0):
        ind_var_cat_vals = []
    #     print(reg_data.loc[:, ind_vars_cat].head())
        for ind_var_cat in ind_vars_cat:
            ind_var_unique_vals = list(reg_data.loc[:, ind_var_cat].unique())
    #             print(unique_val)
            reg_data = reg_data.assign(**{clean_var_name(x):(reg_data.loc[:, ind_var_cat]==x).astype(int) for x in ind_var_unique_vals})
            # fix bad strings
            ind_var_unique_vals = [clean_var_name(x) for x in ind_var_unique_vals]
            ind_var_cat_vals += ind_var_unique_vals
            reg_data.drop(ind_var_cat, axis=1, inplace=True)
    #     print('data cols %s'%(str(reg_data.columns)))
        ind_vars_full = (set(ind_vars) - set(['C(%s)'%(x) for x in ind_vars_cat])) | set(ind_var_cat_vals)
        formula_full = '%s ~ %s'%(dep_var, ' + '.join(ind_vars_full))
    else:
        formula_full = '%s ~ %s'%(dep_var, ' + '.join(ind_vars))
#     print('formula full => %s'%(formula_full))
    kfold = KFold(n_splits=k, shuffle=True)
    predict_acc = []
    reg_data.loc[:, dep_var] = reg_data.loc[:, dep_var].astype(int)
    for train_idx, test_idx in kfold.split(reg_data):
        data_train = reg_data.iloc[train_idx, :]
        data_test = reg_data.iloc[test_idx, :]
#         print('train data %s'%(str(data_train.columns)))
        model_i = logit(formula=formula_full, data=data_train)
#         model_i = logit(endog=train_data.loc[:, dep_var], exog=train_data.loc[:, ind_vars])
        model_i_results = model_i.fit(full_output=False, disp=True)
        model_i_results.predict(data_test)
        pred_vals_i = np.array([int(x > 0.5) for x in model_i_results.predict(data_test)])
        y = data_test.loc[:, dep_var].astype(int)
#         predict_results_i = 1 - ((y - pred_vals_i) / len(y))
        predict_results_i = (y == pred_vals_i)
        predict_acc_i = np.mean(predict_results_i)
        predict_acc.append(predict_acc_i)
    return model_results, predict_acc

## regression helpers
def compute_err_data(model_results):
    """
    Compute error data for regularized regression.
    """
    exog_names = model_results.model.exog_names
    design_mat = model_results.model.exog
    pred_probs = model_results.model.predict(model_results.params)
    # need sparse matrix! to avoid memory explosion
    prob_mat = diags(pred_probs, 0).tocsr()
    design_mat = csr_matrix(design_mat)
    cov_mat = linalg.inv(design_mat.T.dot(prob_mat).dot(design_mat))
    param_err = np.sqrt(np.diag(cov_mat.todense()))
    model_err_data = pd.DataFrame(model_results.params, columns=['mean'])
    model_err_data = model_err_data.assign(**{'SE' : param_err})
    # compute test stat, p-val for two-sided test
    # https://stats.stackexchange.com/questions/60074/wald-test-for-logistic-regression
    model_err_data = model_err_data.assign(**{'z_score' : model_err_data.loc[:, 'mean'] / model_err_data.loc[:, 'SE']})
    # use Wald test
    model_err_data = model_err_data.assign(**{'p_val' : model_err_data.loc[:, 'z_score'].apply(lambda x: 1-chi.cdf(x**2, 1))})
    # model_err_data = model_err_data.assign(**{'p_val' : model_err_data.loc[:, 'z_score'].apply(lambda x: norm.cdf(x))})
    # confidence intervals
    alpha = 0.05
    Z_alpha = norm.ppf(1-alpha/2)
    model_err_data = model_err_data.assign(**{'conf_%.1f'%(alpha/2*100) : model_err_data.loc[:, 'mean'] - Z_alpha*model_err_data.loc[:, 'SE']})
    model_err_data = model_err_data.assign(**{'conf_%.1f'%((1-alpha/2)*100) : model_err_data.loc[:, 'mean'] + Z_alpha*model_err_data.loc[:, 'SE']})
    # model_err_data = model_err_data.assign(**{'p_val' : norm.cdf(model_err_data.loc[:, 't_score'])})
    return model_err_data

def compute_chi2_null_test(model_results, data, dep_var, max_iter, l2_weight):
    """
    Compute difference from null model using deviance:
    P(null) - P(model) ~ chi_2
    """
    null_formula = '%s ~ 1'%(dep_var)
    null_model = GLM.from_formula(null_formula, data, family=Binomial(link=logit()))
    null_model_results = null_model.fit_regularized(maxiter=max_iter, method='elastic_net', alpha=l2_weight, L1_wt=0.0)
    model_loglike = model_results.model.loglike(model_results.params)
    null_model_loglike = null_model_results.model.loglike(null_model_results.params)
    llr = -2*(null_model_loglike - model_loglike)
    model_df = model_results.model.df_model
    p_val = chi2.sf(llr, model_df)
    return llr, model_df, p_val