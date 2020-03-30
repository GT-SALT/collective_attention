"""
Run descriptor regression on cleaned data.
"""
import numpy as np
import pandas as pd
## need data
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import Binomial
from statsmodels.genmod.families.links import logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from model_helpers import compute_err_data
from argparse import ArgumentParser
import logging
import os
from sklearn.preprocessing import StandardScaler
from scipy.stats.distributions import chi2
import re
from ast import literal_eval
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

def compute_VIF(model_results, var_name):
    """
    Compute variance inflation factor for a given variable.
    https://stats.idre.ucla.edu/stata/webbooks/reg/chapter2/stata-webbooksregressionwith-statachapter-2-regression-diagnostics/
    """
    exog = model_results.model.exog
    var_idx = np.where(model_results.params.index == var_name)[0][0]
    VIF = variance_inflation_factor(exog, var_idx)
    return VIF

# formula stolen from here: https://www.statsmodels.org/stable/_modules/statsmodels/discrete/discrete_model.html#Logit
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

def k_fold_acc(X, Y, k=10, train_pct=0.9):
    """
    Compute k-fold accuracy
    with logistic regression.
    """
    # balance on dependent variable
    np.random.seed(123)
    class_counts = pd.Series(Y).value_counts()
    min_class = class_counts[class_counts == class_counts.min()].index[0]
    min_class_idx = list(np.where(Y == min_class)[0])
    max_class_idx = list(np.where(Y != min_class)[0])
#     print('N=%d'%(X.shape[0]))
#     print('max class = %s'%(max_class_idx))
    N_y = len(min_class_idx)
    N = len(X)
    N_train = int(train_pct * N_y)
    accs = []
    for i in range(k):
        np.random.shuffle(min_class_idx)
        max_class_idx_i = list(np.random.choice(max_class_idx, N_y, replace=False))
        train_idx = min_class_idx[:N_train] + max_class_idx_i[:N_train]
        test_idx = min_class_idx[N_train:] + max_class_idx_i[N_train:]
#         print('test idx = %s'%(test_idx))
        logit_model = LogisticRegression(penalty='l2', solver='lbfgs')
        X_train_i = X[train_idx, :]
        Y_train_i = Y[train_idx]
        X_test_i = X[test_idx, :]
        Y_test_i = Y[test_idx]
        # tmp debugging => baseline should have 50% accuracy
#         logging.info('on fold %d, train data has %d/%d min class'%(i, len(Y_train_i[Y_train_i==min_class]), len(Y_train_i)))
        logit_model_fit = logit_model.fit(X_train_i, Y_train_i)
        Y_pred_i = logit_model_fit.predict(X_test_i)
        N_test_i = len(test_idx)
        errs = abs(Y_pred_i - Y_test_i).sum()
        acc = (N_test_i - errs) / N_test_i
        accs.append(acc)
    # old k-fold code
#     np.random.shuffle(X)
#     kfolds = KFold(n_splits=k)
#     accs = []
#     for train_idx, test_idx in kfolds.split(X):
#     #     print(train_idx)
#     #     print(test_idx)
#         logit_model = LogisticRegression(penalty='l2', solver='lbfgs')
#         X_train_i = X[train_idx, :]
#         Y_train_i = Y[train_idx]
#         X_test_i = X[test_idx, :]
#         Y_test_i = Y[test_idx]
#         logit_model_fit = logit_model.fit(X_train_i, Y_train_i)
#         Y_pred_i = logit_model_fit.predict(X_test_i)
#         N_test_i = len(test_idx)
#         errs = abs(Y_pred_i - Y_test_i).sum()
#         acc = (N_test_i - errs) / N_test_i
#         accs.append(acc)
    return accs

def run_regression(data, formula, regression_type, dep_var='anchor', out_dir='../../output', split_var=None, split_var_val=0):
    """
    Run logit regression on data with given formula 
    and write to file.
    Option: use regularized logit (reduce variable inflation).
    
    :param data: full data
    :param formula: regression formula
    :param regression_type: type of regression (logit|regularized_logit)
    :param dep_var: dependent variable
    :param out_dir: output directory
    :param split_var: optional variable to split data (e.g. only organization accounts)
    :param split_var_val: value of split value variable (if included)
    """
    l2_weight = 0.01
    max_iter = 100
    model_full = GLM.from_formula(formula, data, family=Binomial(link=logit()))
    if(regression_type == 'regularized_logit'):
        model_res_full = model_full.fit_regularized(maxiter=max_iter, method='elastic_net', alpha=l2_weight, L1_wt=0.0)
    else:
        model_res_full = model_full.fit()
    
    ## summary stats
    model_res_full_err = compute_err_data(model_res_full)
    # write to file
    reg_out_str = 'anchor_%s_output_%s.tsv'%(regression_type, formula.replace(' ', ''))
    if(split_var is not None):
        reg_out_str = 'anchor_%s_output_%s_split_%s=%s.tsv'%(regression_type, formula.replace(' ', ''), split_var, split_var_val)
    res_out_file = os.path.join(out_dir, reg_out_str)
    model_res_full_err.to_csv(res_out_file, sep='\t', index=True)
    
    ## save coeffs to file => pretty print as latex
    # need lots of decimal points! for multiple variable correction
    pd.options.display.float_format = '{:,.5f}'.format
    tex_out_str = reg_out_str.replace('.tsv', '.tex')
    tex_res_out_file = os.path.join(out_dir, tex_out_str)
    model_res_full_err = model_res_full_err.assign(**{'coeff' : model_res_full_err.index})
    tex_data_cols = ['coeff', 'mean', 'SE', 'p_val']
    model_res_full_err.to_latex(tex_res_out_file, columns=tex_data_cols, index=False)

    ## compute regression fit parameters => deviance, AIC, etc.
    # start with chi2 test against null model
    llr, model_df, p_val = compute_chi2_null_test(model_res_full, data, dep_var, max_iter, l2_weight)
    logging.debug('N=%d, LLR=%.5f, df=%d, p-val=%.3E'%(data.shape[0], llr, model_df, p_val))
    # variance inflation factor: are some of the covariates highly collinear?
    # for sanity we only look at non-categorical vars
    cat_var_matcher = re.compile('C\(.+\)\[T\..+\]|Intercept') # format="C(var_name)[T.var_val]" ("C(username)[T.barackobama]")
    non_cat_params = [param for param in model_res_full.params.index if cat_var_matcher.search(param) is None]
    for param in non_cat_params:
        VIF_i = compute_VIF(model_res_full, param)
        logging.debug('VIF test: param=%s, VIF=%.3f'%(param, VIF_i))
    
    ## compute accuracy on k-fold classification
    ## we would use R-squared but that doesn't work for logistic regression
    # first get data into usable format
    n_splits = 10
    accs = k_fold_acc(model_full.exog, model_full.endog, k=n_splits)
    mean_acc = np.mean(accs)
    se_acc = np.std(accs) / n_splits**.5
    logging.debug('%d-fold mean accuracy = %.3f +/- %.3f'%(n_splits, mean_acc, se_acc))

def clean_categorical_data(data, cat_var_1, cat_var_2):
    """
    Clean data by removing instances where categorical variable 1 has
    1 unique type of categorical variable 2 (i.e. resulting in singular matrix).
    
    :param data: data
    :param cat_var_1: categorical variable 1
    :param cat_var_2: categorical variable 2
    :returns data: cleaned data
    """
    data_clean = []
    for cat_var_1_i, data_i in data.groupby(cat_var_1):
        if(data_i.loc[:, cat_var_2].nunique() > 1):
            data_clean.append(data_i)
    data_clean = pd.concat(data_clean, axis=0)
    return data_clean
    
def main():
    parser = ArgumentParser()
    # original data
    parser.add_argument('--clean_data', default='../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed_anchor_NE_peak_times_consistent_authors.gz')
    # original data: regular authors (opposite of active authors)
#     parser.add_argument('--clean_data', default='../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed_anchor_NE_peak_times_consistent_authors_regular_authors.gz')
    # active author data
#     parser.add_argument('--clean_data', default='../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed_anchor_prior_author_stats.gz')
    # categorical vars
#     parser.add_argument('--cat_vars', nargs='+', default=['data_name_fixed', 'username', 'NE_fixed'])
    parser.add_argument('--cat_vars', nargs='+', default=['data_name_fixed', 'NE_fixed'])
    # binary vars: time
#     parser.add_argument('--binary_vars', nargs='+', default=['during_peak', 'post_peak'])
    # binary vars: info + author
#     parser.add_argument('--binary_vars', nargs='+', default=['has_URL', 'image_video_URL', 'organization', 'is_local'])
    # binary vars: time + info
#     parser.add_argument('--binary_vars', nargs='+', default=['during_peak', 'post_peak', 'has_URL', 'image_video_URL'])
    # binary vars: time + info + author
    parser.add_argument('--binary_vars', nargs='+', default=['during_peak', 'post_peak', 'has_URL', 'image_video_URL', 'organization', 'is_local'])
    # scalar vars
    # original data: prior NE count
#     parser.add_argument('--scalar_vars', nargs='+', default=['NE_count_prior'])
    # original data: time since start
#     parser.add_argument('--scalar_vars', nargs='+', default=['since_start'])
    # original data: text length + time since start
#     parser.add_argument('--scalar_vars', nargs='+', default=['txt_len_norm', 'since_start'])
    # original data: time since start + prior NE count
#     parser.add_argument('--scalar_vars', nargs='+', default=['since_start', 'NE_count_prior'])
    # original data: time since start + prior NE count + prior post count
#     parser.add_argument('--scalar_vars', nargs='+', default=['since_start', 'NE_count_prior', 'post_count_prior'])
    # regular author data
    parser.add_argument('--scalar_vars', nargs='+', default=['NE_count_prior', 'post_count_prior', 'post_count', 'NE_count', 'since_start'])
    # active author data
#     parser.add_argument('--scalar_vars', nargs='+', default=['since_start'])
#     parser.add_argument('--scalar_vars', nargs='+', default=['since_start', 'post_count', 'NE_count'])
#     parser.add_argument('--scalar_vars', nargs='+', default=['NE_count_prior', 'post_count', 'NE_count', 'prior_engagement', 'txt_len_norm', 'engagement_change'])
#     parser.add_argument('--scalar_vars', nargs='+', default=['NE_count_prior', 'post_count', 'NE_count', 'prior_engagement', 'engagement_change', 'since_start'])
    # log-scale vars
    # original data: prior NE count
#     parser.add_argument('--log_vars', default=['post_count_prior', 'NE_count_prior', 'txt_len_norm'])
    # active author
#     parser.add_argument('--log_vars', default=['NE_count_prior', 'post_count', 'NE_count'])
#     parser.add_argument('--log_vars', default=['post_count', 'NE_count'])
#     parser.add_argument('--log_vars', default=['prior_engagement'])
    # interact vars
    # active author
#     parser.add_argument('--interact_vars', default=[['post_peak', 'C(username)']])
    # original data
    # split var: do separate regression for data subsets
#     parser.add_argument('--split_var', default='organization')
#     parser.add_argument('--split_var', default='is_local')
#     parser.add_argument('--split_var', default='regular_author')
    # clean vars: remove any null values for var (useful for metadata)
    parser.add_argument('--clean_vars', nargs='+', default=[('is_local', -1), ('organization', -1)])
    # regression type
    parser.add_argument('--regression_type', default='regularized_logit')
#     parser.add_argument('--regression_type', default='logit')
    parser.add_argument('--dep_var', default='anchor')
    # original data
#     parser.add_argument('--out_dir', default='../../output/')
    # regular author data
    parser.add_argument('--out_dir', default='../../output/regular_author_data/')
    # active author data
#     parser.add_argument('--out_dir', default='../../output/power_user_full_data/')
    args = vars(parser.parse_args())
    cat_vars = args.get('cat_vars')
    binary_vars = args.get('binary_vars')
    scalar_vars = args.get('scalar_vars')
    log_vars = args.get('log_vars')
    interact_vars = args.get('interact_vars')
    split_var = args.get('split_var')
    clean_vars = args.get('clean_vars')
    if(cat_vars is None or cat_vars[0]==''):
        cat_vars = []
    if(binary_vars is None or binary_vars[0]==''):
        binary_vars = []
    if(scalar_vars is None or scalar_vars[0]==''):
        scalar_vars = []
    if(log_vars is None or log_vars[0]==''):
        log_vars = []
    if(interact_vars is None or interact_vars[0]==''):
        interact_vars = []
    # convert clean vars to tuples
    if(clean_vars is None or clean_vars[0]==''):
        clean_vars = []
    elif(type(clean_vars[0]) is str):
        clean_vars = list(map(literal_eval, clean_vars))
        
    cat_var_str = '+'.join('C(%s)'%(x) for x in cat_vars)
    binary_var_str = '+'.join(binary_vars)
    scalar_var_str = '+'.join(scalar_vars)
    interact_var_str = '+'.join(['%s*%s'%(i,j) for (i,j) in interact_vars])
    var_strs = [cat_var_str, binary_var_str, scalar_var_str, interact_var_str]
    valid_var_str = [var_str for var_str in var_strs if var_str != '']
    combined_var_str = '+'.join(valid_var_str)
    formula = '%s~%s'%(args['dep_var'], combined_var_str)
    
    ## set up logging
    logging_file = os.path.join(args['out_dir'], 'run_descriptor_regression_%s.txt'%(formula))
    if(split_var is not None):
        logging_file = os.path.join(args['out_dir'], 'run_descriptor_regression_%s_split_%s.txt'%(formula, split_var))
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)

    ## load data
    if(args['clean_data'].endswith('gz')):
        anchor_data = pd.read_csv(args['clean_data'], sep='\t', index_col=False, compression='gzip')
    else:
        anchor_data = pd.read_csv(args['clean_data'], sep='\t', index_col=False)
    
    ## debug: check author vs. scalar
#     anchor_data_pre_transform = anchor_data.copy()
#     for author_i, data_i in anchor_data_pre_transform.groupby('username'):
#         scalar_var_err = data_i.loc[:, scalar_vars].std(axis=0)
#         scalar_var_err_min = scalar_var_err.min()
#         if(scalar_var_err_min <= 1e-3):
#             logging.debug('pre-transform debug: author %s has 0 err for scalar var'%(author_i))
    # log-scale vars
    v_smooth = 1
    for v in log_vars:
        anchor_data = anchor_data.assign(**{v : np.log(anchor_data.loc[:, v] + v_smooth)})
    # Z-norm all scalar vars
    scaler = StandardScaler()
    for v in scalar_vars:
        anchor_data = anchor_data.assign(**{v : scaler.fit_transform(anchor_data.loc[:, v].values.reshape(-1,1))})
    # optional: remove data with null values
    if(len(clean_vars) > 0):
        anchor_data_clean = anchor_data.copy()
        for clean_var, null_val in clean_vars:
            anchor_data_clean = anchor_data_clean[anchor_data_clean.loc[:, clean_var] != null_val]
            logging.debug('cleaning var %s, retained %d/%d data'%(clean_var, anchor_data_clean.shape[0], anchor_data.shape[0]))
        anchor_data = anchor_data_clean.copy()
    
    ## debug: remove NE/author combos
#     anchor_data = clean_categorical_data(anchor_data, 'username', 'NE_fixed')
    
    ## debug: check author vs. scalar
    author_var = 'username'
    for author_i, data_i in anchor_data.groupby(author_var):
        scalar_var_err = data_i.loc[:, scalar_vars].std(axis=0)
        scalar_var_err_min = scalar_var_err.min()
        if(scalar_var_err_min <= 1e-3):
            logging.debug('debug: author %s has 0 err for scalar data:\n%s'%(author_i, data_i.loc[:, scalar_vars]))
#             pre_transform_data_i = anchor_data[anchor_data.loc[:, 'username']==author_i]
#             logging.debug('pre-transform compare: author %s has scalar data:\n%s\n'%(author_i, pre_transform_data_i.loc[:, scalar_vars]))
#             logging.debug('pre-transform compare: author %s has scalar err:\n%s\n'%(author_i, pre_transform_data_i.loc[:, scalar_vars].std(axis=0)))
    
    ## remove singular factors that result from
    ## categorical variables correlating exactly with binary variables
    allowed_binary_vars = set(['organization', 'is_local'])
    filter_binary_vars = set(binary_vars) - allowed_binary_vars
    anchor_data_clean = []
    for cat_var in cat_vars:
        anchor_data_clean_v = []
        for cat_var_i, data_i in anchor_data.groupby(cat_var):
            singular_factor = False
            for binary_var in filter_binary_vars:
                if(data_i.loc[:, binary_var].var()==0.):
                    singular_factor = True
                    break
            if(not singular_factor):
                anchor_data_clean_v.append(data_i)
        anchor_data_clean_v = pd.concat(anchor_data_clean_v, axis=0)
        anchor_data_clean.append(anchor_data_clean_v)
    # restrict to data intersection
    id_var = 'id'
    anchor_data_clean_ids = set.intersection(*[set(anchor_data_clean_v.loc[:, id_var].unique()) for anchor_data_clean_v in anchor_data_clean])
    anchor_data_clean = anchor_data[anchor_data.loc[:, id_var].isin(anchor_data_clean_ids)]
    logging.debug('%d/%d clean data without categorical/binary correlation'%(anchor_data_clean.shape[0], anchor_data.shape[0]))
    anchor_data = anchor_data_clean.copy()
    
    ## fit model with full data 
    ## regularization with best hyperparam in terms of LL
    ## scripts/data_processing/compare_anchor_probability_before_after_peak_in_twitter.ipynb#Fixed-effect-regression,-regularized
    if(split_var is None):
        run_regression(anchor_data, formula, args['regression_type'], dep_var=args['dep_var'], out_dir=args['out_dir'])
    else:
        # 
        # restrict data to valid data (i.e. split var is not null)
        split_var_null_val = -1
        anchor_data = anchor_data[anchor_data.loc[:, split_var] != split_var_null_val]
        for split_var_i, data_i in anchor_data.groupby(split_var):
            # remove possibly singular data
            author_var = 'username'
            NE_var = 'NE_fixed'
            data_name_var = 'data_name_fixed'
            data_i = clean_categorical_data(data_i, author_var, NE_var)
#             data_i = clean_categorical_data(data_i, NE_var, data_name_var)
            run_regression(data_i, formula, args['regression_type'], dep_var=args['dep_var'], out_dir=args['out_dir'], split_var=split_var, split_var_val=split_var_i)
    
if __name__ == '__main__':
    main()