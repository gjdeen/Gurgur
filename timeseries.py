import numpy as np
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math as m
import itertools
import pygam
import warnings
import calendar

def get_regressors(y,x_array,daily=True,weekly=True,seasonal=True,holidays=True,vacation=True,exogenous=True\
    ,daily_=None,weekly_=None,seasonal_=None,holidays_=None,vacation_=None,exogenous_=None):

    if y.index[1].minute ==15:
        hourly=False
    else:
        hourly=True

    if daily_ == None:
        if hourly:
            daily_ = [str(i) for i in np.arange(0,24)]
        else:
            daily_ = [str(i) for i in np.arange(0,24,0.25)]
    if weekly == True:
        weekly_ = ['mon','tue','wen','thu','fri','sat','sun']
    else:
        weekly_ = []

    if seasonal == True:
        seasonal_ = ['sin1','cos1','sin2','cos2']
    else:
        seasonal_ = []

    if holidays == False:
        holidays_ = []
    else:
        holidays = ['holiday']

    if vacation == False:
        vacation = []
    else:
        vacation = ['vacation']

    if exogenous_==None:
        exogenous_ = []
    else:
        exogenous_ = exogenous_

    df = pd.DataFrame(y)
    df.columns = ['y']

    if len(exogenous_)==0:
        for i,x in enumerate(x_array):
            exogenous_.append('exog{}'.format(i))
            df[exogenous_[i]] = x
    else:
        for i,exog in enumerate(exogenous_):
            df[exog] = x_array[i]

    if len(holidays) > 0:
        df['holiday'] = 0
        for date in holidays_:
            try:
                df['holiday'][date[0]] = 1
            except:
                pass

    if len(vacation) > 0:
        df['vacation'] = 0
        for row in vacation_:
            try:
                df['vacation'][row[0]:row[1]] = 1
            except:
                pass

    # Check for NaN's, otherwise the sm.OLS function raises an error
    if np.sum(df.isnull().values) > 0:
        df=df.fillna(method='ffill')


    if daily==True:
        df['TimeOfDay'] = df.index.hour+df.index.minute/60
        for hour in daily_:
            df[hour] = 0
            df[hour][df.TimeOfDay==float(hour)] = 1

    if weekly==True:
        df['day'] = df.index.dayofweek
        for day in weekly_:
            i = ['mon','tue','wen','thu','fri','sat','sun'].index(day)
            df[day] = 0
            df[day][df.day==i] = 1

    if seasonal==True:
        n_days = (365+calendar.isleap(df.index.year[0]))
        df['decimal'] = (df.index.dayofyear-1)/n_days+df.TimeOfDay/(n_days*24)
        df['sin1'] = np.sin(2*m.pi*df.decimal)
        df['cos1'] = np.cos(2*m.pi*df.decimal)
        df['sin2'] = np.sin(4*m.pi*df.decimal)
        df['cos2'] = np.cos(4*m.pi*df.decimal)

    return df[['y']+daily_+weekly_+seasonal_+holidays+vacation+exogenous_], daily_+weekly_+seasonal_+holidays+vacation, exogenous_

def lm(df,xvars):

    x = df[xvars]
    x = sm.add_constant(x)
    y = df.y

    model = sm.OLS(y,x)
    results = model.fit()

    print(results.summary())

    return results

def get_trend(df,results):

    trend = results.params.const + np.sum([results.params[param]*\
            df[param] for param in results.params.index[results.params.index!='const']],axis=0)

    return trend

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def forward_selection(data_train,response,method,display=True):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data_train.columns)
    remaining.remove(response)
    selected = []

    current_score, best_new_score = 1e99, 1e99

    while remaining and current_score == best_new_score:
        scores_with_candidates = []

        for candidate in remaining:
            if method=='lm':
                res_temp = sm.OLS(data_train[response], sm.add_constant(data_train[selected+[candidate]])).fit()
                score = res_temp.aic
                scores_with_candidates.append((score, candidate,res_temp.pvalues[-1]))
            elif method=='glm':
                res_temp = sm.GLM(data_train[response], sm.add_constant(data_train[selected+[candidate]]),family=sm.families.Gaussian()).fit()
                score = res_temp.aic
                scores_with_candidates.append((score, candidate,res_temp.pvalues[-1]))
            elif method=='gam':
                res_temp = pygam.LinearGAM().fit(data_train[selected+[candidate]],data_train[response])
                score = res_temp.statistics_['AIC']
                scores_with_candidates.append((score, candidate,res_temp.statistics_['p_values'][-1]))

        scores_with_candidates.sort()
        best_new_score, best_candidate, p_value = scores_with_candidates[::-1].pop()

        # print(best_candidate,best_new_score,p_value, current_score)
        if (current_score > best_new_score) & (p_value<=0.05):
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score

    if method=='lm':
        res_temp = sm.OLS(data_train[response], sm.add_constant(data_train[selected])).fit()
    elif method=='glm':
        res_temp = sm.GLM(data_train[response], sm.add_constant(data_train[selected]),family=sm.families.Gaussian()).fit()
    elif method=='gam':
        res_temp = pygam.LinearGAM().fit(data_train[selected],data_train[response])

    if display==True:
        print(res_temp.summary())

    return res_temp

def backward_reduction(data_train,response):
    """Linear model designed by backward reduction.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted AIC
    """
    remaining = set(data_train.columns)
    remaining.remove(response)


    res_temp = sm.OLS(data_train[response], sm.add_constant(data_train[list(remaining)])).fit()
    score_init = res_temp.aic

    current_score, best_new_score = score_init, -1e99

    while remaining and best_new_score < current_score:
        scores_with_candidates = []
        if best_new_score!=-1e99:
            current_score = best_new_score

        for candidate in remaining:
            selected = remaining.copy()
            selected.remove(candidate)
            res_temp = sm.OLS(data_train[response], sm.add_constant(data_train[list(selected)])).fit()
            score = res_temp.aic
            scores_with_candidates.append((score, candidate))

        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[::-1].pop()

        print(best_candidate,best_new_score, current_score,end='\n')
        if (best_new_score < current_score):
            remaining.remove(best_candidate)

    res_temp = sm.OLS(data_train[response], sm.add_constant(data_train[list(remaining)])).fit()

    return res_temp

def get_rmse(y, y_hat):
    '''Root Mean Square Error
    https://en.wikipedia.org/wiki/Root-mean-square_deviation
    '''
    mse = np.mean((y - y_hat)**2)
    return np.sqrt(mse)

def get_mape(y, y_hat):
    '''Mean Absolute Percent Error
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    '''
    perc_err = (100*(y - y_hat))/y
    return np.mean(abs(perc_err))

def get_mase(y, y_hat):
    '''Mean Absolute Scaled Error
    https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
    '''
    abs_err = abs(y - y_hat)
    dsum=sum(abs(y[1:] - y_hat[1:]))
    t = len(y)
    denom = (1/(t - 1))* dsum
    return np.mean(abs_err/denom)



def tsplot(y, lags=None, title='', figsize=(14, 8)):
    '''Examine the patterns of ACF and PACF, along with the time series plot and histogram.

    Original source: https://tomaugspurger.github.io/modern-7-timeseries.html
    '''
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax   = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax  = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax

def test_stationarity(timeseries,
                      maxlag=None, regression=None, autolag=None,
                      window=None, plot=False, verbose=False):
    '''
    Check unit root stationarity of time series.

    Null hypothesis: the series is non-stationary.
    If p >= alpha, the series is non-stationary.
    If p < alpha, reject the null hypothesis (has unit root stationarity).

    Original source: http://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/

    Function: http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.stattools.adfuller.html

    window argument is only required for plotting rolling functions. Default=4.
    '''

    # set defaults (from function page)
    if regression is None:
        regression = 'c'

    if verbose:
        print('Running Augmented Dickey-Fuller test with paramters:')
        print('maxlag: {}'.format(maxlag))
        print('regression: {}'.format(regression))
        print('autolag: {}'.format(autolag))

    if plot:
        if window is None:
            window = 4
        #Determing rolling statistics
        rolmean = timeseries.rolling(window=window, center=False).mean()
        rolstd = timeseries.rolling(window=window, center=False).std()

        #Plot rolling statistics:
        orig = plt.plot(timeseries, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean ({})'.format(window))
        std = plt.plot(rolstd, color='black', label='Rolling Std ({})'.format(window))
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)

    #Perform Augmented Dickey-Fuller test:
    dftest = smt.adfuller(timeseries, maxlag=maxlag, regression=regression, autolag=autolag)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                                             'p-value',
                                             '#Lags Used',
                                             'Number of Observations Used',
                                            ])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    if verbose:
        print('Results of Augmented Dickey-Fuller Test:')
        print(dfoutput)
    return dfoutput

def model_resid_stats(model_results,
                      het_method='breakvar',
                      norm_method='jarquebera',
                      sercor_method='ljungbox',
                      verbose=True,
                      ):
    '''More information about the statistics under the ARIMA parameters table, tests of standardized residuals:

    Test of heteroskedasticity
    http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_heteroskedasticity.html#statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_heteroskedasticity

    Test of normality (Default: Jarque-Bera)
    http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_normality.html#statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_normality

    Test of serial correlation (Default: Ljung-Box)
    http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_serial_correlation.html
    '''
    # Re-run the ARIMA model statistical tests, and more. To be used when selecting viable models.
    (het_stat, het_p) = model_results.test_heteroskedasticity(het_method)[0]
    norm_stat, norm_p, skew, kurtosis = model_results.test_normality(norm_method)[0]
    sercor_stat, sercor_p = model_results.test_serial_correlation(method=sercor_method)[0]
    sercor_stat = sercor_stat[-1] # last number for the largest lag
    sercor_p = sercor_p[-1] # last number for the largest lag

    # Run Durbin-Watson test on the standardized residuals.
    # The statistic is approximately equal to 2*(1-r), where r is the sample autocorrelation of the residuals.
    # Thus, for r == 0, indicating no serial correlation, the test statistic equals 2.
    # This statistic will always be between 0 and 4. The closer to 0 the statistic,
    # the more evidence for positive serial correlation. The closer to 4,
    # the more evidence for negative serial correlation.
    # Essentially, below 1 or above 3 is bad.
    dw_stat = sm.stats.stattools.durbin_watson(model_results.filter_results.standardized_forecasts_error[0, model_results.loglikelihood_burn:])

    # check whether roots are outside the unit circle (we want them to be);
    # will be True when AR is not used (i.e., AR order = 0)
    arroots_outside_unit_circle = np.all(np.abs(model_results.arroots) > 1)
    # will be True when MA is not used (i.e., MA order = 0)
    maroots_outside_unit_circle = np.all(np.abs(model_results.maroots) > 1)

    if verbose:
        print('Test heteroskedasticity of residuals ({}): stat={:.3f}, p={:.3f}'.format(het_method, het_stat, het_p));
        print('\nTest normality of residuals ({}): stat={:.3f}, p={:.3f}'.format(norm_method, norm_stat, norm_p));
        print('\nTest serial correlation of residuals ({}): stat={:.3f}, p={:.3f}'.format(sercor_method, sercor_stat, sercor_p));
        print('\nDurbin-Watson test on residuals: d={:.2f}\n\t(NB: 2 means no serial correlation, 0=pos, 4=neg)'.format(dw_stat))
        print('\nTest for all AR roots outside unit circle (>1): {}'.format(arroots_outside_unit_circle))
        print('\nTest for all MA roots outside unit circle (>1): {}'.format(maroots_outside_unit_circle))

    stat = {'het_method': het_method,
            'het_stat': het_stat,
            'het_p': het_p,
            'norm_method': norm_method,
            'norm_stat': norm_stat,
            'norm_p': norm_p,
            'skew': skew,
            'kurtosis': kurtosis,
            'sercor_method': sercor_method,
            'sercor_stat': sercor_stat,
            'sercor_p': sercor_p,
            'dw_stat': dw_stat,
            'arroots_outside_unit_circle': arroots_outside_unit_circle,
            'maroots_outside_unit_circle': maroots_outside_unit_circle,
            }
    return stat

def model_gridsearch(ts,
                     p_min,
                     d_min,
                     q_min,
                     p_max,
                     d_max,
                     q_max,
                     sP_min,
                     sD_min,
                     sQ_min,
                     sP_max,
                     sD_max,
                     sQ_max,
                     trends,
                     exog=None,
                     s=None,
                     enforce_stationarity=True,
                     enforce_invertibility=True,
                     simple_differencing=False,
                     plot_diagnostics=False,
                     verbose=False,
                     filter_warnings=True,
                    ):
    '''Run grid search of SARIMAX models and save results.
    '''

    cols = ['p', 'd', 'q', 'sP', 'sD', 'sQ', 's', 'trend',
            'enforce_stationarity', 'enforce_invertibility', 'simple_differencing',
            'aic', 'bic',
            'het_p', 'norm_p', 'sercor_p', 'dw_stat',
            'arroots_gt_1', 'maroots_gt_1',
            'datetime_run']

    # Initialize a DataFrame to store the results
    df_results = pd.DataFrame(columns=cols)

    # # Initialize a DataFrame to store the results
    # results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
    #                            columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])

    mod_num=0
    for trend,p,d,q,sP,sD,sQ in itertools.product(trends,
                                                  range(p_min,p_max+1),
                                                  range(d_min,d_max+1),
                                                  range(q_min,q_max+1),
                                                  range(sP_min,sP_max+1),
                                                  range(sD_min,sD_max+1),
                                                  range(sQ_min,sQ_max+1),
                                                  ):
        print(p,d,q,sP,sD,sQ,end='\r')
        # initialize to store results for this parameter set
        this_model = pd.DataFrame(index=[mod_num], columns=cols)

        if p==0 and d==0 and q==0:
            continue

        try:
            model = smt.SARIMAX(ts,
                                   trend=trend,
                                   order=(p, d, q),
                                   seasonal_order=(sP, sD, sQ, s),
                                   enforce_stationarity=enforce_stationarity,
                                   enforce_invertibility=enforce_invertibility,
                                   simple_differencing=simple_differencing,
                                   exog=exog
                                  )

            if filter_warnings is True:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    model_results = model.fit(disp=0)
            else:
                model_results = model.fit()

            if verbose:
                print(model_results.summary())

            if plot_diagnostics:
                model_results.plot_diagnostics();

            stat = model_resid_stats(model_results,
                                     verbose=verbose)

            this_model.loc[mod_num, 'p'] = p
            this_model.loc[mod_num, 'd'] = d
            this_model.loc[mod_num, 'q'] = q
            this_model.loc[mod_num, 'sP'] = sP
            this_model.loc[mod_num, 'sD'] = sD
            this_model.loc[mod_num, 'sQ'] = sQ
            this_model.loc[mod_num, 's'] = s
            this_model.loc[mod_num, 'trend'] = trend
            this_model.loc[mod_num, 'enforce_stationarity'] = enforce_stationarity
            this_model.loc[mod_num, 'enforce_invertibility'] = enforce_invertibility
            this_model.loc[mod_num, 'simple_differencing'] = simple_differencing

            this_model.loc[mod_num, 'aic'] = model_results.aic
            this_model.loc[mod_num, 'bic'] = model_results.bic

            # this_model.loc[mod_num, 'het_method'] = stat['het_method']
            # this_model.loc[mod_num, 'het_stat'] = stat['het_stat']
            this_model.loc[mod_num, 'het_p'] = stat['het_p']
            # this_model.loc[mod_num, 'norm_method'] = stat['norm_method']
            # this_model.loc[mod_num, 'norm_stat'] = stat['norm_stat']
            this_model.loc[mod_num, 'norm_p'] = stat['norm_p']
            # this_model.loc[mod_num, 'skew'] = stat['skew']
            # this_model.loc[mod_num, 'kurtosis'] = stat['kurtosis']
            # this_model.loc[mod_num, 'sercor_method'] = stat['sercor_method']
            # this_model.loc[mod_num, 'sercor_stat'] = stat['sercor_stat']
            this_model.loc[mod_num, 'sercor_p'] = stat['sercor_p']
            this_model.loc[mod_num, 'dw_stat'] = stat['dw_stat']
            this_model.loc[mod_num, 'arroots_gt_1'] = stat['arroots_outside_unit_circle']
            this_model.loc[mod_num, 'maroots_gt_1'] = stat['maroots_outside_unit_circle']

            this_model.loc[mod_num, 'datetime_run'] = pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')

            df_results = df_results.append(this_model)
            mod_num+=1

        except:
            continue
    return df_results
