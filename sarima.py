import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import time
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_absolute_error


def smape(preds, target):
    '''
    Function to calculate SMAPE
    '''
    n = len(preds)
    # masked_arr = ~((preds==0) & (target==0))
    # preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds)+np.abs(target)
    smape_val = (200*np.sum(num/denom))/n
    return smape_val


def visual_stationary_check(ts, window):
    """
    Visualize ts, its moving average and moving std with window length of window. This is the simplest check for
    stationarity of the provided time series

    :param ts : pandas Series data
    :param window : int
    :return None
    """
    # finding moving average and plotting it
    moving_avg = ts.rolling(window).mean()
    moving_std = ts.rolling(window).std()

    #plotting
    plt.plot(ts, color="blue", label="Original Series")
    plt.plot(moving_avg, color='red', label="Moving average")
    plt.plot(moving_std, color='green', label="Moving std")
    plt.legend()
    plt.show()


def perform_adfuler(ts):
    """
    Return True iff ts is recognized to be a stationary time-series usind adfuller test. Also print the results of
    test in a nice way to interpret

    More info: https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html

    :param ts: Series time series data
    :return:
    """

    # we use AIC criteria (Akaike information criteria)
    result = \
        sm.tsa.stattools.adfuller(ts, autolag="AIC")
    test_stat, p_val, critical_vals = result[0], result[1], result[4]
    print("The resutls are:")
    print("Test statistic: %f" % test_stat)
    print("Critical value 1%%: %f" % critical_vals["1%"])
    print("Critical value 5%%: %f" % critical_vals["5%"])
    print("Critical value 10%%: %f" % critical_vals["10%"])
    print("p-value: %f" % p_val)

    return p_val < 0.05, result


def make_stationary_exp(ts, half_life):
    """
    Return input ts converted to a stationary time series.

    We use removal of exponentially weighted moving average to get rid of trend and seasonality and return the movign
    average

    :param ts: Series input time series
    :param half_life : half_life used for exponential moving average,
     (see https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.ewm.html)
    :return: stat_ts, moving_avg
    """

    log_ts = np.log(ts)
    moving_avg = log_ts.ewm(halflife=half_life).mean()
    return log_ts - moving_avg, moving_avg


def decompose(ts, visualize=True):
    """
    Return ts decomposed into trend, seasonality and residual

    See :  https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html

    :param ts: Series input time series
    :return:  trend, seasonality, residual
    """
    dec = sm.tsa.seasonal_decompose(ts)

    if visualize:
        plt.plot(dec.observed, color="blue", label="Observed series")
        plt.plot(dec.trend, color="red", label="Trend")
        plt.plot(dec.seasonal, color="green", label="Seasonality")
        plt.plot(dec.resid, color="black", label="Residual series")
        plt.legend()
        plt.show()
    return dec.trend, dec.seasonal, dec.resid


def fit_arma(ts, p=1, q=1):
    """
    Fit an ARMA model on time series and return predcitions using this model

    :param ts: Series input time series data
    :param p: int, order of AR process
    :param q: int, order of MA process
    :return: result
    """
    trend, seasonal, resid = decompose(ts, visualize=False)
    resid.dropna(inplace=True)
    model = sm.tsa.ARMA(resid, order=(p, q), dates=ts.index)
    arma_fitted = model.fit(disp=-1)  # disp=-1 suppresses fitting information
    return arma_fitted.fittedvalues + trend + seasonal


def fit_arma_ewm(ts, p=1, q=1):
    """
    Fit an ARMA model on time series and return predcitions using this model

    :param ts: Series input time series data
    :param p: int, order of AR process
    :param q: int, order of MA process
    :return: result
    """
    ewm_ts, moving_avg = make_stationary_exp(ts, 12)
    model = sm.tsa.ARMA(ewm_ts, order=(p, q), dates=ts.index)
    arma_fitted = model.fit(disp=-1)  # disp=-1 suppresses fitting information
    return np.exp(arma_fitted.fittedvalues + moving_avg)


def fit_arima(ts, p=1, q=1, d=1):
    """
    Fit an ARMA model on time series and return predcitions using this model

    :param ts: Series input time series data
    :param p: int, order of AR process
    :param q: int, order of MA process
    :return: result
    """
    ewm_ts, moving_avg = make_stationary_exp(ts, 12)
    model = sm.tsa.ARIMA(ewm_ts, order=(p, d, q), dates=ewm_ts.index)
    arima_fitted = model.fit(disp=-1)  # disp=-1 suppresses fitting information
    return np.exp(arima_fitted.fittedvalues.cumsum() + ewm_ts[0] + moving_avg)


def nested_cv(ts, order, s_order):
    """
    Peform nested cross-validation on sample ts fitting a SARIMA model with parameters order and seasonal parameters
    s_order and return MAE.

    :param sample:  Series, time series data
    :param order:  Tuple (p, d, q) order of ARIMA model
    :param s_order: Tuple (p, d, q) order of seasonal part of SARIMA model
    :return: MAE
    """
    ts = np.log(ts)
    train_size = int(len(ts) * 0.66)
    train, test = ts[:train_size], ts[train_size:]
    history = list(train.values)

    predictions = []
    for t in range(len(test)):
        model = sm.tsa.SARIMAX(history, order=order, seasonal_order=s_order)
        model_fit = model.fit(disp=0)
        y_hat = model_fit.forecast()[0]
        predictions.append(np.exp(y_hat))
        history.append(test[t])

    prediction_series = pd.Series(predictions, test.index)
    return smape(predictions, np.exp(test)), prediction_series


if __name__ == "__main__":
    # playground area
    parser = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    data_train = pd.read_csv("train_small.csv", parse_dates=["date"], index_col="date", date_parser=parser)

    # explore this data set
    print(data_train.describe())
    print(data_train.head())
    ask_user = False  # determine whether or not to ask user to choose model order or use defaults


    # it looks like we have a synthetic data set, no missing information for each store or item, in fact we have equal
    # amount of data points for all 10 items across 5 stores :
    # length of data : (5 * 365 + 1 day for leap year 2012) * 5 * 10 = 913000
    series_dict = {}
    for store_idx in range(1, 2):
        for item_idx in range(1, 3):
            key = "{}-{}".format(store_idx, item_idx)
            store_bool = data_train['store'] == store_idx
            item_bool = data_train["item"] == item_idx
            s = data_train[store_bool & item_bool]["sales"]
            series_dict[key] = s

    # lets focus on series for store 1, item 1
    sample = series_dict['1-1']
    print(sample.describe)

    # visual test for stationarity
    #visual_stationary_check(sample, 12)

    # I know this is not stationary but just for fun, trying dicky-fueler test
    is_stat, adf_result = perform_adfuler(sample)

    # setting model orders
    d = 1
    D = 0
    if not ask_user:
        p = 5
        q = 1
        P = 3
        Q = 0
        s = 7
    else:

        # determining seasonality length using acf
        acf = sm.tsa.stattools.pacf(sample, nlags=20)
        plt.plot(acf)
        plt.title('ACF')
        plt.show()
        s = int(input("Enter seasonality length"))

        # decomposing time-series to estimate the orders of models, the manual way!
        trend, season, resid = decompose(sample, visualize=False)

        # using acf and pacf on resid to determine ordes of MA(q) and AR(p)
        acf = sm.tsa.stattools.acf(sample, nlags=20)
        plt.plot(acf)
        plt.title('ACF')
        plt.show()
        q = int(input("Based on ACF, enter MA order: "))

        pacf = sm.tsa.stattools.pacf(sample, nlags=20)
        plt.plot(pacf)
        plt.title('PACF')
        plt.show()
        p = int(input("Based on PACF, enter AR order: "))

        # repeating same for seasonal part (this could go in a function!)
        acf = sm.tsa.stattools.acf(season, nlags=20)
        plt.plot(acf)
        plt.title('ACF')
        plt.show()
        Q = int(input("Based on ACF, enter MA order for seasonal part: "))

        pacf = sm.tsa.stattools.pacf(season, nlags=20)
        plt.plot(pacf)
        plt.title('PACF')
        plt.show()
        P = int(input("Based on PACF, enter AR order for seosnal part: "))


    # calling nested cross-validation function and printing result
    p_values = []
    error, predictions = nested_cv(sample, (p, d, q), (P, D, Q, s))
    plt.plot(sample, color="blue", label="Original series")
    plt.plot(predictions, color="red", label="Predictions")
    plt.title('SMAPE = {}'.format(error))
    plt.legend()
    plt.show()



