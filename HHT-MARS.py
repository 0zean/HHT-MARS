#from pyhht.visualization import plot_imfs
from pyhht import EMD
from pyearth import Earth
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
import xgboost as xgb

np.random.seed(123)

ticker = 'SPY'
stock = pdr.get_data_yahoo(ticker.upper(), start='2009-01-01', end=str(datetime.now())[0:11])
stock = stock.Close
returns = np.log(stock).diff().dropna()


decomposer = EMD(returns)
imfs = decomposer.decompose()
#t = np.linspace(0, 1, len(returns))
#plot_imfs(returns, imfs, t) 

imfs = imfs.T
imfs = pd.DataFrame(imfs)


series = returns[len(returns)-2000:]
series = np.array(series)
series = series.reshape(-1, 1)

D = 4
T = 1
N = 2000
series = series[500:]
lbls = np.zeros((N - 500 - T - (D - 1) * T,))

for t in range((D - 1) * T, N - 500 - T):
	lbls[t - (D - 1) * T] = series[t + T]
trnLbls = lbls[:lbls.size - round(lbls.size * 0.3)]
chkLbls = lbls[lbls.size - round(lbls.size * 0.3):]


def HHT_MARS_TEST(series, regressors=4, delay=1, N=2000):
    series = series[len(series)-2000:]
    series = np.array(series)
    series = series.reshape(-1, 1)

    D = regressors  # number of regressors
    T = delay  # delay
    N = N
    series = series[500:]
    data = np.zeros((N - 500 - T - (D - 1) * T, D))
    lbls = np.zeros((N - 500 - T - (D - 1) * T,))

    for t in range((D - 1) * T, N - 500 - T):
        data[t - (D - 1) * T, :] = [series[t - 3 * T], series[t - 2 * T], series[t - T], series[t]]
        lbls[t - (D - 1) * T] = series[t + T]
    trnData = data[:lbls.size - round(lbls.size * 0.3), :]
    trnLbls = lbls[:lbls.size - round(lbls.size * 0.3)]
    chkData = data[lbls.size - round(lbls.size * 0.3):, :]
    chkLbls = lbls[lbls.size - round(lbls.size * 0.3):]

    aa = np.array(chkLbls[-4:]).reshape(1, -1)
    chkData = np.append(chkData, aa, axis=0)

    mars = Earth()
    mars.fit(trnData, trnLbls)
    boosted_mars = AdaBoostRegressor(base_estimator=mars, n_estimators=25, learning_rate=0.1, loss='exponential')
    bag = BaggingRegressor(base_estimator=mars, n_estimators=25)
    bag.fit(trnData, trnLbls)
    boosted_mars.fit(trnData, trnLbls)
    pred2 = bag.predict(chkData)
    oos_preds = boosted_mars.predict(chkData)
    
    stack_predict = np.vstack([oos_preds, pred2]).T
    
    params_xgd = {
            'max_depth': 7,
            'objective': 'reg:linear',
            'learning_rate': 0.05,
            'n_estimators': 10000
            }
    clf = xgb.XGBRegressor(**params_xgd)
    clf.fit(stack_predict[:-1,:], chkLbls, eval_set=[(stack_predict[:-1,:], chkLbls)], 
        eval_metric='rmse', early_stopping_rounds=20, verbose=False)

    xgb_pred = clf.predict(stack_predict)

    return xgb_pred


imf1 = pd.DataFrame(HHT_MARS_TEST(imfs[0]))
imf2 = pd.DataFrame(HHT_MARS_TEST(imfs[1]))
imf3 = pd.DataFrame(HHT_MARS_TEST(imfs[2]))
imf4 = pd.DataFrame(HHT_MARS_TEST(imfs[3]))
imf5 = pd.DataFrame(HHT_MARS_TEST(imfs[4]))
imf6 = pd.DataFrame(HHT_MARS_TEST(imfs[5]))
imf7 = pd.DataFrame(HHT_MARS_TEST(imfs[6]))
imf8 = pd.DataFrame(HHT_MARS_TEST(imfs[7]))
imf9 = pd.DataFrame(HHT_MARS_TEST(imfs[8]))
imfr = pd.DataFrame(HHT_MARS_TEST(imfs[9]))


imf1 = imf1.rename(columns={0: 'imf1'})
imf2 = imf2.rename(columns={0: 'imf2'})
imf3 = imf3.rename(columns={0: 'imf3'})
imf4 = imf4.rename(columns={0: 'imf4'})
imf5 = imf5.rename(columns={0: 'imf5'})
imf6 = imf6.rename(columns={0: 'imf6'})
imf7 = imf7.rename(columns={0: 'imf7'})
imf8 = imf8.rename(columns={0: 'imf8'})
imf9 = imf9.rename(columns={0: 'imf9'})
imfr = imfr.rename(columns={0: 'imfr'})


imf1 = pd.concat([imf1, imf2], axis=1)
imf1 = pd.concat([imf1, imf3], axis=1)
imf1 = pd.concat([imf1, imf4], axis=1)
imf1 = pd.concat([imf1, imf5], axis=1)
imf1 = pd.concat([imf1, imf6], axis=1)
imf1 = pd.concat([imf1, imf7], axis=1)
imf1 = pd.concat([imf1, imf8], axis=1)
imf1 = pd.concat([imf1, imf9], axis=1)
imf1 = pd.concat([imf1, imfr], axis=1)


imf1['sum'] = imf1.sum(axis=1)
plt.plot(np.array(imf1['sum']), color='g')
plt.plot(chkLbls)
plt.legend(['Pred','Actual'])


DA_test = pd.DataFrame(chkLbls)
DA_test['com'] = DA_test[0].shift(1)
DA_test['ACC'] = DA_test[0] - DA_test['com']
DA_test['ACC'] = DA_test['ACC'].mask(DA_test['ACC'] > 0 , 1)
DA_test['ACC'] = DA_test['ACC'].mask(DA_test['ACC'] < 0 , 0)

DA_pred = pd.DataFrame(imf1['sum'])
DA_pred['com'] = DA_pred['sum'].shift(1)
DA_pred['ACC2'] = DA_pred['sum'] - DA_pred['com']
DA_pred['ACC2'] = DA_pred['ACC2'].mask(DA_pred['ACC2'] > 0 , 1)
DA_pred['ACC2'] = DA_pred['ACC2'].mask(DA_pred['ACC2'] < 0 , 0)

DA = pd.DataFrame(DA_test['ACC'])
DA = DA.join(DA_pred['ACC2'])
DA['score'] = 0
DA['score'] = DA['score'].mask(DA['ACC'] == DA['ACC2'], 1)

AC = DA['score'].value_counts()
ACC = round((AC[1] / len(chkLbls)) * 100, 3)

print('Directional Accuracy: ' + str(ACC) + ' %')

pred = imf1['sum']
pred = pred[:-1]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

print('MSE: ' + str(mean_squared_error(chkLbls, pred)))
print('RMSE: ' + str(rmse(np.array(pred), chkLbls)))
print('R Squared: ' + str(r2_score(chkLbls, pred)))

Direction = imf1['sum']

if Direction[449] > Direction[448]:
    print("UP")
else:
    print("DOWN")
