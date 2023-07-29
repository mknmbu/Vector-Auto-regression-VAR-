#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import datetime as datetime


# In[2]:


df = pd.read_csv('Crack_eia.csv',index_col='Date', parse_dates=True)
df=df.dropna()
print('Shape of data',df.shape)
df.head()


# In[3]:


df.describe()


# In[4]:


# create new variables
df[ 'lck'] = np.log(df['crack'])
df['dlck'] = df['lck'] - df['lck'].shift(1)
df['lwti' ] = np.log(df['wti'])
df['lgas' ] = np.log(df['gasoline'])
df['lheat' ] = np.log(df['heatoil'])


# In[5]:


df.describe()


# In[6]:


#
df=df.dropna()
df = df.sort_index().copy()


# In[7]:


print(df.info())
print(df.head())


# In[8]:


# plot level data
fig, ax = plt.subplots(2,1,figsize=(9,8))
ax[0].plot(df['wti'],label="Wti Futures", color="black")
ax[0].set_xlabel('Date')
ax[0].legend(loc="upper left")
ax[1].plot(df['crack' ],label="Crack Spread", color="blue")
ax[1].set_xlabel('Date')
ax[1].legend(loc="upper left")
ax[0].grid(alpha=0.6)
ax[1].grid(alpha=0.6)
fig.tight_layout()
fig.savefig('./price_crack2.png')
plt.show()


# In[9]:


#
# non-stationary - use first difference
df['dcrack'] = df.crack.diff(1)
df['dwti'] = df.wti.diff(1)


# In[10]:


#
# plot difference data
fig, ax = plt.subplots(2,1,figsize=(9,8))
ax[0].plot(df['dwti'],label="Wti Futures", color="black")
ax[0].set_xlabel('Date')
ax[0].legend(loc="upper left")
ax[1].plot(df['dcrack' ],label="Crack Spread", color="blue")
ax[1].set_xlabel('Date')
ax[1].legend(loc="upper left")
ax[0].grid(alpha=0.6)
ax[1].grid(alpha=0.6)
fig.tight_layout()
fig.savefig('./dprice_dcrack2.png')
plt.show()


# In[11]:


#
# lagged variables
df['dwti_L1'] = df.dwti.shift(1)
df['dwti_L2'] = df.dwti.shift(2)
df['dwti_L3'] = df.dwti.shift(3)
df['dwti_L4'] = df.dwti.shift(4)
df['dwti_L5'] = df.dwti.shift(5)
df['dcrack_L1'] = df.dcrack.shift(1)
df['dcrack_L2'] = df.dcrack.shift(2)
df['dcrack_L3'] = df.dcrack.shift(3)
df['dcrack_L4'] = df.dcrack.shift(4)
df['dcrack_L5'] = df.dcrack.shift(5)


# In[12]:


#
# Granger causality testing
#

#
# AR(5) VAR model for brent oil
brm = smf.ols('dwti ~ dwti_L1 + dwti_L2 + dwti_L3 + dwti_L4 + dwti_L5 + dcrack_L1 + dcrack_L2 + dcrack_L3 + dcrack_L4 + dcrack_L5', data=df).fit()
brr = brm.get_robustcov_results(cov_type='HAC', maxlags=5)
print(brr.summary())


# In[13]:


# test for Granger causality
hdbr = ['dcrack_L1 = 0', 'dcrack_L2 = 0', 'dcrack_L3 = 0', 'dcrack_L4 = 0' , 'dcrack_L5 = 0']
fdbr = brr.f_test(hdbr)
print('Testing if crack spread Granger causes Wti oil:')
print('F-stat : {}'.format(fdbr.statistic))
print('p-value: {}'.format(fdbr.pvalue))
print()


# In[14]:


#
# AR(5) VAR model for crack spread
csm = smf.ols('dcrack ~ dwti_L1 + dwti_L2 + dwti_L3 + dwti_L4 + dwti_L5 + dcrack_L1 + dcrack_L2 + dcrack_L3 + dcrack_L4 + dcrack_L5', data=df).fit()
csr = csm.get_robustcov_results(cov_type='HAC', maxlags=5)
print(csr.summary())


# In[15]:


# test for Granger causality
hdcs = ['dwti_L1 = 0', 'dwti_L2 = 0', 'dwti_L3 = 0', 'dwti_L4 = 0' , 'dwti_L5 = 0']
fdcs = csr.f_test(hdcs)
print('Testing if Wti Granger causes crack spread:')
print('F-stat : {}'.format(fdcs.statistic))
print('p-value: {}'.format(fdcs.pvalue))
print()


# In[16]:


#
# out-of-sample forecast
#

dfest = df[df.index <  '2022-10-10'].copy()
dftst = df[df.index >= '2022-10-10'].copy()
dfplt = df[df.index >= '2022-01-01'].copy()


# In[17]:


#
# AR(5)
brm = smf.ols('dwti ~ dwti_L1 + dwti_L2 + dwti_L3 + dwti_L4 + dwti_L5 + dcrack_L1 + dcrack_L2 + dcrack_L3 + dcrack_L4 + dcrack_L5', data=dfest).fit()
csm = smf.ols('dcrack ~ dwti_L1 + dwti_L2 + dwti_L3 + dwti_L4 + dwti_L5 + dcrack_L1 + dcrack_L2 + dcrack_L3 + dcrack_L4 + dcrack_L5', data=dfest).fit()


# In[18]:


#
bbeta = brm.params
cbeta = csm.params


# In[19]:


# VAR(5) model
#
dfplt['dbhat'] = 0.0
dfplt[ 'bhat'] = 0.0
dfplt['dchat'] = 0.0
dfplt[ 'chat'] = 0.0
dfplt['fcast'] = False

n = dfplt.shape[0]
db = np.zeros(n)
dc = np.zeros(n)
bt = np.zeros(n)
ct = np.zeros(n)

i = 0
for t in dfplt.index:
    if t in dfest.index:
        db[i] = dfest.loc[[t]]['dwti']
        bt[i] = dfest.loc[[t]][ 'wti']
        dc[i] = dfest.loc[[t]]['dcrack']
        ct[i] = dfest.loc[[t]][ 'crack']
    else:
        dfplt.loc[[t],['fcast']] = True

        db[i] = (bbeta[ 0] +
                 bbeta[ 1]*db[i-1] +
                 bbeta[ 2]*db[i-2] +
                 bbeta[ 3]*db[i-3] +
                 bbeta[ 4]*db[i-4] +
                 bbeta[ 5]*db[i-5] +
                 bbeta[ 6]*dc[i-1] +
                 bbeta[ 7]*dc[i-2] +
                 bbeta[ 8]*dc[i-3] +
                 bbeta[ 9]*dc[i-4] +
                 bbeta[10]*dc[i-5])
        bt[i] = bt[i-1] + db[i]
        dc[i] = (cbeta[ 0] +
                 cbeta[ 1]*db[i-1] +
                 cbeta[ 2]*db[i-2] +
                 cbeta[ 3]*db[i-3] +
                 cbeta[ 4]*db[i-4] +
                 cbeta[ 5]*db[i-5] +
                 cbeta[ 6]*dc[i-1] +
                 cbeta[ 7]*dc[i-2] +
                 cbeta[ 8]*dc[i-3] +
                 cbeta[ 9]*dc[i-4] +
                 cbeta[10]*dc[i-5])
        ct[i] = ct[i-1] + dc[i]

    #
    dfplt.loc[[t],['dbhat']] = db[i]
    dfplt.loc[[t],[ 'bhat']] = bt[i]
    dfplt.loc[[t],['dchat']] = dc[i]
    dfplt.loc[[t],[ 'chat']] = ct[i]

    #
    i += 1


# In[20]:


# RMSE of forecast
dx = dfplt[dfplt.fcast==True].copy()

dx['br_ferr'] = (dx.wti - dx.bhat)**2
dx['cr_ferr'] = (dx.crack - dx.chat)**2


print()
print('RSME Wti: {}'.format(np.sqrt(dx.br_ferr.mean())))
print('RSME Crack: {}'.format(np.sqrt(dx.cr_ferr.mean())))
print('Forecast periods: {}'.format(len(dx)))
print()


# In[21]:


#
# plot forecast data
fig, ax = plt.subplots(2,1,figsize=(9,8))
ax[0].plot(dfplt[ 'bhat'],label="Wti Forecast", color="red")
ax[0].plot(dfplt['wti'],label="Wti Observed", color="black")
ax[0].set_xlabel('Date')
ax[0].legend(loc="upper left")
ax[1].plot(dfplt[ 'chat' ],label="Crack Forecast", color="red")
ax[1].plot(dfplt['crack' ],label="Crack Spread", color="blue")
ax[1].set_xlabel('Date')
ax[1].legend(loc="upper left")
ax[0].grid(alpha=0.6)
ax[1].grid(alpha=0.6)
fig.tight_layout()
fig.savefig('./forecast_var52.png')
plt.show()


# In[ ]:




