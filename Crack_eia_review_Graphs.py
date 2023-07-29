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


# create new variables
df[ 'lcrack'] = np.log(df['crack'])
df['lwti' ] = np.log(df['wti'])
df['lgas' ] = np.log(df['gasoline'])
df['lheat' ] = np.log(df['heatoil'])


# In[4]:


#
# non-stationary - use first difference
df['dcrack'] = df.crack.diff(1)
df['dlcrack'] = df.lcrack.diff(1)
df['dwti'] = df.wti.diff(1)
df['dlwti'] = df.lwti.diff(1)


# In[5]:



# --------------------------------------------------------------------
# -- plot graph of raw data for WTI Oil                             --
# --------------------------------------------------------------------

fig, ax = plt.subplots(2,1,figsize=(12,8))
fig.suptitle('Historical price of WTI oil, 2005-2023', fontsize=18)
df['wti'].plot(ax=ax[0],c='k')
ax[0].set_ylabel('WTI oil')
ax[0].grid()
ax[0].set_xlabel('Date')

df['dwti'].plot(ax=ax[1],c='k')
ax[1].set_ylabel('Change in price of WTI oil (%)')
ax[1].grid()
ax[1].set_xlabel('Date')
fig.tight_layout()
plt.show()


# In[6]:



# --------------------------------------------------------------------
# -- plot graph of log of WTI oil price                             --
# --------------------------------------------------------------------

fig, ax = plt.subplots(2,1,figsize=(12,8))
fig.suptitle('log of historical price of WTI oil, 2005-2023', fontsize=18)
df['lwti'].plot(ax=ax[0],c='k')
ax[0].set_ylabel('log of WTI oil')
ax[0].grid()
ax[0].set_xlabel('Date')

df['dlwti'].plot(ax=ax[1],c='k')
ax[1].set_ylabel('Change in price of log of WTI oil (%)')
ax[1].grid()
ax[1].set_xlabel('Date')
fig.tight_layout()
plt.show()


# In[7]:



# --------------------------------------------------------------------
# -- plot graph of raw data for Crack spread                        --
# --------------------------------------------------------------------

fig, ax = plt.subplots(2,1,figsize=(12,8))
fig.suptitle('Historical price of Crack spread, 2005-2023', fontsize=18)
df['crack'].plot(ax=ax[0],c='blue')
ax[0].set_ylabel('Crack spread')
ax[0].grid()
ax[0].set_xlabel('Date')

df['dcrack'].plot(ax=ax[1],c='blue')
ax[1].set_ylabel('Change in Crack spread (%)')
ax[1].grid()
ax[1].set_xlabel('Date')
fig.tight_layout()
plt.show()


# In[8]:



# --------------------------------------------------------------------
# -- plot graph of log of Crack spread                              --
# --------------------------------------------------------------------

fig, ax = plt.subplots(2,1,figsize=(12,8))
fig.suptitle('log of historical price of Crack spread, 2005-2023', fontsize=18)
df['lcrack'].plot(ax=ax[0],c='blue')
ax[0].set_ylabel('log of Crack spread')
ax[0].grid()
ax[0].set_xlabel('Date')

df['dlcrack'].plot(ax=ax[1],c='blue')
ax[1].set_ylabel('Change in log of Crack spread (%)')
ax[1].grid()
ax[1].set_xlabel('Date')
fig.tight_layout()
plt.show()


# In[10]:



# --------------------------------------------------------------------
# -- plot graph of raw data for Crack spread and WTI Crude oil      --
# --------------------------------------------------------------------

fig, ax = plt.subplots(2,1,figsize=(12,8))
fig.suptitle('Historical price of Crack spread and WTI crude oil, 2005-2023', fontsize=18)
df['wti'].plot(ax=ax[0],c='black')
ax[0].set_ylabel('WTI crude oil')
ax[0].grid()
ax[0].set_xlabel('Date')

df['crack'].plot(ax=ax[1],c='blue')
ax[1].set_ylabel('Crack')
ax[1].grid()
ax[1].set_xlabel('Date')
fig.tight_layout()
plt.show()


# In[ ]:




