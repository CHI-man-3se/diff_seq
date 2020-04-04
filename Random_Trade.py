import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import chain
from scipy import stats


from tqdm import tqdm

##############################################################
###            ALLデータかランダムで100サンプル数だけ取得する      ###
##############################################################

def get_rand_sample_fromALLrate(df , sample_size):

    df_len = len(df)
    rand_index = random.randint(0, df_len-sample_size)

    date = df.at[rand_index, 'DTYYYYMMDD']

    sample_set = df.iloc[rand_index:rand_index+sample_size,:]  
    
    return sample_set ,date , rand_index



##############################################################
###           rate ( candle ) の平均、分散を取得する           ###
##############################################################

def get_Statistics_sample_candle(sample_set):

    candle_std = sample_set['OPEN'].std()
    candle_mean = sample_set['OPEN'].mean()

    return candle_mean , candle_std

##############################################################
###                diff の平均、分散を取得する                 ###
##############################################################

def get_Statistics_sample_diff(sample_set):

    diff_std = sample_set['DIFF'].std()
    diff_mean = sample_set['DIFF'].mean()

    return diff_mean , diff_std


#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################


def transaction(rate_np ,openpoint, closepoint):

    openval = rate_np[openpoint]
    closeval = rate_np[closepoint]

    result = closeval - openval

    return result

###################################################################################
###################################################################################
###           　　　　　　　　            main文                                   ###
###################################################################################
###################################################################################


##############################################################
###                   Fullのサンプルブロックを作る             ###
##############################################################

#たまには絶対パス指定
# index_col = 0　としているのはもともとのCSVにすでにINDEXを持っているため
'''after 2014'''
#df = pd.read_csv('/Users/apple/python/oanda/input_USDJPY_CSV/USDJPY_10m_after2014.csv',index_col=0)
'''debug after 2014'''
#df = pd.read_csv('/Users/apple/python/oanda/input_USDJPY_CSV/for_debug__USDJPY_10m_after2014.csv',index_col=0)
'''ALL'''
df = pd.read_csv('/Users/apple/python/oanda/input_USDJPY_CSV/USDJPY_10m_DIFF.csv',index_col=0)
df_1m = pd.read_csv('/Users/apple/python/oanda/input_USDJPY_CSV/USDJPY_1m.csv',index_col=0)
df_CD = pd.read_csv('/Users/apple/python/oanda/output_classified_csv/test_variation/classified_sample_100_ABSOLUTE_DIFF_THESHOLD_std1_64.csv',index_col=0)


##############################################################
###                    パラメータ設定 　　　　　　　            ###
##############################################################

### 10m 足のサンプル数 ###
SAMPLE_SIZE = 6 * 24        #1日
Before_SAMPLE_SIZE = 6 * 24 * 5 # 5日
ans_size = 6 * 24           #1日

### 1m 足のサンプル数 ###
SAMPLE_SIZE_1m = 60 * 1        #1h
Before_SAMPLE_SIZE_1m = SAMPLE_SIZE_1m * 5 

ans_size = 60 * 1           #1h

config = 1.9  # diffが大きいか、小さいかを判定するためのしきい値の倍率

print(df_CD.describe())
print(df_CD.head())
print( df_CD['extreme_point'].value_counts() )


##############################################################
###                　　　　　　ループ 　　　　　　　            ###
###     ランダムでデータを取得し、askをしたいタイミングにマーカー   ###
##############################################################

p_transaction = 0
m_transaction = 0
z_transaction = 0

for i in range(100):

    ### 10m足のグラフ

    sample_block , date , rand_index = get_rand_sample_fromALLrate(df,SAMPLE_SIZE)
    
    RATE = df.loc[ rand_index:rand_index+SAMPLE_SIZE , 'OPEN'] # Y軸用
    rate_np = df.loc[ rand_index:rand_index+SAMPLE_SIZE , 'OPEN'].values # 最大/最小を調べる関数の引数としてnumpyにする

    result_transaction = transaction(rate_np,0,5)

    if result_transaction==0:
        z_transaction = z_transaction+1
    elif result_transaction>0:
        p_transaction = p_transaction+1
    elif result_transaction<0:
        m_transaction = m_transaction+1

   
print(z_transaction,p_transaction,m_transaction)