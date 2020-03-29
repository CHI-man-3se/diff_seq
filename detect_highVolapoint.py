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




##############################################################
###           分散が大きいindex番号だけをとってくる              ###
##############################################################

def get_highVolatility_index(sample_set ,under_Threshold, over_Threshold):

    ## highVolaの検出なしで、すべてのindexを持ってきたいときdropアルゴリズムによって終わる
    ## なので、debug用としてあらたなdropアルゴリズムのパスをつくる
    debug_ALL = 0
    if under_Threshold==over_Threshold:
        debug_ALL = 1
    else:
        debug_ALL = 0

    # high_volaのdataframeをゲット　※ほしいのはindexだけなので、それだけ貰えればいいかも
    high_volatility = sample_set.query('OPEN <= @under_Threshold or @over_Threshold <= OPEN')

    sample_len = len(high_volatility)
    
    index_num = high_volatility.index.values
    #open_rate = high_volatility.loc[:,'OPEN']

    open_rate = []
    drop_sequence_index = []
    extreme_point = []

    # 分散がでかくなった瞬間を持ってきたいので、連続indexが連続になっているのは省く
    # indexが連続しているのをdropさせるloop
    if debug_ALL == 0:    
        if sample_len == 0 :
            None
        else:
            
            if 0<=index_num[0]<=6 : ## high vola検出のindexが6以下のときはeach blocksを形成できないためここもパスする
                None
            else:
                drop_sequence_index = [index_num[0]] ## SAMPLE BLOCK100毎のhigh vola INDEXの先頭をreturn用のlistに追加する

            for i in range(sample_len):

                # listがオーバフローしないために
                if i == sample_len-1:
                    None
                elif 0<=index_num[i]<=1 : ## high vola検出のindexが6以下のときはeach blocksをけいせいできないためここもパスする
                    None
                else :  ##連続している、indexを省く
                    is_sequence = index_num[i+1] - index_num[i]

                    # 連続しているindexは省き、分散がでかくなった瞬間のindexを持ってくる
                    if is_sequence < 6:
                        None
                    else:
                        drop_sequence_index.append(index_num[i+1])
    elif debug_ALL == 1:
        index_debug = np.arange(6, 97, 12)
        for i in index_debug:
            drop_sequence_index.append(index_num[i])
    else:
        None

    for i in drop_sequence_index:
        rate = high_volatility.loc[i,'OPEN']
        open_rate.append(rate)

        if rate < under_Threshold:
            extreme_point.append('UNDER')
        elif rate > over_Threshold:
            extreme_point.append('OVER')
        else:
            extreme_point.append('UNEXPETED')

    return drop_sequence_index , open_rate , extreme_point



##############################################################
###           diffの連続性を確認するための関数                ###
###           4連続だったら、Seqフラグをセットする               ###
###           引数は、np array 　　　　　　　　                ###
###           返り値は、before afterの連続数　                ###
##############################################################

def get_Sequencial(diff_block):
    
    len_diff_block = len(diff_block)
    
    cnt_before_p = 0
    cnt_before_m = 0
    cnt_after_p = 0
    cnt_after_m = 0
    f_before_p = 0
    f_before_m = 0
    f_after_p = 0
    f_after_m = 0

    for i in range(len_diff_block):
        if i <= 5:
            ##### before #####
            if diff_block[i] >= 0:
                if f_before_p == 1:
                    cnt_before_p = cnt_before_p + 1
                else :
                    #cnt_before_p = 0
                    None
                f_before_p = 1
                f_before_m = 0

            else :
                if f_before_m == 1:
                    cnt_before_m = cnt_before_m + 1
                else :
                    #cnt_before_m = 0
                    None
                
                f_before_m = 1
                f_before_p = 0

        else:
            ##### after #####
            if diff_block[i] >= 0:
                if f_after_p == 1:
                    cnt_after_p = cnt_after_p + 1
                else :
                    cnt_after_p = 0
                
                f_after_p = 1
                f_after_m = 0

            else :
                if f_after_m == 1:
                    cnt_after_m = cnt_after_m + 1
                else :
                    cnt_after_m = 0
                
                f_after_m = 1
                f_after_p = 0

    return cnt_before_p, cnt_before_m, cnt_after_p, cnt_after_m


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


def drow_Graph_Term(rate_np,size,highVola_index_list,min_rate , max_rate,rate_before_np):


    plt.subplot(2,1,1)

    # diffが大きいポイントにマーカーを付ける用のループ
    for i in highVola_index_list:

        marker_y = rate_np[i]
        marker_x = i
        #print("b")
        plt.plot(marker_x, marker_y,marker='o', markersize=7,color='red')


    index_num = np.arange(0, size+1)  # X軸用
    plt.plot(index_num, rate_np)       #折れ線グラフ
    #plt.bar(index_num, rate_np)         #棒グラフ
    plt.ylim([min_rate,max_rate])
    plt.title('Random')

    plt.subplot(2,1,2)
    before_size = len(rate_before_np)
    index_num_before = np.arange(0, before_size)  # X軸用
    plt.plot(index_num_before, rate_before_np) #折れ線グラフ
    #plt.bar(index_num_before, rate_before_np)   #棒グラフ

    marker_x_before = before_size - size
    marker_y_before = rate_before_np[marker_x_before-1]

    plt.plot(marker_x_before, marker_y_before,marker='o', markersize=7)

    #plt.ylim([min_rate,max_rate])
    plt.title('Before')

    plt.show()

    return min,max

def drow_Graph_Ans(df,rand_index,samplesize,ans_size,min,max,d_open, Flag,d_index):
    
    #endpoint = rand_index+samplesize+ans_size
    endpoint = d_index+ans_size

    RATE = df.loc[ rand_index:endpoint , 'OPEN'] # Y軸用
    
    sample_endmarker_index = rand_index+samplesize
    sample_endmarker = df.at[ sample_endmarker_index , 'OPEN']

    
    index_num = np.arange(rand_index, endpoint +1)  # X軸用



    plt.plot(index_num, RATE)
    plt.ylim([min,max])
    plt.title('ANS')
    
    plt.plot( sample_endmarker_index,sample_endmarker,marker='o', markersize=10)

    plt.show()


def minmax_serch( open_np ):

    max = 0
    min = 0

    for i in range( len(open_np) ):
        if(i == 0):
            max = open_np[i]
            min = open_np[i]
        else:
            if( open_np[i] > max):
                max = open_np[i]
            elif( open_np[i] < min):
                min = open_np[i]
            else:
                None

    return min , max

def get_diff_Theshold(rate_np_long):
    
    diff_open = np.diff(rate_np_long)    # diffをとる

    diff_abs = np.abs(diff_open)        # diffの大きさを確認したいので絶対値を取る
    diff_var_abs = np.var(diff_abs)     # 絶対値の分散
    diff_mean_abs = np.mean(diff_abs)   # 絶対値の平均

    return diff_mean_abs

def highvola_serch(open_np,config,diff_mean_abs):
    
    diff_open = np.diff(open_np)    # diffをとる

    diff_var = np.var(diff_open)
    diff_mean = np.mean(diff_open)

    #diffの絶対値の統計値
    #diff_abs = np.abs(diff_open)
    #diff_var_abs = np.var(diff_abs)
    #diff_mean_abs = np.mean(diff_abs)

    # しきい値の設定
    diff_Th_over = diff_mean + diff_mean_abs*config
    diff_Th_under = diff_mean - diff_mean_abs*config

    """
    plt.hist(diff_open, bins=100)
    plt.axvline(x=diff_mean, color='red')
    plt.axvline(x=diff_Th_over, color='red')
    plt.axvline(x=diff_Th_under, color='red')
    plt.show()
    """

    index = 0
    
    d_index_list = []
    for i in diff_open:
        index = index+1
     
        # diffが下に、しきい値を超えた     
        if i < diff_Th_under:
            d_index_list.append( index )
    
        # diffが上にに、しきい値を超えた     
        if diff_Th_over < i:
            d_index_list.append( index )

    return d_index_list


def Seq_highvola_serch(open_np,config,diff_mean_abs):
    #diffの統計値
    diff_open = np.diff(open_np)
    diff_var = np.var(diff_open)
    diff_mean = np.mean(diff_open)

    #diffの絶対値の統計値
    #diff_abs = np.abs(diff_open)
    #diff_var_abs = np.var(diff_abs)
    #diff_mean_abs = np.mean(diff_abs)
    
    # しきい値の設定
    diff_Th_over = diff_mean + diff_mean_abs*config
    diff_Th_under = diff_mean - diff_mean_abs*config

    """
    plt.hist(diff_open, bins=100)
    plt.axvline(x=diff_mean, color='red')
    plt.axvline(x=diff_Th_over, color='red')
    plt.axvline(x=diff_Th_under, color='red')
    plt.show()
    """

    index = 0
    seq_f_u = 0
    seq_f_o = 0
    
    d_index_list = []
    for i in diff_open:
        index = index+1
        
        """
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        111bounceではなく、Traceオーダーのポイントを検出するため、
        1111今は2連続で、大きいdiffがきたときをトリガとしている。
        2回めが同符号で小さいdiffのときはどうなるのだろうか？これもTraceオーダーのポイントになるのではないか？

        5min足をつくったほうがいいんじゃない？？
        """

    
        # diffが下に、しきい値を超えた     
        if i < diff_Th_under:
            # diffの大きさが連続でしきい値を超えた
            if seq_f_u == 1:
                d_index_list.append( index )
            seq_f_u = 1
        
        else:
            seq_f_u = 0
    
        # diffが上にに、しきい値を超えた     
        if diff_Th_over < i:
            # diffの大きさが連続でしきい値を超えた
            if seq_f_o == 1:
                d_index_list.append( index )
            seq_f_o = 1
        
        else:
            seq_f_o = 0

    return d_index_list


def Seq_Loose_highvola_serch(open_np,config,diff_mean_abs):
    #diffの統計値
    diff_open = np.diff(open_np)
    diff_var = np.var(diff_open)
    diff_mean = np.mean(diff_open)

    #diffの絶対値の統計値
    #diff_abs = np.abs(diff_open)
    #diff_var_abs = np.var(diff_abs)
    #diff_mean_abs = np.mean(diff_abs)
    
    Loose_Rate = 0.2

    # しきい値の設定
    diff_Th_over_Loose = diff_mean + diff_mean_abs*Loose_Rate
    diff_Th_under_Loose = diff_mean - diff_mean_abs*Loose_Rate

    # 2回目のLooseしきい値
    diff_Th_over = diff_mean + diff_mean_abs*config
    diff_Th_under = diff_mean - diff_mean_abs*config

    """
    plt.hist(diff_open, bins=100)
    plt.axvline(x=diff_mean, color='red')
    plt.axvline(x=diff_Th_over, color='red')
    plt.axvline(x=diff_Th_under, color='red')
    plt.show()
    """

    index = 0
    seq_f_u = 0
    seq_f_o = 0
    
    d_index_list = []
    for i in diff_open:
        index = index+1
        
        """
        2回めが同符号で小さいdiffのときはどうなるのだろうか？これもTraceオーダーのポイントになるのではないか？
        反発さえしなければ問題ない
        """

        # diffが下に、しきい値を超えた     
        # diffの大きさが連続でしきい値を超えた
        if seq_f_u == 1:
            if i < diff_Th_over_Loose:          #2回目は反発じゃないかを確認するために、diffがちょいプラスまでは許容範囲とする
                d_index_list.append( index )

        if i < diff_Th_under:
            seq_f_u = 1        
        else:
            seq_f_u = 0

    
        if seq_f_o == 1:
            if diff_Th_under_Loose < i:          #2回目は反発じゃないかを確認するために、diffがちょいマイナスまでは許容範囲とする
                d_index_list.append( index )
        
        # diffが上に、しきい値を超えた     
        if diff_Th_over < i:
            seq_f_o = 1
        else:
            seq_f_o = 0

    return d_index_list


def detect_minmax(df, min, max, rand_index, SAMPLE_SIZE):

    i = 0
    Flag = 0
    while True:
        i = i+1
        tmp_open = df.at[rand_index+SAMPLE_SIZE+i,'OPEN']

        d_index = rand_index+SAMPLE_SIZE+i

        if tmp_open >= max:
            d_max = tmp_open
            Flag = 0
            return d_max , d_index , Flag

        elif tmp_open <= min:
            d_min = tmp_open
            Flag = 1
            return d_min , d_index ,Flag

    

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

for i in range(100):

    ### 10m足のグラフ
    '''
    sample_block , date , rand_index = get_rand_sample_fromALLrate(df,SAMPLE_SIZE)
    
    RATE = df.loc[ rand_index:rand_index+SAMPLE_SIZE , 'OPEN'] # Y軸用
    rate_np = df.loc[ rand_index:rand_index+SAMPLE_SIZE , 'OPEN'].values # 最大/最小を調べる関数の引数としてnumpyにする

    min_rate , max_rate = minmax_serch(rate_np)
    highVola_index_list = highvola_serch(rate_np)
    Seq_highVola_index_list = Seq_highvola_serch(rate_np)

    rate_before_np = df.loc[ rand_index-Before_SAMPLE_SIZE:rand_index+SAMPLE_SIZE , 'OPEN'].values # 最大/最小を調べる関数の引数としてnumpyにする


    drow_Graph_Term(rate_np, SAMPLE_SIZE,highVola_index_list,min_rate , max_rate,rate_before_np)        # レートのグラフを作成し後にどう動くのか予想する
    drow_Graph_Term(rate_np, SAMPLE_SIZE,Seq_highVola_index_list,min_rate , max_rate,rate_before_np)        # レートのグラフを作成し後にどう動くのか予想する
    '''

    ### 1m足のグラフ
    
    sample_block , date , rand_index = get_rand_sample_fromALLrate(df_1m,SAMPLE_SIZE_1m)
    
    RATE = df_1m.loc[ rand_index:rand_index+SAMPLE_SIZE_1m , 'OPEN'] # Y軸用
    rate_np = df_1m.loc[ rand_index:rand_index+SAMPLE_SIZE_1m , 'OPEN'].values # 最大/最小を調べる関数の引数としてnumpyにする

    # diffが大きいかどうか判定するためのしきい値は、少し長めの区間をもとにして判別する
    rate_np_long = df_1m.loc[ rand_index-Before_SAMPLE_SIZE_1m:rand_index+SAMPLE_SIZE_1m , 'OPEN'].values # 最大/最小を調べる関数の引数としてnumpyにする

    diff_Th_abs = get_diff_Theshold(rate_np_long)

    min_rate , max_rate = minmax_serch(rate_np)
    highVola_index_list = highvola_serch(rate_np,config,diff_Th_abs)
    Seq_double_highVola_index_list = Seq_highvola_serch(rate_np,config,diff_Th_abs)
    Seq_Loose_highVola_index_list = Seq_Loose_highvola_serch(rate_np,config,diff_Th_abs)

    rate_before_np = df_1m.loc[ rand_index-Before_SAMPLE_SIZE_1m:rand_index+SAMPLE_SIZE_1m , 'OPEN'].values # 最大/最小を調べる関数の引数としてnumpyにする


    drow_Graph_Term(rate_np, SAMPLE_SIZE_1m,highVola_index_list,min_rate , max_rate,rate_before_np)        # レートのグラフを作成し後にどう動くのか予想する
    drow_Graph_Term(rate_np, SAMPLE_SIZE_1m,Seq_Loose_highVola_index_list,min_rate , max_rate,rate_before_np)        # レートのグラフを作成し後にどう動くのか予想する
    

    
   
