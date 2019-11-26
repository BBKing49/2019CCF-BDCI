#!/usr/bin/env python
# coding: utf-8
import math
import numpy as np
import pandas as pd
import os 
from tqdm import tqdm
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


train_sales = pd.read_csv('data/train2_dataset/train_sales_data.csv')
train_search = pd.read_csv('data/train2_dataset/train_search_data.csv')
train_user = pd.read_csv('data/train2_dataset/train_user_reply_data.csv')
evaluation_public = pd.read_csv('data/train2_dataset/evaluation_public.csv')
#
train_sales = train_sales.sort_values(['regYear','regMonth','model'])
evaluation_public = evaluation_public.sort_values(['regYear','regMonth','model'])

# 消除异常值
high_sales = train_sales.salesVolume.quantile(0.9)
low_sales = train_sales.salesVolume.quantile(0.1)
print('high_sales:',high_sales,'low_sales:',low_sales)
for m in train_sales.model.unique():
    for pro in train_sales.adcode.unique():
        tiaojian = (train_sales.model==m)&(train_sales.adcode==pro)
        index = train_sales[tiaojian][(train_sales.salesVolume>high_sales)|(train_sales.salesVolume<low_sales)].index
        index = index.tolist()
        if len(index)==0:
            continue
        train_sales.loc[tiaojian,'roll_mean'] = train_sales['salesVolume'][tiaojian].rolling(window=5).mean().round().astype('float')
        for i in index:
            if  train_sales.loc[i,'regMonth']<5:
                continue
            train_sales.loc[i,'salesVolume'] = train_sales.loc[i,'roll_mean']
        del train_sales['roll_mean']
        
high_sales = train_sales.salesVolume.quantile(0.9)
low_sales = train_sales.salesVolume.quantile(0.1)        
print('high_sales:',high_sales,'low_sales:',low_sales)  

data = pd.concat([train_sales, evaluation_public], ignore_index=True)
data = data.merge(train_search, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
data = data.merge(train_user, 'left', on=['model', 'regYear', 'regMonth'])
data['label'] = data['salesVolume']
data['id'] = data['id'].fillna(0).astype(int)
data['bodyType'] = data['model'].map(train_sales.drop_duplicates('model').set_index('model')['bodyType']) #填充bodytype

#LabelEncoder
for i in ['bodyType', 'model']:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))

data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']



# 利用规则填充popularity
m1_1 = data.loc[(data.regYear==2017)&(data.regMonth==1) , 'popularity'].values / data.loc[(data.regYear==2016)&(data.regMonth==1), 'popularity'].values
m1_2 = data.loc[(data.regYear==2017)&(data.regMonth==2) , 'popularity'].values / data.loc[(data.regYear==2016)&(data.regMonth==2), 'popularity'].values
m1_3 = data.loc[(data.regYear==2017)&(data.regMonth==3) , 'popularity'].values / data.loc[(data.regYear==2016)&(data.regMonth==3), 'popularity'].values
m1_4 = data.loc[(data.regYear==2017)&(data.regMonth==4) , 'popularity'].values / data.loc[(data.regYear==2016)&(data.regMonth==4), 'popularity'].values

data.loc[(data.regYear==2018)&(data.regMonth==1) , 'popularity'] = np.round( data.loc[(data.regYear==2017)&(data.regMonth==1) , 'popularity'].values * m1_1*1.02 )
data.loc[(data.regYear==2018)&(data.regMonth==2) , 'popularity'] = np.round( data.loc[(data.regYear==2017)&(data.regMonth==2) , 'popularity'].values * m1_2*0.95 )
data.loc[(data.regYear==2018)&(data.regMonth==3) , 'popularity'] = np.round( data.loc[(data.regYear==2017)&(data.regMonth==3) , 'popularity'].values * m1_3*1.02)
data.loc[(data.regYear==2018)&(data.regMonth==4) , 'popularity'] = np.round( data.loc[(data.regYear==2017)&(data.regMonth==4) , 'popularity'].values * m1_4*1.02 )

del m1_1,m1_2,m1_3,m1_4

# 构造历史特征
def get_stat_feature(df_):   
    df = df_.copy()
    stat_feat = []
    df['model_adcode'] = df['adcode'] + df['model']   # 地区车型特征
    df['model_adcode_mt'] = df['model_adcode'] * 100 + df['mt']  #新的标签，地区车型时间
    for col in tqdm(['label','popularity']):   # 分别从label，popularity获取col
        # shift
        for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
            stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col,i))
            df['model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'] + i  #每个月要获取的历时平移数据的月份
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,i)) #设置索引
            df['shift_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])    # df_last[col]视作一个二维表，历史数据匹配填充  
    return df,stat_feat


# # 构造销量相关特征
def get_count_feature(df):
    temp = df.copy()
    count_feat = ['adcode_model_sum','adcode_model_mean','adcode_model_median','adcode_model_std',
             'model_sum',  'model_mean', 'model_median', 'model_std','model_occupy']
    temp['adcode_model_sum'] = temp.groupby(['model','adcode'])['label'].transform('sum')
    temp['adcode_model_mean'] = temp.groupby(['model','adcode'])['label'].transform('mean')
    temp['adcode_model_median'] = temp.groupby(['model','adcode'])['label'].transform('median')
    temp['adcode_model_std'] = temp.groupby(['model','adcode'])['label'].transform('std')
    temp['model_sum'] = temp.groupby(['model'])['label'].transform('sum')
    temp['model_mean'] = temp.groupby(['model'])['label'].transform('mean')
    temp['model_median'] = temp.groupby(['model'])['label'].transform('median')
    temp['model_std'] = temp.groupby(['model'])['label'].transform('std')
    
    # 每个model在每个省的占比
    temp['adcode_sum'] = temp.groupby(['adcode'])['label'].transform('sum')
    temp['model_occupy'] = temp['adcode_model_sum']/temp['adcode_sum']
    return temp,count_feat

# 获取这个车型过去12/6/3月的销量总/平均/方差
def get_sales_features(data,startmonth,endmonth,feature,list_feature):
    m = endmonth-startmonth

    if startmonth > endmonth:
        return 0
    temp = data[(data.mt>=startmonth)&(data.mt<endmonth)]
    salesVolume_sum = temp['salesVolume'].groupby(temp[feature]).sum().reset_index(drop=True)
    salesVolume_mean = temp['salesVolume'].groupby(temp[feature]).mean().reset_index(drop=True)
    salesVolume_median = temp['salesVolume'].groupby(temp[feature]).median().reset_index(drop=True)
    salesVolume_std = temp['salesVolume'].groupby(temp[feature]).std().reset_index(drop=True)

    sales_feature = pd.DataFrame({feature:list_feature,'mt':endmonth,feature+'_salesVolume_sum'+str(m):salesVolume_sum,
                                  feature+'_salesVolume_mean'+str(m):salesVolume_mean,feature+'_salesVolume_median'+str(m):salesVolume_median,
                                  feature+'_salesVolume_std'+str(m):salesVolume_std,
                                  })
    return sales_feature

# 融合
def merge_sales_features(data,feature,start,end):
    list_feature = data[feature].drop_duplicates().sort_values().reset_index(drop=True)

    final_sales_feature_12 = pd.DataFrame()
    final_sales_feature_6 = pd.DataFrame()
    final_sales_feature_3 = pd.DataFrame()
    final_sales_feature_1 = pd.DataFrame()
    for i in  range(start,end):
        sales_feature_12 = get_sales_features(data,i-12,i,feature,list_feature)
        sales_feature_6 = get_sales_features(data,i-6,i,feature,list_feature)
        sales_feature_3 = get_sales_features(data,i-3,i,feature,list_feature)
        sales_feature_1 = get_sales_features(data,i-1,i,feature,list_feature)

        final_sales_feature_12  = pd.concat([final_sales_feature_12,sales_feature_12], ignore_index=True)
        final_sales_feature_6  = pd.concat([final_sales_feature_6,sales_feature_6], ignore_index=True)
        final_sales_feature_3  = pd.concat([final_sales_feature_3,sales_feature_3], ignore_index=True)
        final_sales_feature_1  = pd.concat([final_sales_feature_1,sales_feature_1], ignore_index=True)

    del sales_feature_3
    data = data.merge(final_sales_feature_12,'left',on=['mt',feature])
    data = data.merge(final_sales_feature_6,'left',on=['mt',feature])
    data = data.merge(final_sales_feature_3,'left',on=['mt',feature])
    data = data.merge(final_sales_feature_1,'left',on=['mt',feature])
    del final_sales_feature_12,final_sales_feature_3,final_sales_feature_6

    return data

# # 趋势特征
def get_trend_feature(df):
    data = df.copy()
    trend_feat = []   

    for gap in range(0,6):
        trend_feat.append('add_{}_month'.format(gap))
        trend_feat.append('muti_{}_month'.format(gap))
        trend_feat.append('rate_{}_month'.format(gap))
        for i in range(1,29):           
            if (i-gap-2)>0:
                data.loc[data.mt==i,'add_{}_month'.format(gap)] = data.loc[data.mt==i-1,'label'].values-data.loc[data.mt==i-2-gap,'label'].values
                data.loc[data.mt==i,'muti_{}_month'.format(gap)] = data.loc[data.mt==i-1,'label'].values/data.loc[data.mt==i-2-gap,'label'].values
                data.loc[data.mt==i,'rate_{}_month'.format(gap)] = (data.loc[data.mt==i-1,'label'].values-data.loc[data.mt==i-2-gap,'label'].values)/data.loc[data.mt==i-2-gap,'label'].values    
    
    for gap in range(0,6):
        for gap2 in range(0,6):

            trend_feat.append('add_{}_month_{}_2'.format(gap2,gap))
            trend_feat.append('muti_{}_month_{}_2'.format(gap2,gap))
            trend_feat.append('rate_{}_month_{}_2'.format(gap2,gap))

            for i in range(1,29):
                if i-gap-2>0:
                    data.loc[data.mt==i,'add_{}_month_{}_2'.format(gap2,gap)] = data.loc[data.mt==i-1,'add_{}_month'.format(gap2)].values-data.loc[data.mt==i-2-gap,'add_{}_month'.format(gap2)].values
                    data.loc[data.mt==i,'muti_{}_month_{}_2'.format(gap2,gap)] = data.loc[data.mt==i-1,'muti_{}_month'.format(gap2)].values-data.loc[data.mt==i-2-gap,'muti_{}_month'.format(gap2)].values
                    data.loc[data.mt==i,'rate_{}_month_{}_2'.format(gap2,gap)] = data.loc[data.mt==i-1,'rate_{}_month'.format(gap2)].values-data.loc[data.mt==i-2-gap,'rate_{}_month'.format(gap2)].values
                    
    return data,trend_feat


# # 构造销量历史相关特征

def get_count_his_feature(df):
    temp = df.copy()
    count_his_feat = []
#    4,5,6,7,8,9,10,11,12
    for i in [3,4,5,6,12]:
        temp['last_{}_month_sum'.format(i)] = np.nan
        temp['last_{}_month_mean'.format(i)] = np.nan
        temp['last_{}_month_median'.format(i)] = np.nan
        temp['last_{}_month_std'.format(i)] = np.nan
        temp['last_{}_month_occupy'.format(i)] = np.nan
        count_his_feat.append('last_{}_month_sum'.format(i))
        count_his_feat.append('last_{}_month_mean'.format(i))
        count_his_feat.append('last_{}_month_median'.format(i))
        count_his_feat.append('last_{}_month_std'.format(i))
        count_his_feat.append('last_{}_month_occupy'.format(i))
    
        for mt in range(1,29):
            
            temp_month = temp[temp.mt.between(mt-i,mt)]
            temp_month['last_{}_month_sum'.format(i)] = temp_month.groupby(['model','province'])['label'].transform('sum')
            temp_month['last_{}_month_mean'.format(i)] = temp_month.groupby(['model','province'])['label'].transform('mean')
            temp_month['last_{}_month_median'.format(i)] = temp_month.groupby(['model','province'])['label'].transform('median')
            temp_month['last_{}_month_std'.format(i)] = temp_month.groupby(['model','province'])['label'].transform('std')
            temp['last_{}_month_occupy'.format(i)] = temp_month.groupby(['model','province'])['label'].transform('sum')/temp_month.groupby(['province'])['label'].transform('sum')
            
            if mt-i>0:
                temp_month2 = temp_month[temp_month.mt == mt][['model','adcode','mt','last_{}_month_sum'.format(i)]]
                temp_month3 = temp_month[temp_month.mt == mt][['model','adcode','mt','last_{}_month_mean'.format(i)]]
                temp_month4 = temp_month[temp_month.mt == mt][['model','adcode','mt','last_{}_month_median'.format(i)]]
                temp_month5 = temp_month[temp_month.mt == mt][['model','adcode','mt','last_{}_month_std'.format(i)]]
                temp_month6 = temp_month[temp_month.mt == mt][['model','adcode','mt','last_{}_month_occupy'.format(i)]]
               
                temp['last_{}_month_sum'.format(i)][temp.mt == mt] = temp_month2['last_{}_month_sum'.format(i)]
                temp['last_{}_month_mean'.format(i)][temp.mt == mt] = temp_month3['last_{}_month_mean'.format(i)]
                temp['last_{}_month_median'.format(i)][temp.mt == mt] = temp_month4['last_{}_month_median'.format(i)]
                temp['last_{}_month_std'.format(i)][temp.mt == mt] = temp_month5['last_{}_month_std'.format(i)]
                temp['last_{}_month_occupy'.format(i)][temp.mt == mt] = temp_month6['last_{}_month_occupy'.format(i)]

    return temp,count_his_feat


# # 构建滑窗特征
def cal_windows_fea(df:pd.DataFrame, cal_col:str, stat_dim:list) -> pd.DataFrame:
    """
    计算滑窗特征
    """
    train_sales_data = df.copy()

    name_prefix = "_".join(stat_dim) + "_%s"%cal_col

    # 滑窗特征
    ## 均值
    feature_data = train_sales_data.groupby(stat_dim)[cal_col].apply(lambda x: x.sum()).groupby(stat_dim[:-1]).rolling(3).mean()
    feature_data = feature_data.dropna().unstack(level=-1)

    if len(stat_dim) == 3:
        feature_data.index = feature_data.index.droplevel(0)
        feature_data.index = feature_data.index.droplevel(0)
    elif len(stat_dim) == 2:
        feature_data.index = feature_data.index.droplevel(0)

    feature_data.reset_index(inplace=True)
    feature_data = feature_data.rename(columns={k:"%s_rolling_mean_%d"%(name_prefix, k) for k in range(13)})

    ## std
    tmp_df = train_sales_data.groupby(stat_dim)[cal_col].apply(lambda x: x.sum()).groupby(stat_dim[:-1]).rolling(3).std()
    tmp_df = tmp_df.dropna().unstack(level=-1)

    if len(stat_dim) == 3:
        tmp_df.index = tmp_df.index.droplevel(0)
        tmp_df.index = tmp_df.index.droplevel(0)
    elif len(stat_dim) == 2:
        tmp_df.index = tmp_df.index.droplevel(0)

    tmp_df.reset_index(inplace=True)
    tmp_df = tmp_df.rename(columns={k:"%s_rolling_std_%d"%(name_prefix, k) for k in range(13)})

    feature_data = pd.merge(feature_data, tmp_df, on=stat_dim[:-1], how="left")

    ## sum
    tmp_df = train_sales_data.groupby(stat_dim)[cal_col].apply(lambda x: x.sum()).groupby(stat_dim[:-1]).rolling(3).sum()
    tmp_df = tmp_df.dropna().unstack(level=-1)

    if len(stat_dim) == 3:
        tmp_df.index = tmp_df.index.droplevel(0)
        tmp_df.index = tmp_df.index.droplevel(0)
    elif len(stat_dim) == 2:
        tmp_df.index = tmp_df.index.droplevel(0)

    tmp_df.reset_index(inplace=True)
    tmp_df = tmp_df.rename(columns={k:"%s_rolling_sum_%d"%(name_prefix, k) for k in range(13)})

    feature_data = pd.merge(feature_data, tmp_df, on=stat_dim[:-1], how="left")
    return feature_data


def merge_rolling(data):
    model2type = data[["model", "bodyType"]].drop_duplicates().set_index("model").to_dict()["bodyType"]
    train_windows_fea = cal_windows_fea(data, cal_col="salesVolume", stat_dim=["adcode", "model", "regMonth"])
    # 城市
    tmp_df = cal_windows_fea(data, "label", ["adcode", "regMonth"],)
    train_windows_fea = pd.merge(train_windows_fea, tmp_df, on="adcode", how="left")
    # 车
    tmp_df = cal_windows_fea(data, "label", ["model", "regMonth"])
    train_windows_fea = pd.merge(train_windows_fea, tmp_df, on="model", how="left")
    # 城市+车型
    tmp_df = cal_windows_fea(data, "label", ["adcode", "bodyType", "regMonth"])
    train_windows_fea["bodyType"] = train_windows_fea.model.apply(lambda x: model2type[x])
    train_windows_fea = pd.merge(train_windows_fea, tmp_df, on=["adcode", "bodyType"], how="left")

    data = pd.merge(data,train_windows_fea,on = ['adcode','model','bodyType'],how = 'left')
    windows_feat = train_windows_fea.columns.drop(['adcode','model','bodyType']).tolist()
    
    return data,windows_feat


# # 评价指标
def score(data, pred='pred_label', label='label', group='model'):
    data['pred_label'] = data['pred_label'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    data_agg = data.groupby('model').agg({
        pred:  list,
        label: [list, 'mean']
    }).reset_index()
    data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns]
    nrmse_score = []
    for raw in data_agg[['{0}_list'.format(pred), '{0}_list'.format(label), '{0}_mean'.format(label)]].values:
        nrmse_score.append(
            mse(raw[0], raw[1]) ** 0.5 / raw[2]
        )
    print(1 - np.mean(nrmse_score))
    return 1 - np.mean(nrmse_score)	


# # 模型选择
def get_model_type(train_x,train_y,valid_x,valid_y,m_type,cate_feat):   
    if m_type == 'lgb':
        model = lgb.LGBMRegressor(
                                num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='mse',
                                max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=2019,
                                n_estimators=1500, subsample=0.9, colsample_bytree=0.7,
                                )
        model.fit(train_x, train_y, 
              eval_set=[(train_x, train_y),(valid_x, valid_y)], 
              categorical_feature=cate_feat, 
              early_stopping_rounds=100, verbose=100)      
    elif m_type == 'xgb':
        model = xgb.XGBRegressor(
                                max_depth=5 , learning_rate=0.05, n_estimators=2000, 
                                objective='reg:gamma', tree_method = 'hist',subsample=0.9, 
                                colsample_bytree=0.7, min_child_samples=5,eval_metric = 'rmse' 
                                )
        model.fit(train_x, train_y, 
              eval_set=[(train_x, train_y),(valid_x, valid_y)], 
              early_stopping_rounds=100, verbose=100)   
    return model


# # 模型训练
def get_train_model(df_, st,m, m_type,feature,cate_feat):
    df = df_.copy()
    # 数据集划分
#    st = 13
    df['label'] =np.log1p(df['label'])
    all_idx   = (df['mt'].between(st , m-1))     # between 是一个闭区间
    train_idx = (df['mt'].between(st , st+11))
    valid_idx = (df['mt'].between(st+12, st+12))
    test_idx  = (df['mt'].between(m  , m  ))
    print('all_idx  :',st ,m-1)
    print('train_idx:',st , st+11)
    print('valid_idx:',st+12 ,st+12)
    print('test_idx :',m  ,m  )  
    # 最终确认
    train_x = df[train_idx][feature]
    train_y = df[train_idx]['label']
    valid_x = df[valid_idx][feature]
    valid_y = df[valid_idx]['label']   
    # get model
    model = get_model_type(train_x,train_y,valid_x,valid_y,m_type,cate_feat)  
    # offline
    df['pred_label'] = model.predict(df[feature])
    df['pred_label'] = np.power(math.e ,df['pred_label'])-1 
 
    # online
    if m_type == 'lgb':
        model.n_estimators = model.best_iteration_ + 100
        model.fit(df[all_idx][feature], df[all_idx]['label'], categorical_feature=cate_feat)
    elif m_type == 'xgb':
        model.n_estimators = model.best_iteration + 100
        model.fit(df[all_idx][feature], df[all_idx]['label'])
    df['forecastVolum'] = model.predict(df[feature]) 
    df['forecastVolum'] = np.power(math.e ,df['forecastVolum'])-1
    df['label'] = np.power(math.e ,df['label'])
    score(df[valid_idx])
    print('valid mean:',df[valid_idx]['pred_label'].mean())
    print('true  mean:',df[valid_idx]['label'].mean())
    print('test  mean:',df[test_idx]['forecastVolum'].mean())
    # 阶段结果
    sub = df[test_idx][['id']]
    sub['forecastVolum'] = (df[test_idx]['forecastVolum']).apply(lambda x: 0 if x < 0 else x).round().astype(int)  
    return sub,df[valid_idx]['pred_label']


# # 逐步预测

def get_sub(df, model_num, st,m_type):
    
        for month in [25, 26, 27, 28]:
    
            if model_num == 1:
                data_df, stat_feat = get_stat_feature(df)
#                data_df,count_feat = get_count_feature(data_df)
                data_df = merge_sales_features(data_df, 'model', 1, month + 1)
                data_df = merge_sales_features(data_df, 'bodyType', 1, month + 1)
                data_df, count_his_feat = get_count_his_feature(data_df)   
            else:
                data_df, stat_feat = get_stat_feature(df)
#                data_df,count_feat = get_count_feature(data_df)
                data_df = merge_sales_features(data_df, 'model', 1, month + 1)
                data_df = merge_sales_features(data_df, 'bodyType', 1, month + 1)
                data_df, count_his_feat = get_count_his_feature(data_df)
                data_df, trend_feat = get_trend_feature(data_df)
                data_df, windows_feat = merge_rolling(data_df)
    
            tp_feat = data_df.columns
            num_feat = tp_feat.drop(['label', 'mt', 'id', 'forecastVolum', 'salesVolume', 'adcode', 'bodyType', 'model', 'regMonth',
                     'province']).tolist()
                #    num_feat = ['regYear'] + stat_feat + count_feat + count_his_feat + windows_feat+trend_feat
            cate_feat = ['adcode', 'bodyType', 'model', 'regMonth']
    
            if m_type == 'lgb':
                    for i in cate_feat:
                        data_df[i] = data_df[i].astype('category')
            elif m_type == 'xgb':
                    lbl = LabelEncoder()
                    for i in tqdm(cate_feat):
                        data_df[i] = lbl.fit_transform(data_df[i].astype(str))
    
            features = num_feat + cate_feat
            print(len(features), len(set(features)))
         
                    
            sub, val_pred = get_train_model(data_df, st, month, m_type, features, cate_feat)
            df.loc[(df.regMonth == (month - 24)) & (df.regYear == 2018) , 'salesVolume'] = sub['forecastVolum'].values
            df.loc[(df.regMonth == (month - 24)) & (df.regYear == 2018) , 'label'] = sub['forecastVolum'].values

            st = st + 1

        sub = df.loc[(df.regMonth >= 1) & (df.regYear == 2018), ['id', 'model', 'adcode', 'regMonth', 'salesVolume']]
        sub.columns = ['id', 'model', 'adcode', 'regMonth', 'forecastVolum']

        return sub



sub1_xgb = get_sub(data,1,5,'xgb')
sub1_lgb = get_sub(data,2,5,'lgb')
sub1 = pd.DataFrame({'id':sub1_lgb.id,'model':sub1_lgb.model,'regMonth':sub1_lgb.regMonth})
sub1['forecastVolum'] = np.round(sub1_xgb.forecastVolum*0.3 + sub1_lgb.forecastVolum*0.7)

print( sub1.forecastVolum.groupby(sub1.regMonth).sum() )
print(sub1['forecastVolum'].mean())
pd.DataFrame(sub1.forecastVolum.groupby(sub1.regMonth).sum()).plot()

sub2_xgb = get_sub(data,1,9,'xgb')
sub2_lgb = get_sub(data,2,9,'lgb')
sub2 = pd.DataFrame({'id':sub2_lgb.id,'model':sub2_lgb.model,'regMonth':sub2_lgb.regMonth})
sub2['forecastVolum'] = np.round(sub2_xgb.forecastVolum*0.3 + sub2_lgb.forecastVolum*0.7)

print( sub2.forecastVolum.groupby(sub2.regMonth).sum() )
print(sub2['forecastVolum'].mean())
pd.DataFrame(sub2.forecastVolum.groupby(sub2.regMonth).sum()).plot()


sub = pd.DataFrame({'id':sub1.id,'model':sub1.model,'regMonth':sub1.regMonth})
sub['forecastVolum'] = np.round( sub1.forecastVolum*0.4 + sub2.forecastVolum*0.6)



pd.DataFrame(sub.forecastVolum.groupby(sub.regMonth).sum()).plot()
print( sub.forecastVolum.groupby(sub.regMonth).sum() )
print(sub['forecastVolum'].mean())

sub = sub.sort_values(['id'])

sub = sub[['id','forecastVolum']].astype(int)
sub.to_csv('CCF_sales.csv',index =None)
