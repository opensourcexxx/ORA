import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
# import torch
import os

def scaler_transform_prediction(y_train, y_test,y_pred):
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train.reshape(-1,1))
    y_train = scaler_y.transform(y_train.reshape(-1,1))
    y_test = scaler_y.transform(y_test.reshape(-1,1)).reshape(-1)
    y_pred = scaler_y.transform(y_pred.reshape(-1,1)).reshape(-1)
    return y_pred,y_test

def scaler_transform(X_train, X_test, y_train, y_test, args):
    if args['scaler'] == "stand":
        scaler_x = StandardScaler() # MinMaxScaler StandardScaler
        scaler_x.fit(X_train.to_numpy())
        X_train = scaler_x.transform(X_train.to_numpy())
        X_test = scaler_x.transform(X_test.to_numpy())
        scaler_y = StandardScaler()
        scaler_y.fit(y_train.reshape(-1,1))
        y_train = scaler_y.transform(y_train.reshape(-1,1))
        y_test = scaler_y.transform(y_test.reshape(-1,1))
    elif args['scaler'] == "minmax":
        scaler_x = MinMaxScaler() # MinMaxScaler StandardScaler
        scaler_x.fit(X_train.to_numpy())
        X_train = scaler_x.transform(X_train.to_numpy())
        X_test = scaler_x.transform(X_test.to_numpy())
        scaler_y = MinMaxScaler()
        scaler_y.fit(y_train.reshape(-1,1))
        y_train = scaler_y.transform(y_train.reshape(-1,1))
        y_test = scaler_y.transform(y_test.reshape(-1,1))
    elif args['scaler'] == "none":
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        scaler_y = 0
    return X_train, X_test, y_train, y_test, scaler_y

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)

def get_features_name(args):
    if args['dataset'] in ['bd','sk','wm']:
        cols_meta = ["id_user","timelimit","mem_req","cpus_req","nodes_alloc",'time_submit'] # time_submit
        if args['data_mode'] == 'runtime':
            cols_meta = cols_meta +["time_start"]
    else:
        cols_meta = ['id_user','ReqWallTime','RequestedMemory','Partition','TaskReq','time_submit']
        if args['data_mode'] == 'runtime':
            cols_meta = cols_meta +['WaitTime','TaskCount','CPUTime','UsedMEM','Status']
    return cols_meta

def fix_times(df):
    UnixStartTime = 1231135224
    df['SubmitTime'] = df['SubmitTime'] + UnixStartTime
    df['StartTime']  = df['SubmitTime'] + df['WaitTime']
    #df['SubmitTime'] = pd.to_datetime(df['SubmitTime'], unit='s')
    #df['StartTime'] = pd.to_datetime(df['StartTime'], unit='s')
    return df

def get_ea(pred,gt): # 要求 pred >0  gt>0
    pred = np.abs(pred)
    pred_high = np.where(pred>=gt,True,False)
    gt_high = np.where(gt>pred,True,False)
    acc1 = gt[pred_high]/(pred[pred_high] +1e-7)
    acc2 = pred[gt_high]/(gt[gt_high]+1e-7)
    acc = (acc1.sum() + acc2.sum())/(len(pred)*1.0)
    return acc

def get_ur(pred,gt): # 要求 pred >0  gt>0
    pred = np.abs(pred)
    pred_high = np.where(pred>=gt,True,False)
    # gt_high = np.where(gt>=pred,True,False)
    return pred_high.mean()


def standard(df,columns_to_scale=None):
    if columns_to_scale is not None:
        for column in columns_to_scale:
            scaler_min =  float(df[column].min())
            scaler_max =  float(df[column].max())
            df[column] = df[column].apply(lambda row:round((row-scaler_min)/(scaler_max-scaler_min+1e-7),4))
    else:
        for column in df.select_dtypes(include='number').columns:
            scaler_min =  float(df[column].min())
            scaler_max =  float(df[column].max())
            df[column] = df[column].apply(lambda row:round((row-scaler_min)/(scaler_max-scaler_min+1e-7),4))
    return df

def get_initial_df(dataset):
    filename =  "data/" + dataset
    print(filename)
    df = pd.read_csv(filename,
                     sep="\s+|\t+|\s+\t+|\t+\s+", comment=';',
                     header = None,
                     names=['JobId', "time_submit", 'WaitTime', 'RunTime','TaskCount', 'CPUTime',
                            'UsedMEM', 'TaskReq',  'ReqWallTime', 'RequestedMemory',
                            'Status', 'id_user', 'Group', 'Exe', 'Class', 'Partition',
                            'prejob', 'thinktime'],
                     engine='python')
    #print(df.memory_usage(index=True, deep=False))
    return df

def get_id_user_int(raw):
    try:
        return raw['id_user'].astype('int')
    except Exception as e:
        return 0

def get_tt_data(args):
    if args['dataset'] not in ["bd","sk","wm"]:
        df = get_initial_df(args['dataset'])
        # Drop fields
        df["time_end"] = df["RunTime"] + df["time_submit"] + df["WaitTime"]
        df["label"] = df["RunTime"]
        # columns_to_drop = ['JobId','Group','Exe','Class','Partition','prejob','thinktime','RunTime','WaitTime','TaskCount','CPUTime','UsedMEM','Status'] # runtime feature and useless feature
        # columns_to_drop = ['RunTime']
        # columns_to_drop = ['RunTime','UsedMEM','WaitTime','CPUTime','Status',] # runtime feature
        # # cols_meta = ['id_user', "time_submit","ReqWallTime","TaskReq","RequestedMemory"]
        # df = df.drop(columns_to_drop, axis=1)
        # df['User'] = df['User'].astype('category')
        # df['Status'] = df['Status'].astype('category')
    else:
        df = pd.read_csv(f'data/{args["dataset"]}.csv')
        df["label"] = df["duration"]
        
        # Metadata
        # id_job,job_db_inx,duration,time_submit,time_start,time_end,timelimit,cpus_req,mem_req,nodes_alloc,tres_req,gres_used,partition,work_dir,batch_script,id_user
        # cols = ['id_user', 'cpus_req','nodes_alloc', 'timelimit', 'time_submit', 'sub_year', 'sub_quarter', 'sub_month', 
        #         'sub_day', 'sub_hour', 'sub_day_of_year', 'sub_day_of_month', 'sub_day_of_week', 'top1_time', 'top2_time', 'top2_mean', 'duration'] #'id_qos', 
        # columns = ['tres_req','partition','work_dir','batch_script','time_start']
        # columns_to_drop = ['duration','gres_used','id_job','job_db_inx','mem_req']
        df['id_user'] = df.apply(get_id_user_int,axis=1)
        # df = df.drop(columns_to_drop, axis=1)

    mid_df = pd.DataFrame()
    if args['data_partition'] == "random":
        # columns_to_scale = ["SubmitTime", 'WaitTime', 'RunTime','TaskCount', 'CPUTime',
        #                 'UsedMEM', 'TaskReq',  'ReqWallTime', 'RequestedMemory',]
        # df = standard(df,columns_to_scale)
        y = df.pop('label').values
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=args['test_data_size'], random_state=42)
    elif args['data_partition'] == "before_after":
        # columns_to_scale = ["SubmitTime", 'WaitTime', 'RunTime','TaskCount', 'CPUTime',
        #                 'UsedMEM', 'TaskReq',  'ReqWallTime', 'RequestedMemory',]
        # df = standard(df,columns_to_scale)
        y = df.pop('label').values
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=args['test_data_size'], shuffle=False)
        
        # random
        X_train["label"] = y_train
        X_train = X_train.sample(frac=1).reset_index(drop=True)
        y_train = X_train["label"].to_numpy()
        X_train = X_train.drop(["label"],axis=1)
        
    elif args['data_partition'] == "real":
        NUM_TRAIN_SAMPLES = int(len(df)*(1-args['test_data_size']))
        NUM_TEST_SAMPLES = int(len(df)*args['test_data_size']/2)

        def split_train_and_test(df):
            train_df = df[:NUM_TRAIN_SAMPLES]
            cover_count = 0
            end_time = train_df.iloc[NUM_TRAIN_SAMPLES-1]["time_end"]
            for i in range(len(df[NUM_TRAIN_SAMPLES:])):
                if df.iloc[NUM_TRAIN_SAMPLES+i]["time_submit"] < end_time:
                    continue
                else:
                    cover_count = i
                    break
            test_df = df[NUM_TRAIN_SAMPLES+cover_count:NUM_TRAIN_SAMPLES+cover_count+NUM_TEST_SAMPLES]
            mid_df = df[NUM_TRAIN_SAMPLES:NUM_TRAIN_SAMPLES+cover_count]
            return train_df,test_df,mid_df
        
        train_df, test_df, mid_df = split_train_and_test(df)
        
        # random
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        
        # columns_to_scale = ["SubmitTime", 'WaitTime', 'RunTime','TaskCount', 'CPUTime',
                        # 'UsedMEM', 'TaskReq',  'ReqWallTime', 'RequestedMemory']
        # train_df = standard(train_df,columns_to_scale)
        # test_df = standard(test_df,columns_to_scale)
        
        X_train = train_df.drop(["label"],axis=1)
        y_train = train_df["label"].to_numpy()
        X_test = test_df.drop(["label"],axis=1)
        y_test = test_df["label"].to_numpy()

    return X_train, X_test, y_train, y_test, mid_df


def split_dates(df, column):
    hour_of_day = range(0, 24)
    day_of_week = range(0,7)

    if column == "SubmitTime":
        df['SubmitTime_TS'] = df['SubmitTime'].copy()
        df['SubmitTime'] = pd.to_datetime(df['SubmitTime'], unit='s')
        df['Weekday_SubmitTime'] = pd.Categorical(pd.Series(pd.DatetimeIndex(pd.to_datetime(df[column],unit='s')).weekday, dtype="category") , categories=day_of_week)
        df['Hour_SubmitTime'] = pd.Categorical(pd.Series(pd.DatetimeIndex(pd.to_datetime(df[column],unit='s')).hour, dtype="category") , categories=hour_of_day)
    if column == "StartTime":
        df['StartTime_TS'] = df['StartTime'].copy()
        df['StartTime'] = pd.to_datetime(df['StartTime'], unit='s')
        df['Weekday_StartTime'] = pd.Categorical(pd.Series(pd.DatetimeIndex(pd.to_datetime(df[column],unit='s')).weekday, dtype="category") , categories=day_of_week)
        df['Hour_StartTime'] = pd.Categorical(pd.Series(pd.DatetimeIndex(pd.to_datetime(df[column],unit='s')).hour, dtype="category") , categories=hour_of_day)
    return df
