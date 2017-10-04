# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#!/usr/bin/python
# -*- coding: GB2312 -*-

import pandas as pd
import chardet
import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
import os
import matplotlib.pyplot as plt



# os.system("cat /Users/wyq/Desktop/final_2g_tr.csv /Users/wyq/Desktop/final_2g_te.csv > /Users/wyq/Desktop/one.csv")
'''def splitDataSet(fileName, split_size, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fr = open(fileName, 'r')  # open fileName to read
    num_line = 0
    onefile = fr.readlines()
    num_line = len(onefile)
    arr = np.arange(num_line)  # get a seq and set len=numLine
    np.random.shuffle(arr)  # generate a random seq from arr
    list_all = arr.tolist()
    each_size = (num_line + 1) / split_size  # size of each split sets
    split_all = []
    each_split = []
    count_num = 0
    count_split = 0  # count_num 统计每次遍历的当前个数
    # count_split 统计切分次数
    for i in range(len(list_all)):  # 遍历整个数字序列
        each_split.append(onefile[int(list_all[i])].strip())
        count_num += 1
        if count_num == each_size:
            count_split += 1
            array_ = np.array(each_split)
            np.savetxt(outdir + "/split_" + str(count_split) + '.txt',array_, fmt="%s", delimiter='\t')  # 输出每一份数据
            split_all.append(each_split)  # 将每一份数据加入到一个list中
            each_split = []
            count_num = 0
    return split_all
    '''
'''
def generateDataset(datadir, outdir):  # 从切分的数据集中，对其中九份抽样汇成一个,\
    # 剩余一个做为测试集,将最后的结果按照训练集和测试集输出到outdir中
    if not os.path.exists(outdir):  # if not outdir,makrdir
        os.makedirs(outdir)
    listfile = os.listdir(datadir)
    train_all = [];
    test_all = [];
    cross_now = 0
    for eachfile1 in listfile:
        train_sets = [];
        test_sets = [];
        cross_now += 1  # 记录当前的交叉次数
        for eachfile2 in listfile:
            if eachfile2 != eachfile1:  # 对其余九份欠抽样构成训练集
                one_sample = underSample(datadir + '/' + eachfile2)
                for i in range(len(one_sample)):
                    train_sets.append(one_sample[i])
        # 将训练集和测试集文件单独保存起来
        with open(outdir + "/test_" + str(cross_now) + ".datasets", 'w') as fw_test:
            with open(datadir + '/' + eachfile1, 'r') as fr_testsets:
                for each_testline in fr_testsets:
                    test_sets.append(each_testline)
            for oneline_test in test_sets:
                fw_test.write(oneline_test)  # 输出测试集
            test_all.append(test_sets)  # 保存训练集
        with open(outdir + "/train_" + str(cross_now) + ".datasets", 'w') as fw_train:
            for oneline_train in train_sets:
                oneline_train = oneline_train
                fw_train.write(oneline_train)  # 输出训练集
            train_all.append(train_sets)  # 保存训练集
    return train_all, test_all
def underSample(datafile): #只针对一个数据集的下采样
    dataMat,labelMat = loadDataSet(datafile) #加载数据
    pos_num = 0; pos_indexs = []; neg_indexs = []
    for i in range(len(labelMat)):#统计正负样本的下标
        if labelMat[i] == 1:
            pos_num +=1
            pos_indexs.append(i)
            continue
        neg_indexs.append(i)
    np.random.shuffle(neg_indexs)
    neg_indexs = neg_indexs[0:pos_num]
    fr = open(datafile, 'r')
    onefile = fr.readlines()
    outfile = []
    for i in range(pos_num):
        pos_line = onefile[pos_indexs[i]]
        outfile.append(pos_line)
        neg_line= onefile[neg_indexs[i]]
        outfile.append(neg_line)
    return outfile #输出单个数据集采样结果
'''
#filname = '/Users/wyq/Desktop/one.csv'
#dirname = '/Users/wyq/Desktop/s'
#splitDataSet(filname, 10, dirname)
#gcfilename = '/Users/wyq/Desktop/final_2g_gongcan.csv'


'''
l=[]
f = open('/Users/wyq/Desktop/s/split_1.txt')
dfile = csv.DictReader(f)
dfile.fieldnames=['', 'MRTime', 'IMSI', 'SRNCID', 'BestCellID', 'SRNTI', 'RAB', 'Delay', 'UE_TXPower', 'LCS_BIT', 'Longitude', 'Latitude', 'RNCID_1', 'CellID_1', 'EcNo_1', 'RSCP_1', 'RTT_1', 'UE_Rx_Tx_1', 'RNCID_2', 'CellID_2', 'EcNo_2', 'RSCP_2', 'RTT_2', 'UE_Rx_Tx_2', 'RNCID_3', 'CellID_3', 'EcNo_3', 'RSCP_3', 'RTT_3', 'UE_Rx_Tx_3', 'RNCID_4', 'CellID_4', 'EcNo_4', 'RSCP_4', 'RTT_4', 'UE_Rx_Tx_4', 'RNCID_5', 'CellID_5', 'EcNo_5', 'RSCP_5', 'RTT_5', 'UE_Rx_Tx_5', 'RNCID_6', 'CellID_6', 'EcNo_6', 'RSCP_6', 'RTT_6', 'UE_Rx_Tx_6', 'Grid_ID', 'Grid_center_x']
for row in dfile:
    l.append(row.get('MRTime'))
'''

'''
f = open('/Volumes/YQWANG/除了/data/final_2g_gongcan.csv')
data = f.readline()
print chardet.detect(data)

with open("/Users/wyq/Desktop/res.csv","w") as f:
    writer = csv.writer(f)
    writer.writerow(["mr_x", "mr_y","gc_x", "gc_y"])
'''

#def zxcvbnmasd(tefilename,trfilename,gcfilename):
tefilename ='/Users/wyq/Desktop/s/split_1.txt'
te = open(tefilename)
dte = csv.DictReader(te)
dte.fieldnames=['', 'MRTime', 'IMSI', 'SRNCID', 'BestCellID', 'SRNTI', 'RAB', 'Delay', 'UE_TXPower', 'LCS_BIT', 'Longitude', 'Latitude', 'RNCID_1', 'CellID_1', 'EcNo_1', 'RSCP_1', 'RTT_1', 'UE_Rx_Tx_1', 'RNCID_2', 'CellID_2', 'EcNo_2', 'RSCP_2', 'RTT_2', 'UE_Rx_Tx_2', 'RNCID_3', 'CellID_3', 'EcNo_3', 'RSCP_3', 'RTT_3', 'UE_Rx_Tx_3', 'RNCID_4', 'CellID_4', 'EcNo_4', 'RSCP_4', 'RTT_4', 'UE_Rx_Tx_4', 'RNCID_5', 'CellID_5', 'EcNo_5', 'RSCP_5', 'RTT_5', 'UE_Rx_Tx_5', 'RNCID_6', 'CellID_6', 'EcNo_6', 'RSCP_6', 'RTT_6', 'UE_Rx_Tx_6', 'Grid_ID', 'Grid_center_x']
ter_l, tec_l, telong_l, tela_l = [], [], [], []
tesignal_strenth = []
#terssi = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
ceshirs=[0,0,0,0,0,0]
ceshiec = [0, 0, 0, 0, 0, 0]
# count =0
for rowe in dte:
    ter_l.append(rowe.get('RNCID_1'))
    tec_l.append(rowe.get('CellID_1'))
    telong_l.append(rowe.get('Longitude'))
    tela_l.append(rowe.get('Latitude'))
#     count=count+1
    for i in range(1, 7):
        j = i
        i = str(i)
        ceshirs[j - 1]=(rowe.get('RSCP_' + i))
        ceshiec[j - 1] = (rowe.get('EcNo_' + i))
        #print ceshiec
    ceshirs_i=[int(x) for x in ceshirs]
    ceshiec_i = [int(x) for x in ceshiec]
    strenth = list(map(lambda x: x[0]-x[1], zip(ceshirs_i,ceshiec_i)))
    #print strenth
    tesignal_strenth.append(strenth)
    #print tesignal_strenth
ter_c = zip(ter_l, tec_l)
telong_la = zip(telong_l, tela_l)


#print tesignal_strenth

        #terssi[j - 1] = float(rowe.get('RSCP_' + i)) - float(rowe.get('EcNo_' + i))
        #tesignal_strenth.append(terssi)


trfilename ='/Users/wyq/Desktop/s/tr.txt'
tr = open(trfilename)
dtr = csv.DictReader(tr)
dtr.fieldnames=['', 'MRTime', 'IMSI', 'SRNCID', 'BestCellID', 'SRNTI', 'RAB', 'Delay', 'UE_TXPower', 'LCS_BIT', 'Longitude', 'Latitude', 'RNCID_1', 'CellID_1', 'EcNo_1', 'RSCP_1', 'RTT_1', 'UE_Rx_Tx_1', 'RNCID_2', 'CellID_2', 'EcNo_2', 'RSCP_2', 'RTT_2', 'UE_Rx_Tx_2', 'RNCID_3', 'CellID_3', 'EcNo_3', 'RSCP_3', 'RTT_3', 'UE_Rx_Tx_3', 'RNCID_4', 'CellID_4', 'EcNo_4', 'RSCP_4', 'RTT_4', 'UE_Rx_Tx_4', 'RNCID_5', 'CellID_5', 'EcNo_5', 'RSCP_5', 'RTT_5', 'UE_Rx_Tx_5', 'RNCID_6', 'CellID_6', 'EcNo_6', 'RSCP_6', 'RTT_6', 'UE_Rx_Tx_6', 'Grid_ID', 'Grid_center_x']
r_l, c_l, long_l, la_l = [], [], [], []
signal_strenth = []
qiangdu_l=[]
#rssi = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
trainrs=[0,0,0,0,0,0]
trainec = [0, 0, 0, 0, 0, 0]
count=0
for row in dtr:
    r_l.append(row.get('RNCID_1'))
    c_l.append(row.get('CellID_1'))
    long_l.append(row.get('Longitude'))
    la_l.append(row.get('Latitude'))
    for i in range(1, 7):
        j = i
        i = str(i)
        trainrs[j - 1]=(row.get('RSCP_' + i))
        trainec[j - 1] = (row.get('EcNo_' + i))
    #count=count+1
    #print trainrs,trainec
       # print trainec
    try:
        trainrs_i=[int(x) for x in trainrs]
        trainec_i = [int(x) for x in trainec]
        trstrenth = list(map(lambda x: x[0]-x[1], zip(trainrs_i,trainec_i)))
        signal_strenth.append(trstrenth)
    except ValueError:
        pass
r_c = zip(r_l, c_l)
long_la = zip(long_l, la_l)


'''
    for i in range(1, 7):
        try:
            j = i
            i = str(i)
            rssi[j - 1] = float(row.get('RSCP_' + i)) - float(row.get('EcNo_' + i))
            signal_strenth.append(rssi)
        except ValueError, e:
            print "error", e, "on line", i,row.get('RSCP_' + i)
'''



gcfilename = '/Users/wyq/Desktop/final_2g_gongcan.csv'
gc = open(gcfilename)
dgc = csv.DictReader(gc)
a = "经度"
c="纬度"
b = a.decode('utf-8').encode('gb2312')
d=c.decode('utf-8').encode('gb2312')
lac_l, ci_l, lacx_l, ciy_l = [], [], [], []
groupno, jing_wei = [], []
tegroupno, tejing_wei = [], []

for row2 in dgc:
    lac_l.append(row2.get('LAC'))
    ci_l.append(row2.get('CI'))
    lacx_l.append(row2.get(b))
    ciy_l.append(row2.get(d))
lac_ci = zip(lac_l, ci_l)
lacx_ciy = zip(lacx_l, ciy_l)
# print lac_ci
for each in r_c:
    # print each
    # print lac_ci.index(each)
    try:
        groupno.append(lac_ci.index(each))
        jing_wei.append(lacx_ciy[lac_ci.index(each)])
    except:
        groupno.append('NAN')
        jing_wei.append('NAN')

for teach in ter_c:
    try:
        tegroupno.append(lac_ci.index(teach))
        tejing_wei.append(lacx_ciy[lac_ci.index(teach)])
    except:
        tegroupno.append('NAN')
        tejing_wei.append('NAN')

distance_x, distance_y, id_l = [], [], []
for i in range(0, len(long_la)):
    if jing_wei[i] != 'NAN' and jing_wei[i][0] is not None:
        distance_x.append(float(long_la[i][0]) - float(jing_wei[i][0]))
        distance_y.append(float(long_la[i][1]) - float(jing_wei[i][1]))
        id_l.append(i)
    else:
        distance_x.append('NAN')
        distance_y.append('NAN')
        id_l.append(i)

tedistance_x, tedistance_y, teid_l = [], [], []
for m in range(0, len(telong_la)):
    if tejing_wei[m] != 'NAN' and  tejing_wei[m][0] is not None:
        tedistance_x.append(float(telong_la[m][0]) - float(tejing_wei[m][0]))
        tedistance_y.append(float(telong_la[m][1]) - float(tejing_wei[m][1]))
        teid_l.append(m)
    else :
        tedistance_x.append('NAN')
        tedistance_y.append('NAN')
        teid_l.append(m)

# distance_l = zip(distance_x,distance_y,groupno,id_l)
# todata = zip (signal_strenth,distance_x,distance_y,groupno,id_l)
# s_distance_l = sorted(distance_l,key = lambda t:t[2])
# datalabel = pd.DataFrame(distance_l, columns=['dis_X', 'dis_Y', 'groupno', 'id_intr'])
# datafeature = pd.DataFrame(signal_strenth, columns=['r_1', 'r_2', 'r_3', 'r_4','r_5','r_6'])
# matsignal_strenth = np.array(signal_strenth)
temp = np.array(signal_strenth).T
tempofte = np.array(tesignal_strenth).T
# signal_strenth= temp.tolist()
disignal_l = zip(distance_x, distance_y, groupno, id_l,temp[0], temp[1], temp[2], temp[3], temp[4], temp[5])
tedisignal_l = zip(tedistance_x, tedistance_y, tegroupno, teid_l, tempofte[0], tempofte[1], tempofte[2], tempofte[3],tempofte[4], tempofte[5])

#tempdic = {"dis_X":distance_x, "dis_Y":distance_y, 'groupno':groupno,'r_1':temp[0], 'r_2':temp[1], 'r_3':temp[2], 'r_4':temp[3], 'r_5':temp[4], 'r_6':temp[5]}
#tempdicofte = {"tdis_X":distance_x, "tdis_Y":distance_y, 'tgroupno':groupno,'e_1':tempofte[0], 'e_2':tempofte[1], 'e_3':tempofte[2], 'e_4':tempofte[3], 'e_5':tempofte[4], 'e_6':tempofte[5]}
#data = pd.DataFrame(tempdic)
#tedata = pd.DataFrame(tempdicofte)

data = pd.DataFrame(disignal_l,columns=['dis_X', 'dis_Y', 'groupno','id_','r_1', 'r_2', 'r_3', 'r_4', 'r_5', 'r_6'])
tedata = pd.DataFrame(tedisignal_l,columns=['tdis_X', 'tdis_Y', 'tgroupno','tid_',  'e_1', 'e_2', 'e_3', 'e_4', 'e_5', 'e_6'])
# categori_va =['r_1', 'r_2', 'r_3', 'r_4','r_5','r_6']
# for va in categori_va:
#    data[va].fillna('NAN',inplace = True)
# dummies = pd.get_dummies(data[va],prefix=va)
# np.any(np.isnan(data.ix[:,4:10]))
# np.all(np.isfinite(data.ix[:,4:10]))

# teandtrdata = pd.concat([tedata,data])

oushijuli = {}


def addWord(theIndex, word, pagenumber):
    theIndex.setdefault(word, []).append(pagenumber)


rfx = RandomForestRegressor(n_estimators= 100, max_depth=13, min_samples_split=110,min_samples_leaf=20,oob_score=True, random_state=40)
rfy = RandomForestRegressor(n_estimators= 100, max_depth=13, min_samples_split=110,min_samples_leaf=20,oob_score=True, random_state=40)
df = data.groupby('groupno')
dfge = tedata.groupby('tgroupno')
group_name = df.groups.keys()
# tegroup_name =dfge.groups.keys()
for key in group_name:
    if key != 'NAN':
        groupdata = df.get_group(key)
        rfx.fit(groupdata.ix[:, 4:10], groupdata.ix[:, 0])
        rfy.fit(groupdata.ix[:, 4:10], groupdata.ix[:, 1])
        tegroupdata = dfge.get_group(key)
        predict_x = rfx.predict(tegroupdata.ix[:, 4:10])
        predict_y = rfy.predict(tegroupdata.ix[:, 4:10])
        juli_series = np.power(np.power(tegroupdata.ix[:, 0] - predict_x, 2) + np.power(tegroupdata.ix[:, 1] - predict_y, 2), 0.5)
        #print juli_series
        addWord(oushijuli, key, juli_series)


#-----------------------------------------------------------------------------------------
#problem2的代码：
dicforproblem2={}
wherethegroup = []
findtheK =[]
for i in range(20,100):
    dicforproblem2[i] = []
    wherethegroup=[]
    for key in group_name:
        if key != 'NAN':
            groupdata = df.get_group(key)
            rfx.fit(groupdata.ix[:, 4:10], groupdata.ix[:, 0])
            rfy.fit(groupdata.ix[:, 4:10], groupdata.ix[:, 1])
            predict_x = rfx.predict(tedata.ix[i, 4:10])
            predict_y = rfy.predict(tedata.ix[i, 4:10])
            juli_series = np.power(np.power(tedata.ix[i, 0] - predict_x, 2) + np.power(tedata.ix[i, 1] - predict_y, 2), 0.5)
           # print juli_series
            dicforproblem2[i].append(juli_series[0])
            
           # print dicforproblem2[i]
for i in range(20,100):
    #wherethegroup.append(dicforproblem2[i].index(min(dicforproblem2[i])))
    #print wherethegroup
    findtheK.append(group_name[wherethegroup[i]])

#print findtheK››

#-----------------------------------------------------------------------------------------



listofoushijuli = []
for key in oushijuli:
    listofoushijuli.extend(list(oushijuli[key][0]))



    #for j in range(0, len(oushijuli[key][0])):
     #   listofoushijuli.append(oushijuli[key][0][j])

listofoushijuli_sort = sorted(listofoushijuli)
    #return listofoushijuli_sort

'''
for i in range(0,10):
    for j in range(2, 11):
        #if j!=1:
            j=str(j)
            os.system("cat /Users/wyq/Desktop/s/split_"+j+ ".txt >> /Users/wyq/Desktop/s/tr.txt")
    i=str(i+1)
    tename = '/Users/wyq/Desktop/s/split_'+i+'.txt'
    #print tename
    trname ='/Users/wyq/Desktop/s/tr.txt'
    print zxcvbnmasd(tename,trname,gcname)
'''


s = str(listofoushijuli_sort)
f = file('/Users/wyq/Desktop/oushijuli_sort.txt','w')
f.writelines(s)

plt.figure(1)  # 创建图表1
plt.plot(listofoushijuli_sort)

plt.annotate('中位误差为0.00042495618446043785',xy=(8237.9, 0.0004249), xytext=(0.8, 0.95))
'''
plt.plot([t, t], [0, np.sin(t)], color='red', linewidth=1, linestyle="--")
scatter([t, ], [np.sin(t), ], 50, color='red')
annotate(r'$cos(frac{2pi}{3})=-frac{1}{2}$',
         xy=(t, np.cos(t)), xycoords='data',
         xytext=(-90, -50), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
'''


# rfy = RandomForestRegressor()
# rfy.fit(df.ix[:,4:10],df.ix[:,1] )

'''
for i in range(0,len(data)):
    if data['groupno'][i]=='NAN':
        data.drop(i,axis =0)
'''
# indexs = list(data[np.isnan(data['groupno'])].index)
# data = data.drop(indexs)
# df = data[data['groupno']!='NAN']
# dfgrouped = df.groupby(groupno)
# df = data[np.isnan(data['groupno']) == False]


# ex_list = list(data.groupno)
# ex_list.remove('NAN')
# df = data[data.groupno.isin(ex_list)]
# df= data.groupby(groupno)
# df[grouped[0].transform(lambda x: x.name != group_name).astype('bool')]
# df.drop(grouped.get_group(group_name).index)

'''
matsignal_strenth = np.array(signal_strenth)
mattesignal_strenth = np.array(tesignal_strenth)

rfx = RandomForestRegressor()
rfx.fit(matsignal_strenth, distance_x)

rfy = RandomForestRegressor()
rfy.fit(matsignal_strenth, distance_y)

predict_x = rfx.predict(mattesignal_strenth)
predict_y = rfy.predict(mattesignal_strenth)

finalres_x = predict_x - distance_x
finalres_y = predict_y - distance_y

oushijuli = np.sqrt(np.power(finalres_x,2) + np.power(finalres_y,2))

s_oushijuli = sorted(oushijuli)
'''

'''
for row2 in dgc:
    if row['RNCID_1']==row2['LAC'] and row['CellID_1']==row2['CI']:
            result = open("/Users/wyq/Desktop/res.csv","wab+")
            writer = csv.writer(result)
            writer.writerow([row['Longitude'],row['Latitude'],row2['经度'],row2['纬度']])
'''

