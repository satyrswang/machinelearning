/**
 * [main description]
 * @Author   WYQ
 * @DateTime 2017-10-07T20:48:53+0800
 * @return   {[type]}                 [description]
 */

import pandas as pd

def map_extract(element):
    file_path, content = element
    year = file_path[-8:-4]
    return [(year, i) for i in content.split("\r\n") if i]

res = sc.wholeTextFiles('hdfs://10.21.208.21:8020/user/mercury/names', 
                        minPartitions=40)  \
        .map(map_extract) \
        .flatMap(lambda x: x) \
        .map(lambda x: (x[0], int(x[1].split(',')[2]))) \
        .reduceByKey(operator.add) \
        .collect()

data = pd.DataFrame.from_records(res, columns=['year', 'birth'])\
         .sort(columns=['year'], ascending=True)
ax = data.plot(x=['year'], y=['birth'], 
                figsize=(20, 6), 
                title='US Baby Birth Data from 1897 to 2014', 
                linewidth=3)
ax.set_axis_bgcolor('white')
ax.grid(color='gray', alpha=0.2, axis='y')
