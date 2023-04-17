import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_order= pd.read_csv('orders.txt',sep='\t',encoding="ISO-8859-1",parse_dates=['orderdate'])
df_customer= pd.read_csv('customer.txt',sep='\t',encoding="ISO-8859-1")

#print(df_order.dtypes)
#print(df_customer.dtypes)

df_order=df_order[['orderid', 'customerid','orderdate','totalprice']]
df = df_order.merge(df_customer[['customerid','householdid']], left_on='customerid',right_on='customerid')

#print(df.groupby('householdid').agg({'orderid':lambda x: len(x)}))

df_1=df.groupby('householdid')['orderdate'].max().reset_index()
df_1.columns=[['householdid','max_date']]
df_1['Recency']=(df_1['max_date'].max()-df_1['max_date']).apply(lambda x : x.dt.days)
#print(df_1)

sse={}
from sklearn.cluster import KMeans
'''
for n in range(1,10):
    kmeans=KMeans(n_clusters=n,random_state=0).fit(df_1['Recency'].to_numpy())
    df_1['cluster']=kmeans.labels_
    sse[n]=kmeans.inertia_
print(df_1)
print(sse.items())

plt.plot(sse.keys(),sse.values())
plt.show()
'''
kmeans=KMeans(n_clusters=4,random_state=0).fit(df_1['Recency'].to_numpy())
df_1['recency_cluster']=kmeans.predict(df_1['Recency'].to_numpy())

df_1.columns=df_1.columns.get_level_values(0)

#function for ordering cluster numbers

def order_cluster(cluster_field_name,target_field_name,df,ascending):
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index']=df_new.index
    df_final=pd.merge(df,df_new[[cluster_field_name,'index']],on=cluster_field_name)
    df_final=df_final.drop([cluster_field_name],axis=1)
    df_final=df_final.rename(columns={'index':cluster_field_name})
    return df_final

df_ordered_by_recency=order_cluster('recency_cluster','Recency',df_1,False)
print(df_ordered_by_recency)