import pandas as pd
import numpy as np

df= pd.read_csv('orders.txt',sep='\t',encoding="ISO-8859-1",parse_dates=['orderdate'])
print(df.columns)

#ltv = (margin* avg_order * freq / churn)-AC

margin=0.05
ac=1

customers=df.groupby('customerid').agg({'orderdate': lambda x : (x.max()-x.min()).days,
                                        'totalprice': lambda x : x.sum(),
                                        'orderid': lambda x:len(x)})

customers.rename(columns={"orderdate": "customer_lifetime", "totalprice": "spending_per_customer","orderid": "#oftransaction"},inplace=True)
customers=customers[customers['customer_lifetime']>0]
avg_order_value=customers['spending_per_customer'].sum()/customers['#oftransaction'].sum()
freq=customers['#oftransaction'].sum()/customers['customer_lifetime'].sum()
retention=(customers[customers['#oftransaction']>1].shape[0])/(customers.shape[0])
print(f'average order :{avg_order_value} and frequency:{freq}')

print((margin* avg_order_value * freq / (1-retention))-ac)
