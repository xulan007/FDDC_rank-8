
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


def Preprocess_3Sheets():


	df_bs=pd.read_excel('../data/[Add June July] FDDC_financial_data_20180711/[New] Financial Data_20180711/Balance Sheet.xls',sheet_name=[0,1,2,3],parse_dates=['END_DATE'])
	df_is=pd.read_excel('../data/[Add June July] FDDC_financial_data_20180711/[New] Financial Data_20180711/Income Statement.xls',sheet_name=[0,1,2,3],parse_dates=['END_DATE'])
	df_cf=pd.read_excel('../data/[Add June July] FDDC_financial_data_20180711/[New] Financial Data_20180711/Cashflow Statement.xls',sheet_name=[0,1,2,3],parse_dates=['END_DATE'])


	# In[4]:


	print('Before concat the shapes are:{} {} {}'.format(df_bs[1].shape,df_bs[2].shape,df_bs[3].shape))
	print('After concat the shapes are:{}\n'.format(pd.concat(df_bs,axis=0).shape))

	print('Before concat the shapes are:{} {} {}'.format(df_is[1].shape,df_is[2].shape,df_is[3].shape))
	print('After concat the shapes are:{}\n'.format(pd.concat(df_is,axis=0).shape))

	print('Before concat the shapes are:{} {} {}'.format(df_cf[1].shape,df_cf[2].shape,df_cf[3].shape))
	print('After concat the shapes are:{}'.format(pd.concat(df_cf,axis=0).shape))


	# In[5]:


	df_bs=pd.concat(df_bs,axis=0)
	df_is=pd.concat(df_is,axis=0)
	df_cf=pd.concat(df_cf,axis=0)


	# In[6]:


	df_group_bs=df_bs.sort_values('PUBLISH_DATE').groupby(
		[df_bs['TICKER_SYMBOL'],df_bs['END_DATE']]).apply(lambda x: x.iloc[-1])

	df_group_is=df_is.sort_values('PUBLISH_DATE').groupby(
		[df_is['TICKER_SYMBOL'],df_is['END_DATE']]).apply(lambda x: x.iloc[-1])

	df_group_cf=df_cf.sort_values('PUBLISH_DATE').groupby(
		[df_cf['TICKER_SYMBOL'],df_cf['END_DATE']]).apply(lambda x: x.iloc[-1])


	# In[7]:


	df_map=pd.read_csv('../data/maket_value_type_map.csv').set_index("TICKER_SYMBOL")


	# ##给每只股票添加股票

	# In[8]:


	temp=[]
	for i in tqdm(df_group_bs["TICKER_SYMBOL"]):
		try:
		    temp.append(df_map.loc[i]["TYPE_NAME_CN"])
		except:
		    temp.append(None)
	df_group_bs["type"]=temp
	temp=[]
	for i in tqdm(df_group_is["TICKER_SYMBOL"]):
		try:
		    temp.append(df_map.loc[i]["TYPE_NAME_CN"])
		except:
		    temp.append(None)
	df_group_is["type"]=temp
	temp=[]
	for i in tqdm(df_group_cf["TICKER_SYMBOL"]):
		try:
		    temp.append(df_map.loc[i]["TYPE_NAME_CN"])
		except:
		    temp.append(None)
	df_group_cf["type"]=temp


	# ##对重要特征进行填充缺失值防止缺失值过滤

	# In[9]:


	df_group_bs["CB_BORR"]=df_group_bs["CB_BORR"].fillna(0)
	df_group_bs["R_D"]=df_group_bs["R_D"].fillna(0)
	df_group_bs["BOND_PAYABLE"]=df_group_bs["BOND_PAYABLE"].fillna(0)
	df_group_bs["GOODWILL"]=df_group_bs["GOODWILL"].fillna(0)
	df_group_bs["TRADING_FA"]=df_group_bs["TRADING_FA"].fillna(0)
	df_group_bs["CIP"]=df_group_bs["CIP"].fillna(0)
	df_group_bs["LT_RECEIV"]=df_group_bs["LT_RECEIV"].fillna(0)
	df_group_bs["INT_PAYABLE"]=df_group_bs["INT_PAYABLE"].fillna(0)
	df_group_bs["TAXES_PAYABLE"]=df_group_bs["TAXES_PAYABLE"].fillna(0)
	df_group_bs["LT_BORR"]=df_group_bs["LT_BORR"].fillna(0)
	df_group_bs["PREMIUM_RECEIV"]=df_group_bs["PREMIUM_RECEIV"].fillna(0)  ##应收保费
	df_group_bs["AR"]=df_group_bs["AR"].fillna(0)                          ##应收账款


	# In[10]:


	null_rate_bs=df_group_bs.isnull().sum()/len(df_group_bs)
	null_rate_is=df_group_is.isnull().sum()/len(df_group_is)
	null_rate_cf=df_group_cf.isnull().sum()/len(df_group_cf)


	# In[11]:


	df_group_bs=df_group_bs.drop(null_rate_bs[null_rate_bs>0.80].index,axis=1)
	df_group_is=df_group_is.drop(null_rate_is[null_rate_is>0.80].index,axis=1)
	df_group_cf=df_group_cf.drop(null_rate_cf[null_rate_cf>0.80].index,axis=1)


	# In[12]:


	print('After drop high null_rate columns the shape of bs are:{}\n'.format(df_group_bs.shape))
	print('After drop high null_rate columns the shape of is are:{}\n'.format(df_group_is.shape))
	print('After drop high null_rate columns the shape of cf are:{}\n'.format(df_group_cf.shape))


	# In[13]:


	###只选择三表都全的公司

	ticker_set=(set(df_bs["TICKER_SYMBOL"].unique()) & set(df_is["TICKER_SYMBOL"].unique())
		       ) & set(df_cf["TICKER_SYMBOL"].unique())
	print("there are {} corporations with full sheets".format(len(ticker_set)))


	# In[14]:


	def extract(df_each,time_list,TICKER_SYMBOL):
		df_temp=pd.DataFrame()
		for col in df_each.columns:
		    for i,time in enumerate(time_list):
		        try:
		            df_temp[col+'_'+str(i)]=[df_each.loc[time,col]]
		        except:
		            df_temp[col+'_'+str(i)]=[np.nan]
		
		df_temp.index=[(6-len(TICKER_SYMBOL))*'0'+TICKER_SYMBOL]
		return df_temp


	# In[15]:


	### 错位时间 DataAugmentation
	train_time0=['2006-06-30','2006-09-30','2006-12-31','2007-03-31','2007-06-30','2007-09-30','2007-12-31',
		         '2008-03-31','2008-06-30','2008-09-30','2008-12-31','2009-03-31']

	train_time1=['2007-06-30','2007-09-30','2007-12-31','2008-03-31','2008-06-30','2008-09-30','2008-12-31',
		         '2009-03-31','2009-06-30','2009-09-30','2009-12-31','2010-03-31']

	train_time2=['2008-06-30','2008-09-30','2008-12-31','2009-03-31','2009-06-30','2009-09-30','2009-12-31',
		         '2010-03-31','2010-06-30','2010-09-30','2010-12-31','2011-03-31']

	train_time3=['2009-06-30','2009-09-30','2009-12-31','2010-03-31','2010-06-30','2010-09-30','2010-12-31',
		         '2011-03-31','2011-06-30','2011-09-30','2011-12-31','2012-03-31']

	train_time4=['2010-06-30','2010-09-30','2010-12-31','2011-03-31','2011-06-30','2011-09-30','2011-12-31',
		         '2012-03-31','2012-06-30','2012-09-30','2012-12-31','2013-03-31']

	train_time5=['2011-06-30','2011-09-30','2011-12-31','2012-03-31','2012-06-30','2012-09-30','2012-12-31',
		         '2013-03-31','2013-06-30','2013-09-30','2013-12-31','2014-03-31']

	train_time6=['2012-06-30','2012-09-30','2012-12-31','2013-03-31','2013-06-30','2013-09-30','2013-12-31',
		         '2014-03-31','2014-06-30','2014-09-30','2014-12-31','2015-03-31']

	train_time7=['2013-06-30','2013-09-30','2013-12-31','2014-03-31','2014-06-30','2014-09-30','2014-12-31',
		         '2015-03-31','2015-06-30','2015-09-30','2015-12-31',"2016-03-31"]

	train_time8=['2014-06-30','2014-09-30','2014-12-31','2015-03-31','2015-06-30','2015-09-30','2015-12-31',
		         "2016-03-31",'2016-06-30',"2016-09-30","2016-12-31","2017-03-31"]

	test_time=['2015-06-30','2015-09-30','2015-12-31',"2016-03-31",'2016-06-30',"2016-09-30","2016-12-31",
		          "2017-03-31","2017-06-30","2017-09-30","2017-12-31","2018-03-31"]


	# In[ ]:


	##初始化参数用于盛放数据
	print("extracting features! please be patient it will take some time.....")
	all_test=[]
	train0=[]
	train1=[]
	train2=[]
	train3=[]
	train4=[]
	train5=[]
	train6=[]
	train7=[]
	train8=[]

	for ticker in tqdm(ticker_set):
		df_ticker_bs=df_group_bs.loc[ticker,:].drop(["PARTY_ID","TICKER_SYMBOL","EXCHANGE_CD",'PUBLISH_DATE',
		                                 'END_DATE_REP','END_DATE','FISCAL_PERIOD','REPORT_TYPE',
		                                  "MERGED_FLAG",],axis=1)
		df_ticker_is=df_group_is.loc[ticker,:].drop(["PARTY_ID","TICKER_SYMBOL","EXCHANGE_CD",'PUBLISH_DATE',
		                                 'END_DATE_REP','END_DATE','FISCAL_PERIOD','REPORT_TYPE',
		                                  "MERGED_FLAG",],axis=1)
		df_ticker_cf=df_group_cf.loc[ticker,:].drop(["PARTY_ID","TICKER_SYMBOL","EXCHANGE_CD",'PUBLISH_DATE',
		                                 'END_DATE_REP','END_DATE','FISCAL_PERIOD','REPORT_TYPE',
		                                  "MERGED_FLAG",],axis=1)
		
		###提取测试集特征
		feature_bs=extract(df_ticker_bs,test_time,str(ticker))
		feature_is=extract(df_ticker_is,test_time,str(ticker))
		feature_cf=extract(df_ticker_cf,test_time,str(ticker))
		all_test.append(pd.concat([feature_bs,feature_is,feature_cf],axis=1))
		
		###提取2009年训练集特征
		feature_bs=extract(df_ticker_bs,train_time0,str(ticker))
		feature_is=extract(df_ticker_is,train_time0,str(ticker))
		feature_cf=extract(df_ticker_cf,train_time0,str(ticker))
		train0.append(pd.concat([feature_bs,feature_is,feature_cf],axis=1))
		
		###提取2010年训练集特征
		feature_bs=extract(df_ticker_bs,train_time1,str(ticker))
		feature_is=extract(df_ticker_is,train_time1,str(ticker))
		feature_cf=extract(df_ticker_cf,train_time1,str(ticker))
		train1.append(pd.concat([feature_bs,feature_is,feature_cf],axis=1))
		
		###提取2011年训练集特征
		feature_bs=extract(df_ticker_bs,train_time2,str(ticker))
		feature_is=extract(df_ticker_is,train_time2,str(ticker))
		feature_cf=extract(df_ticker_cf,train_time2,str(ticker))
		train2.append(pd.concat([feature_bs,feature_is,feature_cf],axis=1))
		
		###提取2012年训练集特征
		feature_bs=extract(df_ticker_bs,train_time3,str(ticker))
		feature_is=extract(df_ticker_is,train_time3,str(ticker))
		feature_cf=extract(df_ticker_cf,train_time3,str(ticker))
		train3.append(pd.concat([feature_bs,feature_is,feature_cf],axis=1))
		
		###提取2013年训练集特征
		feature_bs=extract(df_ticker_bs,train_time4,str(ticker))
		feature_is=extract(df_ticker_is,train_time4,str(ticker))
		feature_cf=extract(df_ticker_cf,train_time4,str(ticker))
		train4.append(pd.concat([feature_bs,feature_is,feature_cf],axis=1))
		
		###提取2014年训练集特征
		feature_bs=extract(df_ticker_bs,train_time5,str(ticker))
		feature_is=extract(df_ticker_is,train_time5,str(ticker))
		feature_cf=extract(df_ticker_cf,train_time5,str(ticker))
		train5.append(pd.concat([feature_bs,feature_is,feature_cf],axis=1))
		
		###提取2015年训练集特征
		feature_bs=extract(df_ticker_bs,train_time6,str(ticker))
		feature_is=extract(df_ticker_is,train_time6,str(ticker))
		feature_cf=extract(df_ticker_cf,train_time6,str(ticker))
		train6.append(pd.concat([feature_bs,feature_is,feature_cf],axis=1))
		
		###提取2016年训练集特征
		feature_bs=extract(df_ticker_bs,train_time7,str(ticker))
		feature_is=extract(df_ticker_is,train_time7,str(ticker))
		feature_cf=extract(df_ticker_cf,train_time7,str(ticker))
		train7.append(pd.concat([feature_bs,feature_is,feature_cf],axis=1))
		
		###提取2017年训练集特征
		feature_bs=extract(df_ticker_bs,train_time8,str(ticker))
		feature_is=extract(df_ticker_is,train_time8,str(ticker))
		feature_cf=extract(df_ticker_cf,train_time8,str(ticker))
		train8.append(pd.concat([feature_bs,feature_is,feature_cf],axis=1))


	# In[86]:


	sumbit=pd.read_csv('../data/FDDC_financial_submit_20180524.csv',header=None)

	### 提取需要预测的股票代码
	sumbit.index=sumbit[0].apply(lambda x: x.split('.')[0])
	###抽取需要预测的列
	X_test=pd.concat(all_test,axis=0)
	intersection=set(sumbit.index) & set(X_test.index)
	X_test=X_test.loc[intersection]
	X_test.to_csv('../data/test_data_all.csv')


	# In[73]:


	X_train0=pd.concat(train0,axis=0)
	X_train1=pd.concat(train1,axis=0)
	X_train2=pd.concat(train2,axis=0)
	X_train3=pd.concat(train3,axis=0)
	X_train4=pd.concat(train4,axis=0)
	X_train5=pd.concat(train5,axis=0)
	X_train6=pd.concat(train6,axis=0)
	X_train7=pd.concat(train7,axis=0)
	X_train8=pd.concat(train8,axis=0)


	# In[74]:


	###提取Label
	y_train0=df_group_is[df_group_is["END_DATE"]=='2009-06-30 00:00:00']["REVENUE"]
	y_train0=y_train0.reset_index().drop('END_DATE',axis=1).set_index('TICKER_SYMBOL')

	##更改index
	new_index=[]
	for index in y_train0.index:
		new_index.append((6-len(str(index)))*'0'+str(index))
	y_train0.index=new_index

	##只提取带Label的样本
	df_train0=pd.concat([X_train0,y_train0],axis=1,join='inner')


	# In[75]:


	###提取Label
	y_train1=df_group_is[df_group_is["END_DATE"]=='2010-06-30 00:00:00']["REVENUE"]
	y_train1=y_train1.reset_index().drop('END_DATE',axis=1).set_index('TICKER_SYMBOL')

	##更改index
	new_index=[]
	for index in y_train1.index:
		new_index.append((6-len(str(index)))*'0'+str(index))
	y_train1.index=new_index

	##只提取带Label的样本
	df_train1=pd.concat([X_train1,y_train1],axis=1,join='inner')


	# In[76]:


	###提取Label
	y_train2=df_group_is[df_group_is["END_DATE"]=='2011-06-30 00:00:00']["REVENUE"]
	y_train2=y_train2.reset_index().drop('END_DATE',axis=1).set_index('TICKER_SYMBOL')

	##更改index
	new_index=[]
	for index in y_train2.index:
		new_index.append((6-len(str(index)))*'0'+str(index))
	y_train2.index=new_index

	##只提取带Label的样本
	df_train2=pd.concat([X_train2,y_train2],axis=1,join='inner')


	# In[77]:


	###提取Label
	y_train3=df_group_is[df_group_is["END_DATE"]=='2012-06-30 00:00:00']["REVENUE"]
	y_train3=y_train3.reset_index().drop('END_DATE',axis=1).set_index('TICKER_SYMBOL')

	##更改index
	new_index=[]
	for index in y_train3.index:
		new_index.append((6-len(str(index)))*'0'+str(index))
	y_train3.index=new_index

	##只提取带Label的样本
	df_train3=pd.concat([X_train3,y_train3],axis=1,join='inner')


	# In[78]:


	###提取Label
	y_train4=df_group_is[df_group_is["END_DATE"]=='2013-06-30 00:00:00']["REVENUE"]
	y_train4=y_train4.reset_index().drop('END_DATE',axis=1).set_index('TICKER_SYMBOL')

	##更改index
	new_index=[]
	for index in y_train4.index:
		new_index.append((6-len(str(index)))*'0'+str(index))
	y_train4.index=new_index

	##只提取带Label的样本
	df_train4=pd.concat([X_train4,y_train4],axis=1,join='inner')


	# In[79]:


	###提取Label
	y_train5=df_group_is[df_group_is["END_DATE"]=='2014-06-30 00:00:00']["REVENUE"]
	y_train5=y_train5.reset_index().drop('END_DATE',axis=1).set_index('TICKER_SYMBOL')

	##更改index
	new_index=[]
	for index in y_train5.index:
		new_index.append((6-len(str(index)))*'0'+str(index))
	y_train5.index=new_index

	##只提取带Label的样本
	df_train5=pd.concat([X_train5,y_train5],axis=1,join='inner')


	# In[80]:


	###提取Label
	y_train6=df_group_is[df_group_is["END_DATE"]=='2015-06-30 00:00:00']["REVENUE"]
	y_train6=y_train6.reset_index().drop('END_DATE',axis=1).set_index('TICKER_SYMBOL')

	##更改index
	new_index=[]
	for index in y_train6.index:
		new_index.append((6-len(str(index)))*'0'+str(index))
	y_train6.index=new_index

	##只提取带Label的样本
	df_train6=pd.concat([X_train6,y_train6],axis=1,join='inner')


	# In[81]:


	###提取Label
	y_train7=df_group_is[df_group_is["END_DATE"]=='2016-06-30 00:00:00']["REVENUE"]
	y_train7=y_train7.reset_index().drop('END_DATE',axis=1).set_index('TICKER_SYMBOL')

	##更改index
	new_index=[]
	for index in y_train7.index:
		new_index.append((6-len(str(index)))*'0'+str(index))
	y_train7.index=new_index

	##只提取带Label的样本
	df_train7=pd.concat([X_train7,y_train7],axis=1,join='inner')


	# In[82]:


	###提取Label
	y_train8=df_group_is[df_group_is["END_DATE"]=='2017-06-30 00:00:00']["REVENUE"]
	y_train8=y_train8.reset_index().drop('END_DATE',axis=1).set_index('TICKER_SYMBOL')

	##更改index
	new_index=[]
	for index in y_train8.index:
		new_index.append((6-len(str(index)))*'0'+str(index))
	y_train8.index=new_index

	##只提取带Label的样本
	df_train8=pd.concat([X_train8,y_train8],axis=1,join='inner')


	# In[83]:


	df_train0.to_csv('../data/df_train2009_all.csv')
	df_train1.to_csv('../data/df_train2010_all.csv')
	df_train2.to_csv('../data/df_train2011_all.csv')
	df_train3.to_csv('../data/df_train2012_all.csv')
	df_train4.to_csv('../data/df_train2013_all.csv')
	df_train5.to_csv('../data/df_train2014_all.csv')
	df_train6.to_csv('../data/df_train2015_all.csv')
	df_train7.to_csv('../data/df_train2016_all.csv')
	df_train8.to_csv('../data/df_train2017_all.csv')

