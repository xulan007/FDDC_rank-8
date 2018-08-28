import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox,norm,skew
import catboost as cab
from tqdm import tqdm
import seaborn as sns
import warnings
warnings.simplefilter("ignore")

def Train_Model():
	### 股票代码与行业和市值的映射(2018-5-31)
	df_map=pd.read_csv('../data/maket_value_type_map.csv').set_index("TICKER_SYMBOL")

	###股票代码/日期与营业收入的映射
	dfs_revenue=pd.read_excel("../data/[Add June July] FDDC_financial_data_20180711/[New] Financial Data_20180711/Income Statement.xls",sheet_name=[0,1,2,3],parse_date='END_DATE')
	dfs_revenue=pd.concat(dfs_revenue,axis=0)
	df_revenue=dfs_revenue.sort_values('PUBLISH_DATE').groupby(
		[dfs_revenue['TICKER_SYMBOL'],dfs_revenue['END_DATE']]).apply(lambda x: x.iloc[-1])["REVENUE"]


	# In[3]:


	def cal_loss(date,ticker,revenue_pred):
		basic=min(abs(revenue_pred/df_revenue.loc[ticker,date]-1),0.8)
		alpha=np.log(max(df_map.loc[ticker]["MARKET_VALUE"],2))/np.log(2)
		
		return basic*alpha


	# 导入数据并进行异常值处理

	# In[17]:

	print("loading data")
	df_train=pd.read_csv("../data/train_all_na_done.csv")
	df_test=pd.read_csv("../data/test_all_na_done.csv")


	# In[18]:


	#df_train=df_train[df_train["END_DATE"]!="2011-06-30"]
	df_train=df_train[df_train["label"]>5e5]
	df_all=pd.concat([df_train,df_test],axis=0).reset_index().drop("index",axis=1)
	df_all["label"]=df_all["label"].apply(np.log)


	# In[19]:


	for col in ["recommend1","recommend2","recommend3"]:
		temp=[]
		for i in df_all[col]:
		    if i<=1:
		        temp.append(0)
		    else:
		        temp.append(np.log(i))
		df_all[col]=temp


	# In[20]:


	for col in tqdm(["recommend1","recommend2","recommend3"]):
		lower_limit=df_all[col].mean()-5*df_all[col].std()
		upper_limit=df_all[col].mean()+5*df_all[col].std()
		for i in range(df_all.shape[0]):
		    if df_all[col][i]>upper_limit:
		        df_all[col][i]=upper_limit
		    elif df_all[col][i]<lower_limit:
		        df_all[col][i]=lower_limit


	# 分离训练数据和测试数据

	# In[21]:


	assert df_all.shape[0]==df_test.shape[0]+df_train.shape[0]
	df_train=df_all[df_all["label"].notnull()]
	df_test=df_all[df_all["label"].isnull()].drop("label",axis=1)


	# 添加每个样本的权重列

	# In[22]:


	temp=[]
	for index in tqdm(df_train.index):
		market_value=df_map.loc[df_train.loc[index,"TICKER_SYMBOL"]]["MARKET_VALUE"]
		temp.append(np.log(max(market_value,2)))
	df_train["coef"]=temp


	# 计算只用,recommend1做计算loss,作为baseline

	# In[23]:


	# t_loss=0
	# for i in df_train.index[-1000:]:
	#     date=df_train.loc[i,"END_DATE"]
	#     ticker=df_train.loc[i,"TICKER_SYMBOL"].astype(int)
	#     val_pred=df_train.loc[i,"recommend1"]
	#     loss=cal_loss(date,ticker,np.exp(val_pred))
	#     if np.isnan(loss):
	#         loss=0.753
	#     t_loss=t_loss+loss
	# print(t_loss)
	# print(t_loss/1000)


	# In[24]:


	df_train=pd.concat([pd.get_dummies(df_train.drop("END_DATE",axis=1)),df_train["END_DATE"]],axis=1)


	# ### 给每个样本加权重,可以通过控制每个样本出现次数的方式实现

	# In[25]:


	#choice_index=np.random.choice(df_train.shape[0]-1000,200000,p=df_train["coef"][:-1000]/df_train["coef"][:-1000].sum())
	choice_index=[]
	for i in df_train.index[:-1000]:
		for j in range(int(np.round(df_train.loc[i,"coef"],0))):
		    choice_index.append(i)
		               
	df_train_new=df_train.loc[choice_index]

	df_test=pd.concat([pd.get_dummies(df_test.drop(["END_DATE","TICKER_SYMBOL"],axis=1)),df_test["TICKER_SYMBOL"]],axis=1)


	# ## tuning parameters

	# In[27]:


	# import xgboost as xgb
	# import lightgbm as lgb
	# from sklearn.linear_model import LinearRegression,Lasso

	# reg_xgb=xgb.XGBRegressor(colsample_bytree=0.725,gamma=0.57,
	#                              learning_rate=0.0083, max_depth=9,
	#                              min_child_weight=0.96, n_estimators=1478,
	#                              reg_alpha=0.38, reg_lambda=0.3,
	#                              subsample=0.84, silent=1,
	#                              random_state =8, nthread = 4)

	# reg_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=54,
	#                               learning_rate=0.02, n_estimators=820,
	#                               max_bin = 455, bagging_fraction = 0.69,
	#                               bagging_freq = 50, feature_fraction = 0.636,
	#                               feature_fraction_seed=9, bagging_seed=9,
	#                               min_data_in_leaf =15, min_sum_hessian_in_leaf = 6)

	# model1=reg_xgb.fit(df_train_new.drop(["END_DATE","TICKER_SYMBOL","label","coef"],axis=1),
	#                   df_train_new["label"])

	# pred1=model1.predict(df_train.drop(["END_DATE","TICKER_SYMBOL","label","coef"],axis=1)[-1000:])


	# model2=reg_lgb.fit(df_train_new.drop(["END_DATE","TICKER_SYMBOL","label","coef"],axis=1),
	#                   df_train_new["label"])

	# pred2=model2.predict(df_train.drop(["END_DATE","TICKER_SYMBOL","label","coef"],axis=1)[-1000:])

	# for i in range(6):
	#     rate_lgb=0.1*i
	#     pred=rate_lgb*pred2+pred1*(1-rate_lgb)
	#     t_loss=0
	#     for j,i in enumerate(df_train.index[-1000:]):
	#         date=df_train.loc[i,"END_DATE"]
	#         ticker=df_train.loc[i,"TICKER_SYMBOL"].astype(int)
	#         val_pred=pred[j]

	#         loss=cal_loss(date,ticker,np.exp(val_pred))
	#         t_loss=t_loss+loss
	#     print("with rate_lgb: {0}, we get loss:{1}".format(rate_lgb,t_loss))
	#     print("and the avg loss is:{}\n".format(t_loss/1000))


	# In[28]:


	#choice_index=np.random.choice(df_train.shape[0]-1000,200000,p=df_train["coef"][:-1000]/df_train["coef"][:-1000].sum())
	choice_index=[]
	for i in df_train.index:
		for j in range(int(np.round(df_train.loc[i,"coef"],0))):
		    choice_index.append(i)
		               
	df_train_all_new=df_train.loc[choice_index]


	# In[29]:


	import xgboost as xgb
	import lightgbm as lgb
	from sklearn.linear_model import LinearRegression,Lasso

	reg_xgb=xgb.XGBRegressor(colsample_bytree=0.725,gamma=0.57,
		                         learning_rate=0.0083, max_depth=9,
		                         min_child_weight=0.96, n_estimators=1478,
		                         reg_alpha=0.38, reg_lambda=0.3,
		                         subsample=0.84, silent=1,
		                         random_state =8, nthread = 4)

	reg_cab=cab.CatBoostRegressor(learning_rate=automatically, depth=9, iterations=1478,
		                         loss_function="MAE", l2_leaf_reg=3,
		                         thread_count=-1, silent=True,
		                         random_state =8)

	reg_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=54,
		                          learning_rate=0.02, n_estimators=820,
		                          max_bin = 455, bagging_fraction = 0.69,
		                          bagging_freq = 50, feature_fraction = 0.636,
		                          feature_fraction_seed=9, bagging_seed=9,
		                          min_data_in_leaf =15, min_sum_hessian_in_leaf = 6)

	model1=reg_xgb.fit(df_train_all_new.drop(["END_DATE","TICKER_SYMBOL","label","coef"],axis=1),
		              df_train_all_new["label"])

	model2=reg_lgb.fit(df_train_all_new.drop(["END_DATE","TICKER_SYMBOL","label","coef"],axis=1),
		              df_train_all_new["label"])

	pred_test1=model1.predict(df_test.drop(["TICKER_SYMBOL"],axis=1))
	pred_test2=model2.predict(df_test.drop(["TICKER_SYMBOL"],axis=1))
	pred_test=np.exp(pred_test1*0.6+pred_test2*0.4)
	df_test["prediction"]=pred_test


	# In[32]:


	df_sub=pd.read_csv("../data/FDDC_financial_submit_20180524.csv",header=None)
	df_sub.index=df_sub[0].apply(lambda x: x.split('.')[0]).apply(int)


	# In[33]:


	df_test=df_test.set_index("TICKER_SYMBOL").loc[df_sub.index]
	assert np.sum(df_sub.index!=df_sub.index)==0

	df_sub["value"]=df_test["prediction"].apply(lambda x: np.round(x/1e6,2))


	# In[34]:


	df_sub.to_csv("../submit/submission_20180715_181356.csv",header=False,index=False)

