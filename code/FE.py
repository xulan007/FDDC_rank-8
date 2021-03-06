

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def Feature_Engineer():

		df_2009=pd.read_csv("../data/df_train2009_all.csv")
		df_2010=pd.read_csv("../data/df_train2010_all.csv")
		df_2011=pd.read_csv("../data/df_train2011_all.csv")
		df_2012=pd.read_csv("../data/df_train2012_all.csv")
		df_2013=pd.read_csv("../data/df_train2013_all.csv")
		df_2014=pd.read_csv("../data/df_train2014_all.csv")
		df_2015=pd.read_csv("../data/df_train2015_all.csv")
		df_2016=pd.read_csv("../data/df_train2016_all.csv")
		df_2017=pd.read_csv("../data/df_train2017_all.csv")
		df_2018=pd.read_csv("../data/test_data_all.csv")


		# In[3]:


		df_holder_share=pd.read_csv("../data/holder_share.csv").set_index("TICKER_SYMBOL")
		df_macro=pd.read_excel('../data/[Add June July] FDDC_financial_data_20180711/[New] Macro&Industry _20180711.xlsx',sheet_name=[1],parse_dates=["PERIOD_DATE"])[1]

		df_map=pd.read_csv('../data/maket_value_type_map.csv').set_index("TICKER_SYMBOL")


		# In[5]:


		X_train_2018=pd.DataFrame()
		##提取过去三年的年中业绩
		X_train_2018["TICKER_SYMBOL"]=df_2018["Unnamed: 0"]

		X_train_2018["last_season"]=0
		X_train_2018["REVENUE_2"]=0
		X_train_2018["REVENUE_1"]=0
		X_train_2018["REVENUE_0"]=0

		for i in tqdm(df_2018.index):
			if np.isnan(df_2018["REVENUE_11"][i]):
				X_train_2018["last_season"][i]=df_2018["REVENUE_10"][i]/4
			else:
				X_train_2018["last_season"][i]=df_2018["REVENUE_11"][i]
			if np.isnan(df_2018["REVENUE_8"][i]):
				X_train_2018["REVENUE_2"][i]=df_2018["REVENUE_10"][i]/2
			else:
				X_train_2018["REVENUE_2"][i]=df_2018["REVENUE_8"][i]
			if np.isnan(df_2018["REVENUE_4"][i]):
				X_train_2018["REVENUE_1"][i]=X_train_2018["REVENUE_2"][i]
			else:
				X_train_2018["REVENUE_1"][i]=df_2018["REVENUE_2"][i]
			if np.isnan(df_2018["REVENUE_0"][i]):
				X_train_2018["REVENUE_0"][i]=X_train_2018["REVENUE_1"][i]
			else:
				X_train_2018["REVENUE_0"][i]=df_2018["REVENUE_0"][i]

		##一季度营业额同比增长率
		X_train_2018["END_DATE"]="2018-06-30"
		X_train_2018["incre_rate"]=df_2018["REVENUE_11"]/df_2018["REVENUE_7"]
		X_train_2018["市盈率"]=df_map.loc[X_train_2018["TICKER_SYMBOL"]]["CLOSE_PRICE_2018_05"].values/df_2018["BASIC_EPS_10"].values

		##推荐值
		X_train_2018["recommend1"]=X_train_2018["incre_rate"]*X_train_2018["REVENUE_2"]
		X_train_2018["recommend2"]=X_train_2018["REVENUE_2"]/X_train_2018["REVENUE_1"]*X_train_2018["REVENUE_2"]
		X_train_2018["recommend3"]=X_train_2018["REVENUE_2"]/df_2018["T_ASSETS_8"]*(df_2018["T_ASSETS_11"]+
				                                                                    df_2018["ESTIMATED_LIAB_11"].fillna(0))

		##获取一二股东持股比例
		X_train_2018["holder1"]=df_holder_share.loc[X_train_2018["TICKER_SYMBOL"]]["holder1"].values
		X_train_2018["holder2"]=df_holder_share.loc[X_train_2018["TICKER_SYMBOL"]]["holder2"].values
		X_train_2018["contain"]=X_train_2018["holder1"]/X_train_2018["holder2"]

		## 关于股票  获取股票流通占比  获取半年股价相对于大盘的增长率
		X_train_2018["flow_share"]=df_holder_share.loc[X_train_2018["TICKER_SYMBOL"]]["flow_share"].values
		X_train_2018["market_incre"]=df_map.loc[X_train_2018["TICKER_SYMBOL"]]["market_incre_2018"].values

		### 关于负债 获取过去三年资产负债率,流动负债比例,债券负债占比,长期负债比例,速动比率
		X_train_2018["liab_ago3"]=df_2018["T_LIAB_0"]/df_2018["T_ASSETS_0"]
		X_train_2018["liab_ago2"]=df_2018["T_LIAB_4"]/df_2018["T_ASSETS_4"]
		X_train_2018["liab_ago1"]=df_2018["T_LIAB_8"]/df_2018["T_ASSETS_8"]
		X_train_2018["liab_ago_half"]=df_2018["T_LIAB_10"]/df_2018["T_ASSETS_10"]
		X_train_2018["liab_ago_last"]=df_2018["T_LIAB_11"]/df_2018["T_ASSETS_11"]
		X_train_2018["fast_rate"]=df_2018["T_CA_11"]-df_2018["INVENTORIES_11"]/df_2018["T_CL_11"]

		##增长力指标
		X_train_2018["T_ASSETS"]=df_2018["T_ASSETS_11"]
		for i in tqdm(X_train_2018.index):
			if np.isnan(X_train_2018["T_ASSETS"][i]):
				X_train_2018["T_ASSETS"][i]=df_2018["T_ASSETS_10"][i]
				
		X_train_2018["每股收益增长率"]=(df_2018["BASIC_EPS_10"])/df_2018["BASIC_EPS_6"]        
		X_train_2018["利润留存率"]=(df_2018["N_INCOME_10"]-df_2018["DIV_PAYABLE_10"].fillna(0))/df_2018["N_INCOME_10"]
		X_train_2018["再投资率"]=(df_2018["N_INCOME_10"]/df_2018["T_SH_EQUITY_10"])*X_train_2018["利润留存率"]
		X_train_2018["总资产增长率"]=(df_2018["T_ASSETS_10"])/df_2018["T_ASSETS_6"]
		X_train_2018["固定资产扩张率"]=(df_2018["FIXED_ASSETS_10"])/df_2018["FIXED_ASSETS_6"] 
		X_train_2018["净利润增长率"]=(df_2018["N_INCOME_10"])/df_2018["N_INCOME_6"] 

		##研发投入
		X_train_2018["R_D_last"]=df_2018["R_D_11"].fillna(0)
		X_train_2018["R_D_ago1"]=df_2018["R_D_10"].fillna(0)
		X_train_2018["R_D_ago2"]=df_2018["R_D_6"].fillna(0)
		X_train_2018["R_D_ago3"]=df_2018["R_D_2"].fillna(0)
		##现金比率
		X_train_2018["cash_rate"]=df_2018["CASH_C_EQUIV_11"]/df_2018["T_CL_11"]

		X_train_2018["cl_rate"]=df_2018["T_CL_11"]/df_2018["T_LIAB_11"]
		X_train_2018["ncl_rate"]=df_2018["T_NCL_11"]/df_2018["T_LIAB_11"]
		X_train_2018["bond_rate"]=df_2018["BOND_PAYABLE_11"].fillna(0)/df_2018["T_LIAB_11"]

		### 关于利润 获取净资产收益率,营业利润率,

		X_train_2018["profit_asset"]=df_2018["OPERATE_PROFIT_10"]/(df_2018["T_ASSETS_10"]-df_2018["T_LIAB_10"])## 需更改为净资产
		X_train_2018["profit_rate"]=df_2018["OPERATE_PROFIT_11"]/df_2018["REVENUE_11"] 

		##关于资产  商誉占资产比重,总资产增长率,股东权益比率,权益乘数,流动资产比率,固定动资产比率,资本周转率
		X_train_2018["goodwill_rate"]=df_2018["GOODWILL_11"].fillna(0)/df_2018["T_ASSETS_10"]
		X_train_2018["T_ASSETS_rate"]=df_2018["T_ASSETS_11"]/df_2018["T_ASSETS_7"]
		X_train_2018["pure_asset_rate"]=(df_2018["T_ASSETS_11"]-df_2018["T_LIAB_11"])/df_2018["T_ASSETS_11"]
		X_train_2018["权益乘数"]=df_2018["T_ASSETS_11"]/((df_2018["T_ASSETS_11"]-df_2018["T_LIAB_11"]))
		X_train_2018["T_CA_rate"]=df_2018["T_CA_11"]/df_2018["T_ASSETS_11"]
		X_train_2018["FIXED_ASSETS_rate"]=df_2018["FIXED_ASSETS_11"]/df_2018["T_ASSETS_11"]
		X_train_2018["资本周转率"]=(df_2018["CASH_C_EQUIV_11"]+df_2018["NOTES_RECEIV_11"].fillna(0)+
				               df_2018["TRADING_FA_11"].fillna(0))/df_2018["T_NCL_11"]
		X_train_2018["固定比率"]=df_2018["FIXED_ASSETS_11"]/(df_2018["T_ASSETS_11"]-df_2018["T_LIAB_11"])
		##添加宏观影响因素

		X_train_2018["工业增加值"]=df_macro[(df_macro["indic_id"]==1020000004) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		t1=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		t2=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2018["火电发电量_增长率"]=(t1-t2)/t1

		X_train_2018["PMI_出口"]=df_macro[(df_macro["indic_id"]==1030000014) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		X_train_2018["PMI_成品库存"]=df_macro[(df_macro["indic_id"]==1030000016) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		X_train_2018["PMI_进口"]=df_macro[(df_macro["indic_id"]==1030000018) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		X_train_2018["PMI_原材料库存"]=df_macro[(df_macro["indic_id"]==1030000020) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		X_train_2018["CPI_消费品"]=df_macro[(df_macro["indic_id"]==1040000046) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		X_train_2018["CPI"]=df_macro[(df_macro["indic_id"]==1040000050) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()


		##货币供应


		t1=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2016-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2017-12-31')]["DATA_VALUE"].values
		X_train_2018["M0_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2016-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2017-12-31')]["DATA_VALUE"].values
		X_train_2018["M1_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2016-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2017-12-31')]["DATA_VALUE"].values
		X_train_2018["M2_rate"]=(t2/t1)[0]


		###利率 汇率

		X_train_2018["美元汇率"]=df_macro[(df_macro["indic_id"]==1080000235) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["贷款利率"]=df_macro[(df_macro["indic_id"]==1090000363) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		###

		X_train_2018["电脑出口"]=df_macro[(df_macro["indic_id"]==1100000874) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["煤进口"]=df_macro[(df_macro["indic_id"]==1100002293) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["煤产量"]=df_macro[(df_macro["indic_id"]==2020100020) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["煤销量"]=df_macro[(df_macro["indic_id"]==2020100024) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		X_train_2018["火车出境"]=df_macro[(df_macro["indic_id"]==1100005542) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["原油进口量"]=df_macro[(df_macro["indic_id"]==1100006640) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["原油价格"]=df_macro[(df_macro["indic_id"]==2020000719) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["汽车产量"]=df_macro[(df_macro["indic_id"]==2070100748) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["汽车销量"]=df_macro[(df_macro["indic_id"]==2070100795) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		X_train_2018["汽车库存"]=df_macro[(df_macro["indic_id"]==2070104273) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["SUV产量"]=df_macro[(df_macro["indic_id"]==2070109977) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["SUV销量"]=df_macro[(df_macro["indic_id"]==2070113040) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		X_train_2018["液晶出货"]=df_macro[(df_macro["indic_id"]==2090100464) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		###用电情况
		X_train_2018["全社会用电量"]=df_macro[(df_macro["indic_id"]==2020101521) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["第一产业用电"]=df_macro[(df_macro["indic_id"]==2020101522) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["第二产业用电"]=df_macro[(df_macro["indic_id"]==2020101523) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["第三产业用电"]=df_macro[(df_macro["indic_id"]==2020101524) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["工业用电"]=df_macro[(df_macro["indic_id"]==2020101526) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		X_train_2018["发电企业耗煤"]=df_macro[(df_macro["indic_id"]==2020102867) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		# 不考虑证券市场变化

		X_train_2018["CIP_0"]=df_2018["CIP_10"]
		X_train_2018["CIP_1"]=df_2018["CIP_6"]
		X_train_2018["CIP_2"]=df_2018["CIP_2"]

		X_train_2018["TAXES_PAYABLE"]=df_2018["TAXES_PAYABLE_11"]
		X_train_2018["AR"]=df_2018["AR_11"]/X_train_2018["last_season"]  ##应收账款占营业收入比
		X_train_2018["AR"]=df_2018["AR_11"]/(df_2018["T_ASSETS_11"]-df_2018["T_LIAB_11"]) ##应收账款占净资产比


		X_train_2018["融资融券余额"]=df_macro[(df_macro["indic_id"]==1170000018) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		X_train_2018["存管证券数量"]=df_macro[(df_macro["indic_id"]==1170000598) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["存管证券面值"]=df_macro[(df_macro["indic_id"]==1170000618) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["存管证券流通市值"]=df_macro[(df_macro["indic_id"]==1170000641) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["一级市场债券余额"]=df_macro[(df_macro["indic_id"]==1170007422) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()



		###房地产交易信息
		X_train_2018["商品房成交"]=df_macro[(df_macro["indic_id"]==2170702521) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["商品房待售面积"]=df_macro[(df_macro["indic_id"]==2170002035) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["信托收益_房地产"]=df_macro[(df_macro["indic_id"]==2210200588) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		X_train_2018["固定资产投资"]=df_macro[(df_macro["indic_id"]==1050000026) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		X_train_2018["房地产开发投资"]=df_macro[(df_macro["indic_id"]==1050000027) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()

		##物流航运
		X_train_2018["波罗的海运价"]=df_macro[(df_macro["indic_id"]==2160000004) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["出口集装箱运价"]=df_macro[(df_macro["indic_id"]==2160000101) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["公路客运量"]=df_macro[(df_macro["indic_id"]==2160000481) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["公路货运量"]=df_macro[(df_macro["indic_id"]==2160000489) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["铁路客运量"]=df_macro[(df_macro["indic_id"]==2160000875) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["铁路货运量"]=df_macro[(df_macro["indic_id"]==2160000883) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["广东水运"]=df_macro[(df_macro["indic_id"]==2160001002) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["港口吞吐"]=df_macro[(df_macro["indic_id"]==2160001523) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()
		X_train_2018["民航客运量"]=df_macro[(df_macro["indic_id"]==2160002772) & (df_macro["PERIOD_DATE"]>='2017-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2018-06-30')]["DATA_VALUE"].mean()


		# In[6]:


		X_train_2017=pd.DataFrame()
		##提取过去三年的年中业绩
		X_train_2017["TICKER_SYMBOL"]=df_2017["Unnamed: 0"]

		X_train_2017["last_season"]=0
		X_train_2017["REVENUE_2"]=0
		X_train_2017["REVENUE_1"]=0
		X_train_2017["REVENUE_0"]=0

		for i in tqdm(df_2017.index):
			if np.isnan(df_2017["REVENUE_11"][i]):
				X_train_2017["last_season"][i]=df_2017["REVENUE_10"][i]/4
			else:
				X_train_2017["last_season"][i]=df_2017["REVENUE_11"][i]
			if np.isnan(df_2017["REVENUE_8"][i]):
				X_train_2017["REVENUE_2"][i]=df_2017["REVENUE_10"][i]/2
			else:
				X_train_2017["REVENUE_2"][i]=df_2017["REVENUE_8"][i]
			if np.isnan(df_2017["REVENUE_4"][i]):
				X_train_2017["REVENUE_1"][i]=X_train_2017["REVENUE_2"][i]
			else:
				X_train_2017["REVENUE_1"][i]=df_2017["REVENUE_2"][i]
			if np.isnan(df_2017["REVENUE_0"][i]):
				X_train_2017["REVENUE_0"][i]=X_train_2017["REVENUE_1"][i]
			else:
				X_train_2017["REVENUE_0"][i]=df_2017["REVENUE_0"][i]

		##一季度营业额同比增长率
		X_train_2017["END_DATE"]="2017-06-30"
		X_train_2017["incre_rate"]=df_2017["REVENUE_11"]/df_2017["REVENUE_7"]
		X_train_2017["市盈率"]=df_map.loc[X_train_2017["TICKER_SYMBOL"]]["CLOSE_PRICE_2017_05"].values/df_2017["BASIC_EPS_10"].values


		##推荐值
		X_train_2017["recommend1"]=X_train_2017["incre_rate"]*X_train_2017["REVENUE_2"]
		X_train_2017["recommend2"]=X_train_2017["REVENUE_2"]/X_train_2017["REVENUE_1"]*X_train_2017["REVENUE_2"]
		X_train_2017["recommend3"]=X_train_2017["REVENUE_2"]/df_2017["T_ASSETS_8"]*(df_2017["T_ASSETS_11"]+
				                                                                    df_2017["ESTIMATED_LIAB_11"].fillna(0))

		##获取一二股东持股比例
		X_train_2017["holder1"]=df_holder_share.loc[X_train_2017["TICKER_SYMBOL"]]["holder1"].values
		X_train_2017["holder2"]=df_holder_share.loc[X_train_2017["TICKER_SYMBOL"]]["holder2"].values
		X_train_2017["contain"]=X_train_2017["holder1"]/X_train_2017["holder2"]

		## 关于股票  获取股票流通占比  获取半年股价相对于大盘的增长率
		X_train_2017["flow_share"]=df_holder_share.loc[X_train_2017["TICKER_SYMBOL"]]["flow_share"].values
		X_train_2017["market_incre"]=df_map.loc[X_train_2017["TICKER_SYMBOL"]]["market_incre_2017"].values

		### 关于负债 获取过去三年资产负债率,流动负债比例,债券负债占比,长期负债比例,速动比率
		X_train_2017["liab_ago3"]=df_2017["T_LIAB_0"]/df_2017["T_ASSETS_0"]
		X_train_2017["liab_ago2"]=df_2017["T_LIAB_4"]/df_2017["T_ASSETS_4"]
		X_train_2017["liab_ago1"]=df_2017["T_LIAB_8"]/df_2017["T_ASSETS_8"]
		X_train_2017["liab_ago_half"]=df_2017["T_LIAB_10"]/df_2017["T_ASSETS_10"]
		X_train_2017["liab_ago_last"]=df_2017["T_LIAB_11"]/df_2017["T_ASSETS_11"]
		X_train_2017["fast_rate"]=df_2017["T_CA_11"]-df_2017["INVENTORIES_11"]/df_2017["T_CL_11"]

		##研发投入
		X_train_2017["T_ASSETS"]=df_2017["T_ASSETS_11"]
		for i in tqdm(X_train_2017.index):
			if np.isnan(X_train_2017["T_ASSETS"][i]):
				X_train_2017["T_ASSETS"][i]=df_2017["T_ASSETS_10"][i]
				
		X_train_2017["每股收益增长率"]=(df_2017["BASIC_EPS_10"])/df_2017["BASIC_EPS_6"]        
		X_train_2017["利润留存率"]=(df_2017["N_INCOME_10"]-df_2017["DIV_PAYABLE_10"].fillna(0))/df_2017["N_INCOME_10"]
		X_train_2017["再投资率"]=(df_2017["N_INCOME_10"]/df_2017["T_SH_EQUITY_10"])*X_train_2017["利润留存率"]
		X_train_2017["总资产增长率"]=(df_2017["T_ASSETS_10"])/df_2017["T_ASSETS_6"]
		X_train_2017["固定资产扩张率"]=(df_2017["FIXED_ASSETS_10"])/df_2017["FIXED_ASSETS_6"] 
		X_train_2017["净利润增长率"]=(df_2017["N_INCOME_10"])/df_2017["N_INCOME_6"] 

		X_train_2017["R_D_last"]=df_2017["R_D_11"].fillna(0)
		X_train_2017["R_D_ago1"]=df_2017["R_D_10"].fillna(0)
		X_train_2017["R_D_ago2"]=df_2017["R_D_6"].fillna(0)
		X_train_2017["R_D_ago3"]=df_2017["R_D_2"].fillna(0)
		##现金比率
		X_train_2017["cash_rate"]=df_2017["CASH_C_EQUIV_11"]/df_2017["T_CL_11"]

		X_train_2017["cl_rate"]=df_2017["T_CL_11"]/df_2017["T_LIAB_11"]
		X_train_2017["ncl_rate"]=df_2017["T_NCL_11"]/df_2017["T_LIAB_11"]
		X_train_2017["bond_rate"]=df_2017["BOND_PAYABLE_11"].fillna(0)/df_2017["T_LIAB_11"]

		### 关于利润 获取净资产收益率,营业利润率,

		X_train_2017["profit_asset"]=df_2017["OPERATE_PROFIT_10"]/(df_2017["T_ASSETS_10"]-df_2017["T_LIAB_10"])## 需更改为净资产
		X_train_2017["profit_rate"]=df_2017["OPERATE_PROFIT_11"]/df_2017["REVENUE_11"] 

		##关于资产  商誉占资产比重,总资产增长率,股东权益比率,权益乘数,流动资产比率,固定动资产比率,资本周转率
		X_train_2017["goodwill_rate"]=df_2017["GOODWILL_11"].fillna(0)/df_2017["T_ASSETS_10"]
		X_train_2017["T_ASSETS_rate"]=df_2017["T_ASSETS_11"]/df_2017["T_ASSETS_7"]
		X_train_2017["pure_asset_rate"]=(df_2017["T_ASSETS_11"]-df_2017["T_LIAB_11"])/df_2017["T_ASSETS_11"]
		X_train_2017["权益乘数"]=df_2017["T_ASSETS_11"]/((df_2017["T_ASSETS_11"]-df_2017["T_LIAB_11"]))
		X_train_2017["T_CA_rate"]=df_2017["T_CA_11"]/df_2017["T_ASSETS_11"]
		X_train_2017["FIXED_ASSETS_rate"]=df_2017["FIXED_ASSETS_11"]/df_2017["T_ASSETS_11"]
		X_train_2017["资本周转率"]=(df_2017["CASH_C_EQUIV_11"]+df_2017["NOTES_RECEIV_11"].fillna(0)+
				               df_2017["TRADING_FA_11"].fillna(0))/df_2017["T_NCL_11"]
		X_train_2017["固定比率"]=df_2017["FIXED_ASSETS_11"]/(df_2017["T_ASSETS_11"]-df_2017["T_LIAB_11"])
		##添加宏观影响因素

		X_train_2017["工业增加值"]=df_macro[(df_macro["indic_id"]==1020000004) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		t1=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		t2=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2017["火电发电量_增长率"]=(t1-t2)/t1

		X_train_2017["PMI_出口"]=df_macro[(df_macro["indic_id"]==1030000014) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		X_train_2017["PMI_成品库存"]=df_macro[(df_macro["indic_id"]==1030000016) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		X_train_2017["PMI_进口"]=df_macro[(df_macro["indic_id"]==1030000018) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		X_train_2017["PMI_原材料库存"]=df_macro[(df_macro["indic_id"]==1030000020) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		X_train_2017["CPI_消费品"]=df_macro[(df_macro["indic_id"]==1040000046) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		X_train_2017["CPI"]=df_macro[(df_macro["indic_id"]==1040000050) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()


		##货币供应


		t1=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2015-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2016-12-31')]["DATA_VALUE"].values
		X_train_2017["M0_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2015-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2016-12-31')]["DATA_VALUE"].values
		X_train_2017["M1_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2015-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2016-12-31')]["DATA_VALUE"].values
		X_train_2017["M2_rate"]=(t2/t1)[0]


		###利率 汇率

		X_train_2017["美元汇率"]=df_macro[(df_macro["indic_id"]==1080000235) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["贷款利率"]=df_macro[(df_macro["indic_id"]==1090000363) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		###

		X_train_2017["电脑出口"]=df_macro[(df_macro["indic_id"]==1100000874) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["煤进口"]=df_macro[(df_macro["indic_id"]==1100002293) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["煤产量"]=df_macro[(df_macro["indic_id"]==2020100020) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["煤销量"]=df_macro[(df_macro["indic_id"]==2020100024) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		X_train_2017["火车出境"]=df_macro[(df_macro["indic_id"]==1100005542) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["原油进口量"]=df_macro[(df_macro["indic_id"]==1100006640) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["原油价格"]=df_macro[(df_macro["indic_id"]==2020000719) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["汽车产量"]=df_macro[(df_macro["indic_id"]==2070100748) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["汽车销量"]=df_macro[(df_macro["indic_id"]==2070100795) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		X_train_2017["汽车库存"]=df_macro[(df_macro["indic_id"]==2070104273) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["SUV产量"]=df_macro[(df_macro["indic_id"]==2070109977) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["SUV销量"]=df_macro[(df_macro["indic_id"]==2070113040) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		X_train_2017["液晶出货"]=df_macro[(df_macro["indic_id"]==2090100464) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		###用电情况
		X_train_2017["全社会用电量"]=df_macro[(df_macro["indic_id"]==2020101521) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["第一产业用电"]=df_macro[(df_macro["indic_id"]==2020101522) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["第二产业用电"]=df_macro[(df_macro["indic_id"]==2020101523) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["第三产业用电"]=df_macro[(df_macro["indic_id"]==2020101524) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["工业用电"]=df_macro[(df_macro["indic_id"]==2020101526) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		X_train_2017["发电企业耗煤"]=df_macro[(df_macro["indic_id"]==2020102867) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		# 不考虑证券市场变化

		X_train_2017["CIP_0"]=df_2017["CIP_10"]
		X_train_2017["CIP_1"]=df_2017["CIP_6"]
		X_train_2017["CIP_2"]=df_2017["CIP_2"]

		X_train_2017["TAXES_PAYABLE"]=df_2017["TAXES_PAYABLE_11"]
		X_train_2017["AR"]=df_2017["AR_11"]/X_train_2017["last_season"]  ##应收账款占营业收入比
		X_train_2017["AR"]=df_2017["AR_11"]/(df_2017["T_ASSETS_11"]-df_2017["T_LIAB_11"]) ##应收账款占净资产比


		X_train_2017["融资融券余额"]=df_macro[(df_macro["indic_id"]==1170000018) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		X_train_2017["存管证券数量"]=df_macro[(df_macro["indic_id"]==1170000598) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["存管证券面值"]=df_macro[(df_macro["indic_id"]==1170000618) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["存管证券流通市值"]=df_macro[(df_macro["indic_id"]==1170000641) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["一级市场债券余额"]=df_macro[(df_macro["indic_id"]==1170007422) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		##房地产交易情况


		X_train_2017["商品房成交"]=df_macro[(df_macro["indic_id"]==2170702521) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["商品房待售面积"]=df_macro[(df_macro["indic_id"]==2170002035) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["信托收益_房地产"]=df_macro[(df_macro["indic_id"]==2210200588) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		X_train_2017["固定资产投资"]=df_macro[(df_macro["indic_id"]==1050000026) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		X_train_2017["房地产开发投资"]=df_macro[(df_macro["indic_id"]==1050000027) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()

		##物流航运
		X_train_2017["波罗的海运价"]=df_macro[(df_macro["indic_id"]==2160000004) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["出口集装箱运价"]=df_macro[(df_macro["indic_id"]==2160000101) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["公路客运量"]=df_macro[(df_macro["indic_id"]==2160000481) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["公路货运量"]=df_macro[(df_macro["indic_id"]==2160000489) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["铁路客运量"]=df_macro[(df_macro["indic_id"]==2160000875) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["铁路货运量"]=df_macro[(df_macro["indic_id"]==2160000883) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["广东水运"]=df_macro[(df_macro["indic_id"]==2160001002) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["港口吞吐"]=df_macro[(df_macro["indic_id"]==2160001523) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["民航客运量"]=df_macro[(df_macro["indic_id"]==2160002772) & (df_macro["PERIOD_DATE"]>='2016-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2017-06-30')]["DATA_VALUE"].mean()
		X_train_2017["label"]=df_2017["REVENUE"]


		# In[7]:


		X_train_2016=pd.DataFrame()
		##提取过去三年的年中业绩
		X_train_2016["TICKER_SYMBOL"]=df_2016["Unnamed: 0"]

		X_train_2016["last_season"]=0
		X_train_2016["REVENUE_2"]=0
		X_train_2016["REVENUE_1"]=0
		X_train_2016["REVENUE_0"]=0

		for i in tqdm(df_2016.index):
			if np.isnan(df_2016["REVENUE_11"][i]):
				X_train_2016["last_season"][i]=df_2016["REVENUE_10"][i]/4
			else:
				X_train_2016["last_season"][i]=df_2016["REVENUE_11"][i]
			if np.isnan(df_2016["REVENUE_8"][i]):
				X_train_2016["REVENUE_2"][i]=df_2016["REVENUE_10"][i]/2
			else:
				X_train_2016["REVENUE_2"][i]=df_2016["REVENUE_8"][i]
			if np.isnan(df_2016["REVENUE_4"][i]):
				X_train_2016["REVENUE_1"][i]=X_train_2016["REVENUE_2"][i]
			else:
				X_train_2016["REVENUE_1"][i]=df_2016["REVENUE_2"][i]
			if np.isnan(df_2016["REVENUE_0"][i]):
				X_train_2016["REVENUE_0"][i]=X_train_2016["REVENUE_1"][i]
			else:
				X_train_2016["REVENUE_0"][i]=df_2016["REVENUE_0"][i]

		##一季度营业额同比增长率
		X_train_2016["END_DATE"]="2016-06-30"
		X_train_2016["incre_rate"]=df_2016["REVENUE_11"]/df_2016["REVENUE_7"]
		X_train_2016["市盈率"]=df_map.loc[X_train_2016["TICKER_SYMBOL"]]["CLOSE_PRICE_2016_05"].values/df_2016["BASIC_EPS_10"].values

		##推荐值
		X_train_2016["recommend1"]=X_train_2016["incre_rate"]*X_train_2016["REVENUE_2"]
		X_train_2016["recommend2"]=X_train_2016["REVENUE_2"]/X_train_2016["REVENUE_1"]*X_train_2016["REVENUE_2"]
		X_train_2016["recommend3"]=X_train_2016["REVENUE_2"]/df_2016["T_ASSETS_8"]*(df_2016["T_ASSETS_11"]+
				                                                                    df_2016["ESTIMATED_LIAB_11"].fillna(0))

		##获取一二股东持股比例
		X_train_2016["holder1"]=df_holder_share.loc[X_train_2016["TICKER_SYMBOL"]]["holder1"].values
		X_train_2016["holder2"]=df_holder_share.loc[X_train_2016["TICKER_SYMBOL"]]["holder2"].values
		X_train_2016["contain"]=X_train_2016["holder1"]/X_train_2016["holder2"]

		## 关于股票  获取股票流通占比  获取半年股价相对于大盘的增长率
		X_train_2016["flow_share"]=df_holder_share.loc[X_train_2016["TICKER_SYMBOL"]]["flow_share"].values
		X_train_2016["market_incre"]=df_map.loc[X_train_2016["TICKER_SYMBOL"]]["market_incre_2016"].values

		### 关于负债 获取过去三年资产负债率,流动负债比例,债券负债占比,长期负债比例,速动比率
		X_train_2016["liab_ago3"]=df_2016["T_LIAB_0"]/df_2016["T_ASSETS_0"]
		X_train_2016["liab_ago2"]=df_2016["T_LIAB_4"]/df_2016["T_ASSETS_4"]
		X_train_2016["liab_ago1"]=df_2016["T_LIAB_8"]/df_2016["T_ASSETS_8"]
		X_train_2016["liab_ago_half"]=df_2016["T_LIAB_10"]/df_2016["T_ASSETS_10"]
		X_train_2016["liab_ago_last"]=df_2016["T_LIAB_11"]/df_2016["T_ASSETS_11"]
		X_train_2016["fast_rate"]=df_2016["T_CA_11"]-df_2016["INVENTORIES_11"]/df_2016["T_CL_11"]

		##研发投入
		X_train_2016["T_ASSETS"]=df_2016["T_ASSETS_11"]
		for i in tqdm(X_train_2016.index):
			if np.isnan(X_train_2016["T_ASSETS"][i]):
				X_train_2016["T_ASSETS"][i]=df_2016["T_ASSETS_10"][i]
				
		X_train_2016["每股收益增长率"]=(df_2016["BASIC_EPS_10"])/df_2016["BASIC_EPS_6"]        
		X_train_2016["利润留存率"]=(df_2016["N_INCOME_10"]-df_2016["DIV_PAYABLE_10"].fillna(0))/df_2016["N_INCOME_10"]
		X_train_2016["再投资率"]=(df_2016["N_INCOME_10"]/df_2016["T_SH_EQUITY_10"])*X_train_2016["利润留存率"]
		X_train_2016["总资产增长率"]=(df_2016["T_ASSETS_10"])/df_2016["T_ASSETS_6"]
		X_train_2016["固定资产扩张率"]=(df_2016["FIXED_ASSETS_10"])/df_2016["FIXED_ASSETS_6"] 
		X_train_2016["净利润增长率"]=(df_2016["N_INCOME_10"])/df_2016["N_INCOME_6"] 

		X_train_2016["R_D_last"]=df_2016["R_D_11"].fillna(0)
		X_train_2016["R_D_ago1"]=df_2016["R_D_10"].fillna(0)
		X_train_2016["R_D_ago2"]=df_2016["R_D_6"].fillna(0)
		X_train_2016["R_D_ago3"]=df_2016["R_D_2"].fillna(0)
		##现金比率
		X_train_2016["cash_rate"]=df_2016["CASH_C_EQUIV_11"]/df_2016["T_CL_11"]

		X_train_2016["cl_rate"]=df_2016["T_CL_11"]/df_2016["T_LIAB_11"]
		X_train_2016["ncl_rate"]=df_2016["T_NCL_11"]/df_2016["T_LIAB_11"]
		X_train_2016["bond_rate"]=df_2016["BOND_PAYABLE_11"].fillna(0)/df_2016["T_LIAB_11"]

		### 关于利润 获取净资产收益率,营业利润率,

		X_train_2016["profit_asset"]=df_2016["OPERATE_PROFIT_10"]/(df_2016["T_ASSETS_10"]-df_2016["T_LIAB_10"])## 需更改为净资产
		X_train_2016["profit_rate"]=df_2016["OPERATE_PROFIT_11"]/df_2016["REVENUE_11"] 

		##关于资产  商誉占资产比重,总资产增长率,股东权益比率,权益乘数,流动资产比率,固定动资产比率,资本周转率
		X_train_2016["goodwill_rate"]=df_2016["GOODWILL_11"].fillna(0)/df_2016["T_ASSETS_10"]
		X_train_2016["T_ASSETS_rate"]=df_2016["T_ASSETS_11"]/df_2016["T_ASSETS_7"]
		X_train_2016["pure_asset_rate"]=(df_2016["T_ASSETS_11"]-df_2016["T_LIAB_11"])/df_2016["T_ASSETS_11"]
		X_train_2016["权益乘数"]=df_2016["T_ASSETS_11"]/((df_2016["T_ASSETS_11"]-df_2016["T_LIAB_11"]))
		X_train_2016["T_CA_rate"]=df_2016["T_CA_11"]/df_2016["T_ASSETS_11"]
		X_train_2016["FIXED_ASSETS_rate"]=df_2016["FIXED_ASSETS_11"]/df_2016["T_ASSETS_11"]
		X_train_2016["资本周转率"]=(df_2016["CASH_C_EQUIV_11"]+df_2016["NOTES_RECEIV_11"].fillna(0)+
				               df_2016["TRADING_FA_11"].fillna(0))/df_2016["T_NCL_11"]
		X_train_2016["固定比率"]=df_2016["FIXED_ASSETS_11"]/(df_2016["T_ASSETS_11"]-df_2016["T_LIAB_11"])
		##添加宏观影响因素

		X_train_2016["工业增加值"]=df_macro[(df_macro["indic_id"]==1020000004) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		t1=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		t2=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2016["火电发电量_增长率"]=(t1-t2)/t1

		X_train_2016["PMI_出口"]=df_macro[(df_macro["indic_id"]==1030000014) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		X_train_2016["PMI_成品库存"]=df_macro[(df_macro["indic_id"]==1030000016) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		X_train_2016["PMI_进口"]=df_macro[(df_macro["indic_id"]==1030000018) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		X_train_2016["PMI_原材料库存"]=df_macro[(df_macro["indic_id"]==1030000020) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		X_train_2016["CPI_消费品"]=df_macro[(df_macro["indic_id"]==1040000046) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		X_train_2016["CPI"]=df_macro[(df_macro["indic_id"]==1040000050) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()


		##货币供应


		t1=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2014-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2015-12-31')]["DATA_VALUE"].values
		X_train_2016["M0_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2014-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2015-12-31')]["DATA_VALUE"].values
		X_train_2016["M1_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2014-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2015-12-31')]["DATA_VALUE"].values
		X_train_2016["M2_rate"]=(t2/t1)[0]


		###利率 汇率

		X_train_2016["美元汇率"]=df_macro[(df_macro["indic_id"]==1080000235) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["贷款利率"]=df_macro[(df_macro["indic_id"]==1090000363) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		###

		X_train_2016["电脑出口"]=df_macro[(df_macro["indic_id"]==1100000874) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["煤进口"]=df_macro[(df_macro["indic_id"]==1100002293) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["煤产量"]=df_macro[(df_macro["indic_id"]==2020100020) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["煤销量"]=df_macro[(df_macro["indic_id"]==2020100024) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		X_train_2016["火车出境"]=df_macro[(df_macro["indic_id"]==1100005542) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["原油进口量"]=df_macro[(df_macro["indic_id"]==1100006640) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["原油价格"]=df_macro[(df_macro["indic_id"]==2020000719) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["汽车产量"]=df_macro[(df_macro["indic_id"]==2070100748) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["汽车销量"]=df_macro[(df_macro["indic_id"]==2070100795) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		X_train_2016["汽车库存"]=df_macro[(df_macro["indic_id"]==2070104273) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["SUV产量"]=df_macro[(df_macro["indic_id"]==2070109977) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["SUV销量"]=df_macro[(df_macro["indic_id"]==2070113040) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		X_train_2016["液晶出货"]=df_macro[(df_macro["indic_id"]==2090100464) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		###用电情况
		X_train_2016["全社会用电量"]=df_macro[(df_macro["indic_id"]==2020101521) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["第一产业用电"]=df_macro[(df_macro["indic_id"]==2020101522) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["第二产业用电"]=df_macro[(df_macro["indic_id"]==2020101523) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["第三产业用电"]=df_macro[(df_macro["indic_id"]==2020101524) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["工业用电"]=df_macro[(df_macro["indic_id"]==2020101526) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		X_train_2016["发电企业耗煤"]=df_macro[(df_macro["indic_id"]==2020102867) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		# 不考虑证券市场变化


		X_train_2016["CIP_0"]=df_2016["CIP_10"]
		X_train_2016["CIP_1"]=df_2016["CIP_6"]
		X_train_2016["CIP_2"]=df_2016["CIP_2"]

		X_train_2016["TAXES_PAYABLE"]=df_2016["TAXES_PAYABLE_11"]
		X_train_2016["AR"]=df_2016["AR_11"]/X_train_2016["last_season"]  ##应收账款占营业收入比
		X_train_2016["AR"]=df_2016["AR_11"]/(df_2016["T_ASSETS_11"]-df_2016["T_LIAB_11"]) ##应收账款占净资产比


		X_train_2016["融资融券余额"]=df_macro[(df_macro["indic_id"]==1170000018) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		X_train_2016["存管证券数量"]=df_macro[(df_macro["indic_id"]==1170000598) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["存管证券面值"]=df_macro[(df_macro["indic_id"]==1170000618) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["存管证券流通市值"]=df_macro[(df_macro["indic_id"]==1170000641) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["一级市场债券余额"]=df_macro[(df_macro["indic_id"]==1170007422) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		##房地产交易情况

		X_train_2016["商品房成交"]=df_macro[(df_macro["indic_id"]==2170702521) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["商品房待售面积"]=df_macro[(df_macro["indic_id"]==2170002035) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["信托收益_房地产"]=df_macro[(df_macro["indic_id"]==2210200588) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		X_train_2016["固定资产投资"]=df_macro[(df_macro["indic_id"]==1050000026) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		X_train_2016["房地产开发投资"]=df_macro[(df_macro["indic_id"]==1050000027) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()

		##物流航运
		X_train_2016["波罗的海运价"]=df_macro[(df_macro["indic_id"]==2160000004) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["出口集装箱运价"]=df_macro[(df_macro["indic_id"]==2160000101) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["公路客运量"]=df_macro[(df_macro["indic_id"]==2160000481) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["公路货运量"]=df_macro[(df_macro["indic_id"]==2160000489) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["铁路客运量"]=df_macro[(df_macro["indic_id"]==2160000875) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["铁路货运量"]=df_macro[(df_macro["indic_id"]==2160000883) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["广东水运"]=df_macro[(df_macro["indic_id"]==2160001002) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["港口吞吐"]=df_macro[(df_macro["indic_id"]==2160001523) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["民航客运量"]=df_macro[(df_macro["indic_id"]==2160002772) & (df_macro["PERIOD_DATE"]>='2015-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2016-06-30')]["DATA_VALUE"].mean()
		X_train_2016["label"]=df_2016["REVENUE"]


		# In[8]:


		X_train_2015=pd.DataFrame()
		##提取过去三年的年中业绩
		X_train_2015["TICKER_SYMBOL"]=df_2015["Unnamed: 0"]

		X_train_2015["last_season"]=0
		X_train_2015["REVENUE_2"]=0
		X_train_2015["REVENUE_1"]=0
		X_train_2015["REVENUE_0"]=0

		for i in tqdm(df_2015.index):
			if np.isnan(df_2015["REVENUE_11"][i]):
				X_train_2015["last_season"][i]=df_2015["REVENUE_10"][i]/4
			else:
				X_train_2015["last_season"][i]=df_2015["REVENUE_11"][i]
			if np.isnan(df_2015["REVENUE_8"][i]):
				X_train_2015["REVENUE_2"][i]=df_2015["REVENUE_10"][i]/2
			else:
				X_train_2015["REVENUE_2"][i]=df_2015["REVENUE_8"][i]
			if np.isnan(df_2015["REVENUE_4"][i]):
				X_train_2015["REVENUE_1"][i]=X_train_2015["REVENUE_2"][i]
			else:
				X_train_2015["REVENUE_1"][i]=df_2015["REVENUE_2"][i]
			if np.isnan(df_2015["REVENUE_0"][i]):
				X_train_2015["REVENUE_0"][i]=X_train_2015["REVENUE_1"][i]
			else:
				X_train_2015["REVENUE_0"][i]=df_2015["REVENUE_0"][i]

		##一季度营业额同比增长率
		X_train_2015["END_DATE"]="2015-06-30"
		X_train_2015["incre_rate"]=df_2015["REVENUE_11"]/df_2015["REVENUE_7"]
		X_train_2015["市盈率"]=df_map.loc[X_train_2015["TICKER_SYMBOL"]]["CLOSE_PRICE_2015_05"].values/df_2015["BASIC_EPS_10"].values

		##推荐值
		X_train_2015["recommend1"]=X_train_2015["incre_rate"]*X_train_2015["REVENUE_2"]
		X_train_2015["recommend2"]=X_train_2015["REVENUE_2"]/X_train_2015["REVENUE_1"]*X_train_2015["REVENUE_2"]
		X_train_2015["recommend3"]=X_train_2015["REVENUE_2"]/df_2015["T_ASSETS_8"]*(df_2015["T_ASSETS_11"]+
				                                                                    df_2015["ESTIMATED_LIAB_11"].fillna(0))

		##获取一二股东持股比例
		X_train_2015["holder1"]=df_holder_share.loc[X_train_2015["TICKER_SYMBOL"]]["holder1"].values
		X_train_2015["holder2"]=df_holder_share.loc[X_train_2015["TICKER_SYMBOL"]]["holder2"].values
		X_train_2015["contain"]=X_train_2015["holder1"]/X_train_2015["holder2"]

		## 关于股票  获取股票流通占比  获取半年股价相对于大盘的增长率
		X_train_2015["flow_share"]=df_holder_share.loc[X_train_2015["TICKER_SYMBOL"]]["flow_share"].values
		X_train_2015["market_incre"]=df_map.loc[X_train_2015["TICKER_SYMBOL"]]["market_incre_2015"].values

		### 关于负债 获取过去三年资产负债率,流动负债比例,债券负债占比,长期负债比例,速动比率
		X_train_2015["liab_ago3"]=df_2015["T_LIAB_0"]/df_2015["T_ASSETS_0"]
		X_train_2015["liab_ago2"]=df_2015["T_LIAB_4"]/df_2015["T_ASSETS_4"]
		X_train_2015["liab_ago1"]=df_2015["T_LIAB_8"]/df_2015["T_ASSETS_8"]
		X_train_2015["liab_ago_half"]=df_2015["T_LIAB_10"]/df_2015["T_ASSETS_10"]
		X_train_2015["liab_ago_last"]=df_2015["T_LIAB_11"]/df_2015["T_ASSETS_11"]
		X_train_2015["fast_rate"]=df_2015["T_CA_11"]-df_2015["INVENTORIES_11"]/df_2015["T_CL_11"]

		##研发投入
		X_train_2015["T_ASSETS"]=df_2015["T_ASSETS_11"]
		for i in tqdm(X_train_2015.index):
			if np.isnan(X_train_2015["T_ASSETS"][i]):
				X_train_2015["T_ASSETS"][i]=df_2015["T_ASSETS_10"][i]
				
		X_train_2015["每股收益增长率"]=(df_2015["BASIC_EPS_10"])/df_2015["BASIC_EPS_6"]        
		X_train_2015["利润留存率"]=(df_2015["N_INCOME_10"]-df_2015["DIV_PAYABLE_10"].fillna(0))/df_2015["N_INCOME_10"]
		X_train_2015["再投资率"]=(df_2015["N_INCOME_10"]/df_2015["T_SH_EQUITY_10"])*X_train_2015["利润留存率"]
		X_train_2015["总资产增长率"]=(df_2015["T_ASSETS_10"])/df_2015["T_ASSETS_6"]
		X_train_2015["固定资产扩张率"]=(df_2015["FIXED_ASSETS_10"])/df_2015["FIXED_ASSETS_6"] 
		X_train_2015["净利润增长率"]=(df_2015["N_INCOME_10"])/df_2015["N_INCOME_6"] 

		X_train_2015["R_D_last"]=df_2015["R_D_11"].fillna(0)
		X_train_2015["R_D_ago1"]=df_2015["R_D_10"].fillna(0)
		X_train_2015["R_D_ago2"]=df_2015["R_D_6"].fillna(0)
		X_train_2015["R_D_ago3"]=df_2015["R_D_2"].fillna(0)
		##现金比率
		X_train_2015["cash_rate"]=df_2015["CASH_C_EQUIV_11"]/df_2015["T_CL_11"]

		X_train_2015["cl_rate"]=df_2015["T_CL_11"]/df_2015["T_LIAB_11"]
		X_train_2015["ncl_rate"]=df_2015["T_NCL_11"]/df_2015["T_LIAB_11"]
		X_train_2015["bond_rate"]=df_2015["BOND_PAYABLE_11"].fillna(0)/df_2015["T_LIAB_11"]

		### 关于利润 获取净资产收益率,营业利润率,

		X_train_2015["profit_asset"]=df_2015["OPERATE_PROFIT_10"]/(df_2015["T_ASSETS_10"]-df_2015["T_LIAB_10"])## 需更改为净资产
		X_train_2015["profit_rate"]=df_2015["OPERATE_PROFIT_11"]/df_2015["REVENUE_11"] 

		##关于资产  商誉占资产比重,总资产增长率,股东权益比率,权益乘数,流动资产比率,固定动资产比率,资本周转率
		X_train_2015["goodwill_rate"]=df_2015["GOODWILL_11"].fillna(0)/df_2015["T_ASSETS_10"]
		X_train_2015["T_ASSETS_rate"]=df_2015["T_ASSETS_11"]/df_2015["T_ASSETS_7"]
		X_train_2015["pure_asset_rate"]=(df_2015["T_ASSETS_11"]-df_2015["T_LIAB_11"])/df_2015["T_ASSETS_11"]
		X_train_2015["权益乘数"]=df_2015["T_ASSETS_11"]/((df_2015["T_ASSETS_11"]-df_2015["T_LIAB_11"]))
		X_train_2015["T_CA_rate"]=df_2015["T_CA_11"]/df_2015["T_ASSETS_11"]
		X_train_2015["FIXED_ASSETS_rate"]=df_2015["FIXED_ASSETS_11"]/df_2015["T_ASSETS_11"]
		X_train_2015["资本周转率"]=(df_2015["CASH_C_EQUIV_11"]+df_2015["NOTES_RECEIV_11"].fillna(0)+
				               df_2015["TRADING_FA_11"].fillna(0))/df_2015["T_NCL_11"]
		X_train_2015["固定比率"]=df_2015["FIXED_ASSETS_11"]/(df_2015["T_ASSETS_11"]-df_2015["T_LIAB_11"])
		##添加宏观影响因素

		X_train_2015["工业增加值"]=df_macro[(df_macro["indic_id"]==1020000004) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		t1=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		t2=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2015["火电发电量_增长率"]=(t1-t2)/t1

		X_train_2015["PMI_出口"]=df_macro[(df_macro["indic_id"]==1030000014) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		X_train_2015["PMI_成品库存"]=df_macro[(df_macro["indic_id"]==1030000016) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		X_train_2015["PMI_进口"]=df_macro[(df_macro["indic_id"]==1030000018) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		X_train_2015["PMI_原材料库存"]=df_macro[(df_macro["indic_id"]==1030000020) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		X_train_2015["CPI_消费品"]=df_macro[(df_macro["indic_id"]==1040000046) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		X_train_2015["CPI"]=df_macro[(df_macro["indic_id"]==1040000050) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()


		##货币供应


		t1=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2013-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2014-12-31')]["DATA_VALUE"].values
		X_train_2015["M0_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2013-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2014-12-31')]["DATA_VALUE"].values
		X_train_2015["M1_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2013-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2014-12-31')]["DATA_VALUE"].values
		X_train_2015["M2_rate"]=(t2/t1)[0]


		###利率 汇率

		X_train_2015["美元汇率"]=df_macro[(df_macro["indic_id"]==1080000235) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["贷款利率"]=df_macro[(df_macro["indic_id"]==1090000363) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		###

		X_train_2015["电脑出口"]=df_macro[(df_macro["indic_id"]==1100000874) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["煤进口"]=df_macro[(df_macro["indic_id"]==1100002293) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["煤产量"]=df_macro[(df_macro["indic_id"]==2020100020) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["煤销量"]=df_macro[(df_macro["indic_id"]==2020100024) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		X_train_2015["火车出境"]=df_macro[(df_macro["indic_id"]==1100005542) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["原油进口量"]=df_macro[(df_macro["indic_id"]==1100006640) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["原油价格"]=df_macro[(df_macro["indic_id"]==2020000719) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["汽车产量"]=df_macro[(df_macro["indic_id"]==2070100748) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["汽车销量"]=df_macro[(df_macro["indic_id"]==2070100795) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		X_train_2015["汽车库存"]=df_macro[(df_macro["indic_id"]==2070104273) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["SUV产量"]=df_macro[(df_macro["indic_id"]==2070109977) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["SUV销量"]=df_macro[(df_macro["indic_id"]==2070113040) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		X_train_2015["液晶出货"]=df_macro[(df_macro["indic_id"]==2090100464) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		###用电情况
		X_train_2015["全社会用电量"]=df_macro[(df_macro["indic_id"]==2020101521) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["第一产业用电"]=df_macro[(df_macro["indic_id"]==2020101522) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["第二产业用电"]=df_macro[(df_macro["indic_id"]==2020101523) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["第三产业用电"]=df_macro[(df_macro["indic_id"]==2020101524) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["工业用电"]=df_macro[(df_macro["indic_id"]==2020101526) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		X_train_2015["发电企业耗煤"]=df_macro[(df_macro["indic_id"]==2020102867) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		# 不考虑证券市场变化

		X_train_2015["CIP_0"]=df_2015["CIP_10"]
		X_train_2015["CIP_1"]=df_2015["CIP_6"]
		X_train_2015["CIP_2"]=df_2015["CIP_2"]

		X_train_2015["TAXES_PAYABLE"]=df_2015["TAXES_PAYABLE_11"]
		X_train_2015["AR"]=df_2015["AR_11"]/X_train_2015["last_season"]  ##应收账款占营业收入比
		X_train_2015["AR"]=df_2015["AR_11"]/(df_2015["T_ASSETS_11"]-df_2015["T_LIAB_11"]) ##应收账款占净资产比


		X_train_2015["融资融券余额"]=df_macro[(df_macro["indic_id"]==1170000018) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		X_train_2015["存管证券数量"]=df_macro[(df_macro["indic_id"]==1170000598) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["存管证券面值"]=df_macro[(df_macro["indic_id"]==1170000618) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["存管证券流通市值"]=df_macro[(df_macro["indic_id"]==1170000641) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["一级市场债券余额"]=df_macro[(df_macro["indic_id"]==1170007422) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		###房地产交易

		X_train_2015["商品房成交"]=df_macro[(df_macro["indic_id"]==2170702521) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["商品房待售面积"]=df_macro[(df_macro["indic_id"]==2170002035) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["信托收益_房地产"]=df_macro[(df_macro["indic_id"]==2210200588) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		X_train_2015["固定资产投资"]=df_macro[(df_macro["indic_id"]==1050000026) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		X_train_2015["房地产开发投资"]=df_macro[(df_macro["indic_id"]==1050000027) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()

		##物流航运
		X_train_2015["波罗的海运价"]=df_macro[(df_macro["indic_id"]==2160000004) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["出口集装箱运价"]=df_macro[(df_macro["indic_id"]==2160000101) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["公路客运量"]=df_macro[(df_macro["indic_id"]==2160000481) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["公路货运量"]=df_macro[(df_macro["indic_id"]==2160000489) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["铁路客运量"]=df_macro[(df_macro["indic_id"]==2160000875) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["铁路货运量"]=df_macro[(df_macro["indic_id"]==2160000883) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["广东水运"]=df_macro[(df_macro["indic_id"]==2160001002) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["港口吞吐"]=df_macro[(df_macro["indic_id"]==2160001523) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["民航客运量"]=df_macro[(df_macro["indic_id"]==2160002772) & (df_macro["PERIOD_DATE"]>='2014-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2015-06-30')]["DATA_VALUE"].mean()
		X_train_2015["label"]=df_2015["REVENUE"]


		# In[9]:


		X_train_2014=pd.DataFrame()
		##提取过去三年的年中业绩
		X_train_2014["TICKER_SYMBOL"]=df_2014["Unnamed: 0"]

		X_train_2014["last_season"]=0
		X_train_2014["REVENUE_2"]=0
		X_train_2014["REVENUE_1"]=0
		X_train_2014["REVENUE_0"]=0

		for i in tqdm(df_2014.index):
			if np.isnan(df_2014["REVENUE_11"][i]):
				X_train_2014["last_season"][i]=df_2014["REVENUE_10"][i]/4
			else:
				X_train_2014["last_season"][i]=df_2014["REVENUE_11"][i]
			if np.isnan(df_2014["REVENUE_8"][i]):
				X_train_2014["REVENUE_2"][i]=df_2014["REVENUE_10"][i]/2
			else:
				X_train_2014["REVENUE_2"][i]=df_2014["REVENUE_8"][i]
			if np.isnan(df_2014["REVENUE_4"][i]):
				X_train_2014["REVENUE_1"][i]=X_train_2014["REVENUE_2"][i]
			else:
				X_train_2014["REVENUE_1"][i]=df_2014["REVENUE_2"][i]
			if np.isnan(df_2014["REVENUE_0"][i]):
				X_train_2014["REVENUE_0"][i]=X_train_2014["REVENUE_1"][i]
			else:
				X_train_2014["REVENUE_0"][i]=df_2014["REVENUE_0"][i]

		##一季度营业额同比增长率
		X_train_2014["END_DATE"]="2014-06-30"
		X_train_2014["incre_rate"]=df_2014["REVENUE_11"]/df_2014["REVENUE_7"]
		X_train_2014["市盈率"]=df_map.loc[X_train_2014["TICKER_SYMBOL"]]["CLOSE_PRICE_2014_05"].values/df_2014["BASIC_EPS_10"].values

		##推荐值
		X_train_2014["recommend1"]=X_train_2014["incre_rate"]*X_train_2014["REVENUE_2"]
		X_train_2014["recommend2"]=X_train_2014["REVENUE_2"]/X_train_2014["REVENUE_1"]*X_train_2014["REVENUE_2"]
		X_train_2014["recommend3"]=X_train_2014["REVENUE_2"]/df_2014["T_ASSETS_8"]*(df_2014["T_ASSETS_11"]+
				                                                                    df_2014["ESTIMATED_LIAB_11"].fillna(0))

		##获取一二股东持股比例
		X_train_2014["holder1"]=df_holder_share.loc[X_train_2014["TICKER_SYMBOL"]]["holder1"].values
		X_train_2014["holder2"]=df_holder_share.loc[X_train_2014["TICKER_SYMBOL"]]["holder2"].values
		X_train_2014["contain"]=X_train_2014["holder1"]/X_train_2014["holder2"]

		## 关于股票  获取股票流通占比  获取半年股价相对于大盘的增长率
		X_train_2014["flow_share"]=df_holder_share.loc[X_train_2014["TICKER_SYMBOL"]]["flow_share"].values
		X_train_2014["market_incre"]=df_map.loc[X_train_2014["TICKER_SYMBOL"]]["market_incre_2014"].values

		### 关于负债 获取过去三年资产负债率,流动负债比例,债券负债占比,长期负债比例,速动比率
		X_train_2014["liab_ago3"]=df_2014["T_LIAB_0"]/df_2014["T_ASSETS_0"]
		X_train_2014["liab_ago2"]=df_2014["T_LIAB_4"]/df_2014["T_ASSETS_4"]
		X_train_2014["liab_ago1"]=df_2014["T_LIAB_8"]/df_2014["T_ASSETS_8"]
		X_train_2014["liab_ago_half"]=df_2014["T_LIAB_10"]/df_2014["T_ASSETS_10"]
		X_train_2014["liab_ago_last"]=df_2014["T_LIAB_11"]/df_2014["T_ASSETS_11"]
		X_train_2014["fast_rate"]=df_2014["T_CA_11"]-df_2014["INVENTORIES_11"]/df_2014["T_CL_11"]

		##研发投入
		X_train_2014["T_ASSETS"]=df_2014["T_ASSETS_11"]
		for i in tqdm(X_train_2014.index):
			if np.isnan(X_train_2014["T_ASSETS"][i]):
				X_train_2014["T_ASSETS"][i]=df_2014["T_ASSETS_10"][i]
				
		X_train_2014["每股收益增长率"]=(df_2014["BASIC_EPS_10"])/df_2014["BASIC_EPS_6"]        
		X_train_2014["利润留存率"]=(df_2014["N_INCOME_10"]-df_2014["DIV_PAYABLE_10"].fillna(0))/df_2014["N_INCOME_10"]
		X_train_2014["再投资率"]=(df_2014["N_INCOME_10"]/df_2014["T_SH_EQUITY_10"])*X_train_2014["利润留存率"]
		X_train_2014["总资产增长率"]=(df_2014["T_ASSETS_10"])/df_2014["T_ASSETS_6"]
		X_train_2014["固定资产扩张率"]=(df_2014["FIXED_ASSETS_10"])/df_2014["FIXED_ASSETS_6"] 
		X_train_2014["净利润增长率"]=(df_2014["N_INCOME_10"])/df_2014["N_INCOME_6"] 

		X_train_2014["R_D_last"]=df_2014["R_D_11"].fillna(0)
		X_train_2014["R_D_ago1"]=df_2014["R_D_10"].fillna(0)
		X_train_2014["R_D_ago2"]=df_2014["R_D_6"].fillna(0)
		X_train_2014["R_D_ago3"]=df_2014["R_D_2"].fillna(0)
		##现金比率
		X_train_2014["cash_rate"]=df_2014["CASH_C_EQUIV_11"]/df_2014["T_CL_11"]

		X_train_2014["cl_rate"]=df_2014["T_CL_11"]/df_2014["T_LIAB_11"]
		X_train_2014["ncl_rate"]=df_2014["T_NCL_11"]/df_2014["T_LIAB_11"]
		X_train_2014["bond_rate"]=df_2014["BOND_PAYABLE_11"].fillna(0)/df_2014["T_LIAB_11"]

		### 关于利润 获取净资产收益率,营业利润率,

		X_train_2014["profit_asset"]=df_2014["OPERATE_PROFIT_10"]/(df_2014["T_ASSETS_10"]-df_2014["T_LIAB_10"])## 需更改为净资产
		X_train_2014["profit_rate"]=df_2014["OPERATE_PROFIT_11"]/df_2014["REVENUE_11"] 

		##关于资产  商誉占资产比重,总资产增长率,股东权益比率,权益乘数,流动资产比率,固定动资产比率,资本周转率
		X_train_2014["goodwill_rate"]=df_2014["GOODWILL_11"].fillna(0)/df_2014["T_ASSETS_10"]
		X_train_2014["T_ASSETS_rate"]=df_2014["T_ASSETS_11"]/df_2014["T_ASSETS_7"]
		X_train_2014["pure_asset_rate"]=(df_2014["T_ASSETS_11"]-df_2014["T_LIAB_11"])/df_2014["T_ASSETS_11"]
		X_train_2014["权益乘数"]=df_2014["T_ASSETS_11"]/((df_2014["T_ASSETS_11"]-df_2014["T_LIAB_11"]))
		X_train_2014["T_CA_rate"]=df_2014["T_CA_11"]/df_2014["T_ASSETS_11"]
		X_train_2014["FIXED_ASSETS_rate"]=df_2014["FIXED_ASSETS_11"]/df_2014["T_ASSETS_11"]
		X_train_2014["资本周转率"]=(df_2014["CASH_C_EQUIV_11"]+df_2014["NOTES_RECEIV_11"].fillna(0)+
				               df_2014["TRADING_FA_11"].fillna(0))/df_2014["T_NCL_11"]
		X_train_2014["固定比率"]=df_2014["FIXED_ASSETS_11"]/(df_2014["T_ASSETS_11"]-df_2014["T_LIAB_11"])
		##添加宏观影响因素

		X_train_2014["工业增加值"]=df_macro[(df_macro["indic_id"]==1020000004) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		t1=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		t2=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2014["火电发电量_增长率"]=(t1-t2)/t1

		X_train_2014["PMI_出口"]=df_macro[(df_macro["indic_id"]==1030000014) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		X_train_2014["PMI_成品库存"]=df_macro[(df_macro["indic_id"]==1030000016) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		X_train_2014["PMI_进口"]=df_macro[(df_macro["indic_id"]==1030000018) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		X_train_2014["PMI_原材料库存"]=df_macro[(df_macro["indic_id"]==1030000020) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		X_train_2014["CPI_消费品"]=df_macro[(df_macro["indic_id"]==1040000046) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		X_train_2014["CPI"]=df_macro[(df_macro["indic_id"]==1040000050) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()


		##货币供应


		t1=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2012-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2013-12-31')]["DATA_VALUE"].values
		X_train_2014["M0_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2012-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2013-12-31')]["DATA_VALUE"].values
		X_train_2014["M1_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2012-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2013-12-31')]["DATA_VALUE"].values
		X_train_2014["M2_rate"]=(t2/t1)[0]


		###利率 汇率

		X_train_2014["美元汇率"]=df_macro[(df_macro["indic_id"]==1080000235) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["贷款利率"]=df_macro[(df_macro["indic_id"]==1090000363) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		###

		X_train_2014["电脑出口"]=df_macro[(df_macro["indic_id"]==1100000874) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["煤进口"]=df_macro[(df_macro["indic_id"]==1100002293) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["煤产量"]=df_macro[(df_macro["indic_id"]==2020100020) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["煤销量"]=df_macro[(df_macro["indic_id"]==2020100024) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		X_train_2014["火车出境"]=df_macro[(df_macro["indic_id"]==1100005542) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["原油进口量"]=df_macro[(df_macro["indic_id"]==1100006640) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["原油价格"]=df_macro[(df_macro["indic_id"]==2020000719) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["汽车产量"]=df_macro[(df_macro["indic_id"]==2070100748) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["汽车销量"]=df_macro[(df_macro["indic_id"]==2070100795) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		X_train_2014["汽车库存"]=df_macro[(df_macro["indic_id"]==2070104273) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["SUV产量"]=df_macro[(df_macro["indic_id"]==2070109977) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["SUV销量"]=df_macro[(df_macro["indic_id"]==2070113040) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		X_train_2014["液晶出货"]=df_macro[(df_macro["indic_id"]==2090100464) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		###用电情况
		X_train_2014["全社会用电量"]=df_macro[(df_macro["indic_id"]==2020101521) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["第一产业用电"]=df_macro[(df_macro["indic_id"]==2020101522) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["第二产业用电"]=df_macro[(df_macro["indic_id"]==2020101523) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["第三产业用电"]=df_macro[(df_macro["indic_id"]==2020101524) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["工业用电"]=df_macro[(df_macro["indic_id"]==2020101526) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		X_train_2014["发电企业耗煤"]=df_macro[(df_macro["indic_id"]==2020102867) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		# 不考虑证券市场变化

		X_train_2014["CIP_0"]=df_2014["CIP_10"]
		X_train_2014["CIP_1"]=df_2014["CIP_6"]
		X_train_2014["CIP_2"]=df_2014["CIP_2"]

		X_train_2014["TAXES_PAYABLE"]=df_2014["TAXES_PAYABLE_11"]
		X_train_2014["AR"]=df_2014["AR_11"]/X_train_2014["last_season"]  ##应收账款占营业收入比
		X_train_2014["AR"]=df_2014["AR_11"]/(df_2014["T_ASSETS_11"]-df_2014["T_LIAB_11"]) ##应收账款占净资产比


		X_train_2014["融资融券余额"]=df_macro[(df_macro["indic_id"]==1170000018) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		X_train_2014["存管证券数量"]=df_macro[(df_macro["indic_id"]==1170000598) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["存管证券面值"]=df_macro[(df_macro["indic_id"]==1170000618) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["存管证券流通市值"]=df_macro[(df_macro["indic_id"]==1170000641) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["一级市场债券余额"]=df_macro[(df_macro["indic_id"]==1170007422) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		## 房地产交易情况

		X_train_2014["商品房成交"]=df_macro[(df_macro["indic_id"]==2170702521) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["商品房待售面积"]=df_macro[(df_macro["indic_id"]==2170002035) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["信托收益_房地产"]=df_macro[(df_macro["indic_id"]==2210200588) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		X_train_2014["固定资产投资"]=df_macro[(df_macro["indic_id"]==1050000026) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		X_train_2014["房地产开发投资"]=df_macro[(df_macro["indic_id"]==1050000027) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()

		##物流航运
		X_train_2014["波罗的海运价"]=df_macro[(df_macro["indic_id"]==2160000004) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["出口集装箱运价"]=df_macro[(df_macro["indic_id"]==2160000101) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["公路客运量"]=df_macro[(df_macro["indic_id"]==2160000481) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["公路货运量"]=df_macro[(df_macro["indic_id"]==2160000489) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["铁路客运量"]=df_macro[(df_macro["indic_id"]==2160000875) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["铁路货运量"]=df_macro[(df_macro["indic_id"]==2160000883) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["广东水运"]=df_macro[(df_macro["indic_id"]==2160001002) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["港口吞吐"]=df_macro[(df_macro["indic_id"]==2160001523) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["民航客运量"]=df_macro[(df_macro["indic_id"]==2160002772) & (df_macro["PERIOD_DATE"]>='2013-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2014-06-30')]["DATA_VALUE"].mean()
		X_train_2014["label"]=df_2014["REVENUE"]


		# In[10]:


		X_train_2013=pd.DataFrame()
		##提取过去三年的年中业绩
		X_train_2013["TICKER_SYMBOL"]=df_2013["Unnamed: 0"]

		X_train_2013["last_season"]=0
		X_train_2013["REVENUE_2"]=0
		X_train_2013["REVENUE_1"]=0
		X_train_2013["REVENUE_0"]=0

		for i in tqdm(df_2013.index):
			if np.isnan(df_2013["REVENUE_11"][i]):
				X_train_2013["last_season"][i]=df_2013["REVENUE_10"][i]/4
			else:
				X_train_2013["last_season"][i]=df_2013["REVENUE_11"][i]
			if np.isnan(df_2013["REVENUE_8"][i]):
				X_train_2013["REVENUE_2"][i]=df_2013["REVENUE_10"][i]/2
			else:
				X_train_2013["REVENUE_2"][i]=df_2013["REVENUE_8"][i]
			if np.isnan(df_2013["REVENUE_4"][i]):
				X_train_2013["REVENUE_1"][i]=X_train_2013["REVENUE_2"][i]
			else:
				X_train_2013["REVENUE_1"][i]=df_2013["REVENUE_2"][i]
			if np.isnan(df_2013["REVENUE_0"][i]):
				X_train_2013["REVENUE_0"][i]=X_train_2013["REVENUE_1"][i]
			else:
				X_train_2013["REVENUE_0"][i]=df_2013["REVENUE_0"][i]

		##一季度营业额同比增长率
		X_train_2013["END_DATE"]="2013-06-30"
		X_train_2013["incre_rate"]=df_2013["REVENUE_11"]/df_2013["REVENUE_7"]
		X_train_2013["市盈率"]=df_map.loc[X_train_2013["TICKER_SYMBOL"]]["CLOSE_PRICE_2013_05"].values/df_2013["BASIC_EPS_10"].values

		##推荐值
		X_train_2013["recommend1"]=X_train_2013["incre_rate"]*X_train_2013["REVENUE_2"]
		X_train_2013["recommend2"]=X_train_2013["REVENUE_2"]/X_train_2013["REVENUE_1"]*X_train_2013["REVENUE_2"]
		X_train_2013["recommend3"]=X_train_2013["REVENUE_2"]/df_2013["T_ASSETS_8"]*(df_2013["T_ASSETS_11"]+
				                                                                    df_2013["ESTIMATED_LIAB_11"].fillna(0))

		##获取一二股东持股比例
		X_train_2013["holder1"]=df_holder_share.loc[X_train_2013["TICKER_SYMBOL"]]["holder1"].values
		X_train_2013["holder2"]=df_holder_share.loc[X_train_2013["TICKER_SYMBOL"]]["holder2"].values
		X_train_2013["contain"]=X_train_2013["holder1"]/X_train_2013["holder2"]

		## 关于股票  获取股票流通占比  获取半年股价相对于大盘的增长率
		X_train_2013["flow_share"]=df_holder_share.loc[X_train_2013["TICKER_SYMBOL"]]["flow_share"].values
		X_train_2013["market_incre"]=df_map.loc[X_train_2013["TICKER_SYMBOL"]]["market_incre_2013"].values

		### 关于负债 获取过去三年资产负债率,流动负债比例,债券负债占比,长期负债比例,速动比率
		X_train_2013["liab_ago3"]=df_2013["T_LIAB_0"]/df_2013["T_ASSETS_0"]
		X_train_2013["liab_ago2"]=df_2013["T_LIAB_4"]/df_2013["T_ASSETS_4"]
		X_train_2013["liab_ago1"]=df_2013["T_LIAB_8"]/df_2013["T_ASSETS_8"]
		X_train_2013["liab_ago_half"]=df_2013["T_LIAB_10"]/df_2013["T_ASSETS_10"]
		X_train_2013["liab_ago_last"]=df_2013["T_LIAB_11"]/df_2013["T_ASSETS_11"]
		X_train_2013["fast_rate"]=df_2013["T_CA_11"]-df_2013["INVENTORIES_11"]/df_2013["T_CL_11"]

		##研发投入
		X_train_2013["T_ASSETS"]=df_2013["T_ASSETS_11"]
		for i in tqdm(X_train_2013.index):
			if np.isnan(X_train_2013["T_ASSETS"][i]):
				X_train_2013["T_ASSETS"][i]=df_2013["T_ASSETS_10"][i]
				
		X_train_2013["每股收益增长率"]=(df_2013["BASIC_EPS_10"])/df_2013["BASIC_EPS_6"]        
		X_train_2013["利润留存率"]=(df_2013["N_INCOME_10"]-df_2013["DIV_PAYABLE_10"].fillna(0))/df_2013["N_INCOME_10"]
		X_train_2013["再投资率"]=(df_2013["N_INCOME_10"]/df_2013["T_SH_EQUITY_10"])*X_train_2013["利润留存率"]
		X_train_2013["总资产增长率"]=(df_2013["T_ASSETS_10"])/df_2013["T_ASSETS_6"]
		X_train_2013["固定资产扩张率"]=(df_2013["FIXED_ASSETS_10"])/df_2013["FIXED_ASSETS_6"] 
		X_train_2013["净利润增长率"]=(df_2013["N_INCOME_10"])/df_2013["N_INCOME_6"] 

		X_train_2013["R_D_last"]=df_2013["R_D_11"].fillna(0)
		X_train_2013["R_D_ago1"]=df_2013["R_D_10"].fillna(0)
		X_train_2013["R_D_ago2"]=df_2013["R_D_6"].fillna(0)
		X_train_2013["R_D_ago3"]=df_2013["R_D_2"].fillna(0)
		##现金比率
		X_train_2013["cash_rate"]=df_2013["CASH_C_EQUIV_11"]/df_2013["T_CL_11"]

		X_train_2013["cl_rate"]=df_2013["T_CL_11"]/df_2013["T_LIAB_11"]
		X_train_2013["ncl_rate"]=df_2013["T_NCL_11"]/df_2013["T_LIAB_11"]
		X_train_2013["bond_rate"]=df_2013["BOND_PAYABLE_11"].fillna(0)/df_2013["T_LIAB_11"]

		### 关于利润 获取净资产收益率,营业利润率,

		X_train_2013["profit_asset"]=df_2013["OPERATE_PROFIT_10"]/(df_2013["T_ASSETS_10"]-df_2013["T_LIAB_10"])## 需更改为净资产
		X_train_2013["profit_rate"]=df_2013["OPERATE_PROFIT_11"]/df_2013["REVENUE_11"] 

		##关于资产  商誉占资产比重,总资产增长率,股东权益比率,权益乘数,流动资产比率,固定动资产比率,资本周转率
		X_train_2013["goodwill_rate"]=df_2013["GOODWILL_11"].fillna(0)/df_2013["T_ASSETS_10"]
		X_train_2013["T_ASSETS_rate"]=df_2013["T_ASSETS_11"]/df_2013["T_ASSETS_7"]
		X_train_2013["pure_asset_rate"]=(df_2013["T_ASSETS_11"]-df_2013["T_LIAB_11"])/df_2013["T_ASSETS_11"]
		X_train_2013["权益乘数"]=df_2013["T_ASSETS_11"]/((df_2013["T_ASSETS_11"]-df_2013["T_LIAB_11"]))
		X_train_2013["T_CA_rate"]=df_2013["T_CA_11"]/df_2013["T_ASSETS_11"]
		X_train_2013["FIXED_ASSETS_rate"]=df_2013["FIXED_ASSETS_11"]/df_2013["T_ASSETS_11"]
		X_train_2013["资本周转率"]=(df_2013["CASH_C_EQUIV_11"]+df_2013["NOTES_RECEIV_11"].fillna(0)+
				               df_2013["TRADING_FA_11"].fillna(0))/df_2013["T_NCL_11"]
		X_train_2013["固定比率"]=df_2013["FIXED_ASSETS_11"]/(df_2013["T_ASSETS_11"]-df_2013["T_LIAB_11"])
		##添加宏观影响因素

		X_train_2013["工业增加值"]=df_macro[(df_macro["indic_id"]==1020000004) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		t1=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		t2=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2013["火电发电量_增长率"]=(t1-t2)/t1

		X_train_2013["PMI_出口"]=df_macro[(df_macro["indic_id"]==1030000014) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		X_train_2013["PMI_成品库存"]=df_macro[(df_macro["indic_id"]==1030000016) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		X_train_2013["PMI_进口"]=df_macro[(df_macro["indic_id"]==1030000018) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		X_train_2013["PMI_原材料库存"]=df_macro[(df_macro["indic_id"]==1030000020) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		X_train_2013["CPI_消费品"]=df_macro[(df_macro["indic_id"]==1040000046) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		X_train_2013["CPI"]=df_macro[(df_macro["indic_id"]==1040000050) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()


		##货币供应


		t1=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2011-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2012-12-31')]["DATA_VALUE"].values
		X_train_2013["M0_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2011-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2012-12-31')]["DATA_VALUE"].values
		X_train_2013["M1_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2011-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2012-12-31')]["DATA_VALUE"].values
		X_train_2013["M2_rate"]=(t2/t1)[0]


		###利率 汇率

		X_train_2013["美元汇率"]=df_macro[(df_macro["indic_id"]==1080000235) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["贷款利率"]=df_macro[(df_macro["indic_id"]==1090000363) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		###

		X_train_2013["电脑出口"]=df_macro[(df_macro["indic_id"]==1100000874) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["煤进口"]=df_macro[(df_macro["indic_id"]==1100002293) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["煤产量"]=df_macro[(df_macro["indic_id"]==2020100020) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["煤销量"]=df_macro[(df_macro["indic_id"]==2020100024) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		X_train_2013["火车出境"]=df_macro[(df_macro["indic_id"]==1100005542) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["原油进口量"]=df_macro[(df_macro["indic_id"]==1100006640) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["原油价格"]=df_macro[(df_macro["indic_id"]==2020000719) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["汽车产量"]=df_macro[(df_macro["indic_id"]==2070100748) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["汽车销量"]=df_macro[(df_macro["indic_id"]==2070100795) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		X_train_2013["汽车库存"]=df_macro[(df_macro["indic_id"]==2070104273) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["SUV产量"]=df_macro[(df_macro["indic_id"]==2070109977) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["SUV销量"]=df_macro[(df_macro["indic_id"]==2070113040) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		X_train_2013["液晶出货"]=df_macro[(df_macro["indic_id"]==2090100464) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		###用电情况
		X_train_2013["全社会用电量"]=df_macro[(df_macro["indic_id"]==2020101521) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["第一产业用电"]=df_macro[(df_macro["indic_id"]==2020101522) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["第二产业用电"]=df_macro[(df_macro["indic_id"]==2020101523) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["第三产业用电"]=df_macro[(df_macro["indic_id"]==2020101524) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["工业用电"]=df_macro[(df_macro["indic_id"]==2020101526) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		X_train_2013["发电企业耗煤"]=df_macro[(df_macro["indic_id"]==2020102867) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		# 不考虑证券市场变化

		X_train_2013["CIP_0"]=df_2013["CIP_10"]
		X_train_2013["CIP_1"]=df_2013["CIP_6"]
		X_train_2013["CIP_2"]=df_2013["CIP_2"]

		X_train_2013["TAXES_PAYABLE"]=df_2013["TAXES_PAYABLE_11"]
		X_train_2013["AR"]=df_2013["AR_11"]/X_train_2013["last_season"]  ##应收账款占营业收入比
		X_train_2013["AR"]=df_2013["AR_11"]/(df_2013["T_ASSETS_11"]-df_2013["T_LIAB_11"]) ##应收账款占净资产比


		X_train_2013["融资融券余额"]=df_macro[(df_macro["indic_id"]==1170000018) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		X_train_2013["存管证券数量"]=df_macro[(df_macro["indic_id"]==1170000598) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["存管证券面值"]=df_macro[(df_macro["indic_id"]==1170000618) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["存管证券流通市值"]=df_macro[(df_macro["indic_id"]==1170000641) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["一级市场债券余额"]=df_macro[(df_macro["indic_id"]==1170007422) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		##房地产交易情况

		X_train_2013["商品房成交"]=df_macro[(df_macro["indic_id"]==2170702521) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["商品房待售面积"]=df_macro[(df_macro["indic_id"]==2170002035) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["信托收益_房地产"]=df_macro[(df_macro["indic_id"]==2210200588) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		X_train_2013["固定资产投资"]=df_macro[(df_macro["indic_id"]==1050000026) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		X_train_2013["房地产开发投资"]=df_macro[(df_macro["indic_id"]==1050000027) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()

		##物流航运
		X_train_2013["波罗的海运价"]=df_macro[(df_macro["indic_id"]==2160000004) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["出口集装箱运价"]=df_macro[(df_macro["indic_id"]==2160000101) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["公路客运量"]=df_macro[(df_macro["indic_id"]==2160000481) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["公路货运量"]=df_macro[(df_macro["indic_id"]==2160000489) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["铁路客运量"]=df_macro[(df_macro["indic_id"]==2160000875) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["铁路货运量"]=df_macro[(df_macro["indic_id"]==2160000883) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["广东水运"]=df_macro[(df_macro["indic_id"]==2160001002) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["港口吞吐"]=df_macro[(df_macro["indic_id"]==2160001523) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["民航客运量"]=df_macro[(df_macro["indic_id"]==2160002772) & (df_macro["PERIOD_DATE"]>='2012-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2013-06-30')]["DATA_VALUE"].mean()
		X_train_2013["label"]=df_2013["REVENUE"]


		# In[11]:


		X_train_2012=pd.DataFrame()
		##提取过去三年的年中业绩
		X_train_2012["TICKER_SYMBOL"]=df_2012["Unnamed: 0"]

		X_train_2012["last_season"]=0
		X_train_2012["REVENUE_2"]=0
		X_train_2012["REVENUE_1"]=0
		X_train_2012["REVENUE_0"]=0

		for i in tqdm(df_2012.index):
			if np.isnan(df_2012["REVENUE_11"][i]):
				X_train_2012["last_season"][i]=df_2012["REVENUE_10"][i]/4
			else:
				X_train_2012["last_season"][i]=df_2012["REVENUE_11"][i]
			if np.isnan(df_2012["REVENUE_8"][i]):
				X_train_2012["REVENUE_2"][i]=df_2012["REVENUE_10"][i]/2
			else:
				X_train_2012["REVENUE_2"][i]=df_2012["REVENUE_8"][i]
			if np.isnan(df_2012["REVENUE_4"][i]):
				X_train_2012["REVENUE_1"][i]=X_train_2012["REVENUE_2"][i]
			else:
				X_train_2012["REVENUE_1"][i]=df_2012["REVENUE_2"][i]
			if np.isnan(df_2012["REVENUE_0"][i]):
				X_train_2012["REVENUE_0"][i]=X_train_2012["REVENUE_1"][i]
			else:
				X_train_2012["REVENUE_0"][i]=df_2012["REVENUE_0"][i]

		##一季度营业额同比增长率
		X_train_2012["END_DATE"]="2012-06-30"
		X_train_2012["incre_rate"]=df_2012["REVENUE_11"]/df_2012["REVENUE_7"]
		X_train_2012["市盈率"]=df_map.loc[X_train_2012["TICKER_SYMBOL"]]["CLOSE_PRICE_2012_05"].values/df_2012["BASIC_EPS_10"].values

		##推荐值
		X_train_2012["recommend1"]=X_train_2012["incre_rate"]*X_train_2012["REVENUE_2"]
		X_train_2012["recommend2"]=X_train_2012["REVENUE_2"]/X_train_2012["REVENUE_1"]*X_train_2012["REVENUE_2"]
		X_train_2012["recommend3"]=X_train_2012["REVENUE_2"]/df_2012["T_ASSETS_8"]*(df_2012["T_ASSETS_11"]+
				                                                                    df_2012["ESTIMATED_LIAB_11"].fillna(0))

		##获取一二股东持股比例
		X_train_2012["holder1"]=df_holder_share.loc[X_train_2012["TICKER_SYMBOL"]]["holder1"].values
		X_train_2012["holder2"]=df_holder_share.loc[X_train_2012["TICKER_SYMBOL"]]["holder2"].values
		X_train_2012["contain"]=X_train_2012["holder1"]/X_train_2012["holder2"]

		## 关于股票  获取股票流通占比  获取半年股价相对于大盘的增长率
		X_train_2012["flow_share"]=df_holder_share.loc[X_train_2012["TICKER_SYMBOL"]]["flow_share"].values
		X_train_2012["market_incre"]=df_map.loc[X_train_2012["TICKER_SYMBOL"]]["market_incre_2012"].values

		### 关于负债 获取过去三年资产负债率,流动负债比例,债券负债占比,长期负债比例,速动比率
		X_train_2012["liab_ago3"]=df_2012["T_LIAB_0"]/df_2012["T_ASSETS_0"]
		X_train_2012["liab_ago2"]=df_2012["T_LIAB_4"]/df_2012["T_ASSETS_4"]
		X_train_2012["liab_ago1"]=df_2012["T_LIAB_8"]/df_2012["T_ASSETS_8"]
		X_train_2012["liab_ago_half"]=df_2012["T_LIAB_10"]/df_2012["T_ASSETS_10"]
		X_train_2012["liab_ago_last"]=df_2012["T_LIAB_11"]/df_2012["T_ASSETS_11"]
		X_train_2012["fast_rate"]=df_2012["T_CA_11"]-df_2012["INVENTORIES_11"]/df_2012["T_CL_11"]

		##研发投入
		X_train_2012["T_ASSETS"]=df_2012["T_ASSETS_11"]
		for i in tqdm(X_train_2012.index):
			if np.isnan(X_train_2012["T_ASSETS"][i]):
				X_train_2012["T_ASSETS"][i]=df_2012["T_ASSETS_10"][i]
				
		X_train_2012["每股收益增长率"]=(df_2012["BASIC_EPS_10"])/df_2012["BASIC_EPS_6"]        
		X_train_2012["利润留存率"]=(df_2012["N_INCOME_10"]-df_2012["DIV_PAYABLE_10"].fillna(0))/df_2012["N_INCOME_10"]
		X_train_2012["再投资率"]=(df_2012["N_INCOME_10"]/df_2012["T_SH_EQUITY_10"])*X_train_2012["利润留存率"]
		X_train_2012["总资产增长率"]=(df_2012["T_ASSETS_10"])/df_2012["T_ASSETS_6"]
		X_train_2012["固定资产扩张率"]=(df_2012["FIXED_ASSETS_10"])/df_2012["FIXED_ASSETS_6"] 
		X_train_2012["净利润增长率"]=(df_2012["N_INCOME_10"])/df_2012["N_INCOME_6"] 

		X_train_2012["R_D_last"]=df_2012["R_D_11"].fillna(0)
		X_train_2012["R_D_ago1"]=df_2012["R_D_10"].fillna(0)
		X_train_2012["R_D_ago2"]=df_2012["R_D_6"].fillna(0)
		X_train_2012["R_D_ago3"]=df_2012["R_D_2"].fillna(0)
		##现金比率
		X_train_2012["cash_rate"]=df_2012["CASH_C_EQUIV_11"]/df_2012["T_CL_11"]

		X_train_2012["cl_rate"]=df_2012["T_CL_11"]/df_2012["T_LIAB_11"]
		X_train_2012["ncl_rate"]=df_2012["T_NCL_11"]/df_2012["T_LIAB_11"]
		X_train_2012["bond_rate"]=df_2012["BOND_PAYABLE_11"].fillna(0)/df_2012["T_LIAB_11"]

		### 关于利润 获取净资产收益率,营业利润率,

		X_train_2012["profit_asset"]=df_2012["OPERATE_PROFIT_10"]/(df_2012["T_ASSETS_10"]-df_2012["T_LIAB_10"])## 需更改为净资产
		X_train_2012["profit_rate"]=df_2012["OPERATE_PROFIT_11"]/df_2012["REVENUE_11"] 

		##关于资产  商誉占资产比重,总资产增长率,股东权益比率,权益乘数,流动资产比率,固定动资产比率,资本周转率
		X_train_2012["goodwill_rate"]=df_2012["GOODWILL_11"].fillna(0)/df_2012["T_ASSETS_10"]
		X_train_2012["T_ASSETS_rate"]=df_2012["T_ASSETS_11"]/df_2012["T_ASSETS_7"]
		X_train_2012["pure_asset_rate"]=(df_2012["T_ASSETS_11"]-df_2012["T_LIAB_11"])/df_2012["T_ASSETS_11"]
		X_train_2012["权益乘数"]=df_2012["T_ASSETS_11"]/((df_2012["T_ASSETS_11"]-df_2012["T_LIAB_11"]))
		X_train_2012["T_CA_rate"]=df_2012["T_CA_11"]/df_2012["T_ASSETS_11"]
		X_train_2012["FIXED_ASSETS_rate"]=df_2012["FIXED_ASSETS_11"]/df_2012["T_ASSETS_11"]
		X_train_2012["资本周转率"]=(df_2012["CASH_C_EQUIV_11"]+df_2012["NOTES_RECEIV_11"].fillna(0)+
				               df_2012["TRADING_FA_11"].fillna(0))/df_2012["T_NCL_11"]
		X_train_2012["固定比率"]=df_2012["FIXED_ASSETS_11"]/(df_2012["T_ASSETS_11"]-df_2012["T_LIAB_11"])
		##添加宏观影响因素

		X_train_2012["工业增加值"]=df_macro[(df_macro["indic_id"]==1020000004) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		t1=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		t2=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2012["火电发电量_增长率"]=(t1-t2)/t1

		X_train_2012["PMI_出口"]=df_macro[(df_macro["indic_id"]==1030000014) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		X_train_2012["PMI_成品库存"]=df_macro[(df_macro["indic_id"]==1030000016) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		X_train_2012["PMI_进口"]=df_macro[(df_macro["indic_id"]==1030000018) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		X_train_2012["PMI_原材料库存"]=df_macro[(df_macro["indic_id"]==1030000020) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		X_train_2012["CPI_消费品"]=df_macro[(df_macro["indic_id"]==1040000046) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		X_train_2012["CPI"]=df_macro[(df_macro["indic_id"]==1040000050) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()


		##货币供应


		t1=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2010-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2011-12-31')]["DATA_VALUE"].values
		X_train_2012["M0_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2010-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2011-12-31')]["DATA_VALUE"].values
		X_train_2012["M1_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2010-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2011-12-31')]["DATA_VALUE"].values
		X_train_2012["M2_rate"]=(t2/t1)[0]


		###利率 汇率

		X_train_2012["美元汇率"]=df_macro[(df_macro["indic_id"]==1080000235) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["贷款利率"]=df_macro[(df_macro["indic_id"]==1090000363) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		###

		X_train_2012["电脑出口"]=df_macro[(df_macro["indic_id"]==1100000874) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["煤进口"]=df_macro[(df_macro["indic_id"]==1100002293) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["煤产量"]=df_macro[(df_macro["indic_id"]==2020100020) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["煤销量"]=df_macro[(df_macro["indic_id"]==2020100024) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		X_train_2012["火车出境"]=df_macro[(df_macro["indic_id"]==1100005542) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["原油进口量"]=df_macro[(df_macro["indic_id"]==1100006640) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["原油价格"]=df_macro[(df_macro["indic_id"]==2020000719) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["汽车产量"]=df_macro[(df_macro["indic_id"]==2070100748) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["汽车销量"]=df_macro[(df_macro["indic_id"]==2070100795) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		X_train_2012["汽车库存"]=df_macro[(df_macro["indic_id"]==2070104273) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["SUV产量"]=df_macro[(df_macro["indic_id"]==2070109977) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["SUV销量"]=df_macro[(df_macro["indic_id"]==2070113040) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		X_train_2012["液晶出货"]=df_macro[(df_macro["indic_id"]==2090100464) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		###用电情况
		X_train_2012["全社会用电量"]=df_macro[(df_macro["indic_id"]==2020101521) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["第一产业用电"]=df_macro[(df_macro["indic_id"]==2020101522) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["第二产业用电"]=df_macro[(df_macro["indic_id"]==2020101523) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["第三产业用电"]=df_macro[(df_macro["indic_id"]==2020101524) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["工业用电"]=df_macro[(df_macro["indic_id"]==2020101526) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		X_train_2012["发电企业耗煤"]=df_macro[(df_macro["indic_id"]==2020102867) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		# 考虑证券市场变化

		X_train_2012["CIP_0"]=df_2012["CIP_10"]
		X_train_2012["CIP_1"]=df_2012["CIP_6"]
		X_train_2012["CIP_2"]=df_2012["CIP_2"]

		X_train_2012["TAXES_PAYABLE"]=df_2012["TAXES_PAYABLE_11"]
		X_train_2012["AR"]=df_2012["AR_11"]/X_train_2012["last_season"]  ##应收账款占营业收入比
		X_train_2012["AR"]=df_2012["AR_11"]/(df_2012["T_ASSETS_11"]-df_2012["T_LIAB_11"]) ##应收账款占净资产比


		X_train_2012["融资融券余额"]=df_macro[(df_macro["indic_id"]==1170000018) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		X_train_2012["存管证券数量"]=df_macro[(df_macro["indic_id"]==1170000598) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["存管证券面值"]=df_macro[(df_macro["indic_id"]==1170000618) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["存管证券流通市值"]=df_macro[(df_macro["indic_id"]==1170000641) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["一级市场债券余额"]=df_macro[(df_macro["indic_id"]==1170007422) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		##房地产情况

		X_train_2012["商品房成交"]=df_macro[(df_macro["indic_id"]==2170702521) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["商品房待售面积"]=df_macro[(df_macro["indic_id"]==2170002035) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["信托收益_房地产"]=df_macro[(df_macro["indic_id"]==2210200588) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		X_train_2012["固定资产投资"]=df_macro[(df_macro["indic_id"]==1050000026) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		X_train_2012["房地产开发投资"]=df_macro[(df_macro["indic_id"]==1050000027) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()

		##物流航运
		X_train_2012["波罗的海运价"]=df_macro[(df_macro["indic_id"]==2160000004) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["出口集装箱运价"]=df_macro[(df_macro["indic_id"]==2160000101) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["公路客运量"]=df_macro[(df_macro["indic_id"]==2160000481) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["公路货运量"]=df_macro[(df_macro["indic_id"]==2160000489) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["铁路客运量"]=df_macro[(df_macro["indic_id"]==2160000875) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["铁路货运量"]=df_macro[(df_macro["indic_id"]==2160000883) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["广东水运"]=df_macro[(df_macro["indic_id"]==2160001002) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["港口吞吐"]=df_macro[(df_macro["indic_id"]==2160001523) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["民航客运量"]=df_macro[(df_macro["indic_id"]==2160002772) & (df_macro["PERIOD_DATE"]>='2011-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2012-06-30')]["DATA_VALUE"].mean()
		X_train_2012["label"]=df_2012["REVENUE"]


		# In[12]:


		X_train_2011=pd.DataFrame()
		##提取过去三年的年中业绩
		X_train_2011["TICKER_SYMBOL"]=df_2011["Unnamed: 0"]

		X_train_2011["last_season"]=0
		X_train_2011["REVENUE_2"]=0
		X_train_2011["REVENUE_1"]=0
		X_train_2011["REVENUE_0"]=0

		for i in tqdm(df_2011.index):
			if np.isnan(df_2011["REVENUE_11"][i]):
				X_train_2011["last_season"][i]=df_2011["REVENUE_10"][i]/4
			else:
				X_train_2011["last_season"][i]=df_2011["REVENUE_11"][i]
			if np.isnan(df_2011["REVENUE_8"][i]):
				X_train_2011["REVENUE_2"][i]=df_2011["REVENUE_10"][i]/2
			else:
				X_train_2011["REVENUE_2"][i]=df_2011["REVENUE_8"][i]
			if np.isnan(df_2011["REVENUE_4"][i]):
				X_train_2011["REVENUE_1"][i]=X_train_2011["REVENUE_2"][i]
			else:
				X_train_2011["REVENUE_1"][i]=df_2011["REVENUE_2"][i]
			if np.isnan(df_2011["REVENUE_0"][i]):
				X_train_2011["REVENUE_0"][i]=X_train_2011["REVENUE_1"][i]
			else:
				X_train_2011["REVENUE_0"][i]=df_2011["REVENUE_0"][i]

		##一季度营业额同比增长率
		X_train_2011["END_DATE"]="2011-06-30"
		X_train_2011["incre_rate"]=df_2011["REVENUE_11"]/df_2011["REVENUE_7"]
		X_train_2011["市盈率"]=df_map.loc[X_train_2011["TICKER_SYMBOL"]]["CLOSE_PRICE_2011_05"].values/df_2011["BASIC_EPS_10"].values

		##推荐值
		X_train_2011["recommend1"]=X_train_2011["incre_rate"]*X_train_2011["REVENUE_2"]
		X_train_2011["recommend2"]=X_train_2011["REVENUE_2"]/X_train_2011["REVENUE_1"]*X_train_2011["REVENUE_2"]
		X_train_2011["recommend3"]=X_train_2011["REVENUE_2"]/df_2011["T_ASSETS_8"]*(df_2011["T_ASSETS_11"]+
				                                                                    df_2011["ESTIMATED_LIAB_11"].fillna(0))

		##获取一二股东持股比例
		X_train_2011["holder1"]=df_holder_share.loc[X_train_2011["TICKER_SYMBOL"]]["holder1"].values
		X_train_2011["holder2"]=df_holder_share.loc[X_train_2011["TICKER_SYMBOL"]]["holder2"].values
		X_train_2011["contain"]=X_train_2011["holder1"]/X_train_2011["holder2"]

		## 关于股票  获取股票流通占比  获取半年股价相对于大盘的增长率
		X_train_2011["flow_share"]=df_holder_share.loc[X_train_2011["TICKER_SYMBOL"]]["flow_share"].values
		X_train_2011["market_incre"]=df_map.loc[X_train_2011["TICKER_SYMBOL"]]["market_incre_2011"].values

		### 关于负债 获取过去三年资产负债率,流动负债比例,债券负债占比,长期负债比例,速动比率
		X_train_2011["liab_ago3"]=df_2011["T_LIAB_0"]/df_2011["T_ASSETS_0"]
		X_train_2011["liab_ago2"]=df_2011["T_LIAB_4"]/df_2011["T_ASSETS_4"]
		X_train_2011["liab_ago1"]=df_2011["T_LIAB_8"]/df_2011["T_ASSETS_8"]
		X_train_2011["liab_ago_half"]=df_2011["T_LIAB_10"]/df_2011["T_ASSETS_10"]
		X_train_2011["liab_ago_last"]=df_2011["T_LIAB_11"]/df_2011["T_ASSETS_11"]
		X_train_2011["fast_rate"]=df_2011["T_CA_11"]-df_2011["INVENTORIES_11"]/df_2011["T_CL_11"]

		##研发投入
		X_train_2011["T_ASSETS"]=df_2011["T_ASSETS_11"]
		for i in tqdm(X_train_2011.index):
			if np.isnan(X_train_2011["T_ASSETS"][i]):
				X_train_2011["T_ASSETS"][i]=df_2011["T_ASSETS_10"][i]
				
		X_train_2011["每股收益增长率"]=(df_2011["BASIC_EPS_10"])/df_2011["BASIC_EPS_6"]        
		X_train_2011["利润留存率"]=(df_2011["N_INCOME_10"]-df_2011["DIV_PAYABLE_10"].fillna(0))/df_2011["N_INCOME_10"]
		X_train_2011["再投资率"]=(df_2011["N_INCOME_10"]/df_2011["T_SH_EQUITY_10"])*X_train_2011["利润留存率"]
		X_train_2011["总资产增长率"]=(df_2011["T_ASSETS_10"])/df_2011["T_ASSETS_6"]
		X_train_2011["固定资产扩张率"]=(df_2011["FIXED_ASSETS_10"])/df_2011["FIXED_ASSETS_6"] 
		X_train_2011["净利润增长率"]=(df_2011["N_INCOME_10"])/df_2011["N_INCOME_6"] 

		X_train_2011["R_D_last"]=df_2011["R_D_11"].fillna(0)
		X_train_2011["R_D_ago1"]=df_2011["R_D_10"].fillna(0)
		X_train_2011["R_D_ago2"]=df_2011["R_D_6"].fillna(0)
		X_train_2011["R_D_ago3"]=df_2011["R_D_2"].fillna(0)
		##现金比率
		X_train_2011["cash_rate"]=df_2011["CASH_C_EQUIV_11"]/df_2011["T_CL_11"]

		X_train_2011["cl_rate"]=df_2011["T_CL_11"]/df_2011["T_LIAB_11"]
		X_train_2011["ncl_rate"]=df_2011["T_NCL_11"]/df_2011["T_LIAB_11"]
		X_train_2011["bond_rate"]=df_2011["BOND_PAYABLE_11"].fillna(0)/df_2011["T_LIAB_11"]

		### 关于利润 获取净资产收益率,营业利润率,

		X_train_2011["profit_asset"]=df_2011["OPERATE_PROFIT_10"]/(df_2011["T_ASSETS_10"]-df_2011["T_LIAB_10"])## 需更改为净资产
		X_train_2011["profit_rate"]=df_2011["OPERATE_PROFIT_11"]/df_2011["REVENUE_11"] 

		##关于资产  商誉占资产比重,总资产增长率,股东权益比率,权益乘数,流动资产比率,固定动资产比率,资本周转率
		X_train_2011["goodwill_rate"]=df_2011["GOODWILL_11"].fillna(0)/df_2011["T_ASSETS_10"]
		X_train_2011["T_ASSETS_rate"]=df_2011["T_ASSETS_11"]/df_2011["T_ASSETS_7"]
		X_train_2011["pure_asset_rate"]=(df_2011["T_ASSETS_11"]-df_2011["T_LIAB_11"])/df_2011["T_ASSETS_11"]
		X_train_2011["权益乘数"]=df_2011["T_ASSETS_11"]/((df_2011["T_ASSETS_11"]-df_2011["T_LIAB_11"]))
		X_train_2011["T_CA_rate"]=df_2011["T_CA_11"]/df_2011["T_ASSETS_11"]
		X_train_2011["FIXED_ASSETS_rate"]=df_2011["FIXED_ASSETS_11"]/df_2011["T_ASSETS_11"]
		X_train_2011["资本周转率"]=(df_2011["CASH_C_EQUIV_11"]+df_2011["NOTES_RECEIV_11"].fillna(0)+
				               df_2011["TRADING_FA_11"].fillna(0))/df_2011["T_NCL_11"]
		X_train_2011["固定比率"]=df_2011["FIXED_ASSETS_11"]/(df_2011["T_ASSETS_11"]-df_2011["T_LIAB_11"])
		##添加宏观影响因素

		X_train_2011["工业增加值"]=df_macro[(df_macro["indic_id"]==1020000004) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		t1=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		t2=df_macro[(df_macro["indic_id"]==1020001544) & (df_macro["PERIOD_DATE"]>='2009-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2010-06-30')]["DATA_VALUE"].mean()
		X_train_2011["火电发电量_增长率"]=(t1-t2)/t1

		X_train_2011["PMI_出口"]=df_macro[(df_macro["indic_id"]==1030000014) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		X_train_2011["PMI_成品库存"]=df_macro[(df_macro["indic_id"]==1030000016) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		X_train_2011["PMI_进口"]=df_macro[(df_macro["indic_id"]==1030000018) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		X_train_2011["PMI_原材料库存"]=df_macro[(df_macro["indic_id"]==1030000020) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		X_train_2011["CPI_消费品"]=df_macro[(df_macro["indic_id"]==1040000046) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		X_train_2011["CPI"]=df_macro[(df_macro["indic_id"]==1040000050) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()


		##货币供应


		t1=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2009-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000033) & (df_macro["PERIOD_DATE"]=='2010-12-31')]["DATA_VALUE"].values
		X_train_2011["M0_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2009-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000035) & (df_macro["PERIOD_DATE"]=='2010-12-31')]["DATA_VALUE"].values
		X_train_2011["M1_rate"]=(t2/t1)[0]

		t1=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2009-12-31')]["DATA_VALUE"].values
		t2=df_macro[(df_macro["indic_id"]==1070000039) & (df_macro["PERIOD_DATE"]=='2010-12-31')]["DATA_VALUE"].values
		X_train_2011["M2_rate"]=(t2/t1)[0]


		###利率 汇率

		X_train_2011["美元汇率"]=df_macro[(df_macro["indic_id"]==1080000235) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["贷款利率"]=df_macro[(df_macro["indic_id"]==1090000363) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		###

		X_train_2011["电脑出口"]=df_macro[(df_macro["indic_id"]==1100000874) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["煤进口"]=df_macro[(df_macro["indic_id"]==1100002293) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["煤产量"]=df_macro[(df_macro["indic_id"]==2020100020) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["煤销量"]=df_macro[(df_macro["indic_id"]==2020100024) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		X_train_2011["火车出境"]=df_macro[(df_macro["indic_id"]==1100005542) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["原油进口量"]=df_macro[(df_macro["indic_id"]==1100006640) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["原油价格"]=df_macro[(df_macro["indic_id"]==2020000719) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["汽车产量"]=df_macro[(df_macro["indic_id"]==2070100748) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["汽车销量"]=df_macro[(df_macro["indic_id"]==2070100795) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		X_train_2011["汽车库存"]=df_macro[(df_macro["indic_id"]==2070104273) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["SUV产量"]=df_macro[(df_macro["indic_id"]==2070109977) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["SUV销量"]=df_macro[(df_macro["indic_id"]==2070113040) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		X_train_2011["液晶出货"]=df_macro[(df_macro["indic_id"]==2090100464) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		###用电情况
		X_train_2011["全社会用电量"]=df_macro[(df_macro["indic_id"]==2020101521) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["第一产业用电"]=df_macro[(df_macro["indic_id"]==2020101522) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["第二产业用电"]=df_macro[(df_macro["indic_id"]==2020101523) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["第三产业用电"]=df_macro[(df_macro["indic_id"]==2020101524) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["工业用电"]=df_macro[(df_macro["indic_id"]==2020101526) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		X_train_2011["发电企业耗煤"]=df_macro[(df_macro["indic_id"]==2020102867) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		# 考虑证券市场变化

		X_train_2011["CIP_0"]=df_2011["CIP_10"]
		X_train_2011["CIP_1"]=df_2011["CIP_6"]
		X_train_2011["CIP_2"]=df_2011["CIP_2"]

		X_train_2011["TAXES_PAYABLE"]=df_2011["TAXES_PAYABLE_11"]
		X_train_2011["AR"]=df_2011["AR_11"]/X_train_2011["last_season"]  ##应收账款占营业收入比
		X_train_2011["AR"]=df_2011["AR_11"]/(df_2011["T_ASSETS_11"]-df_2011["T_LIAB_11"]) ##应收账款占净资产比


		X_train_2011["融资融券余额"]=df_macro[(df_macro["indic_id"]==1170000018) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		X_train_2011["存管证券数量"]=df_macro[(df_macro["indic_id"]==1170000598) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["存管证券面值"]=df_macro[(df_macro["indic_id"]==1170000618) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["存管证券流通市值"]=df_macro[(df_macro["indic_id"]==1170000641) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["一级市场债券余额"]=df_macro[(df_macro["indic_id"]==1170007422) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		###房地产交易情况

		X_train_2011["商品房成交"]=df_macro[(df_macro["indic_id"]==2170702521) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["商品房待售面积"]=df_macro[(df_macro["indic_id"]==2170002035) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["信托收益_房地产"]=df_macro[(df_macro["indic_id"]==2210200588) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		X_train_2011["固定资产投资"]=df_macro[(df_macro["indic_id"]==1050000026) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		X_train_2011["房地产开发投资"]=df_macro[(df_macro["indic_id"]==1050000027) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()

		##物流航运
		X_train_2011["波罗的海运价"]=df_macro[(df_macro["indic_id"]==2160000004) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["出口集装箱运价"]=df_macro[(df_macro["indic_id"]==2160000101) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["公路客运量"]=df_macro[(df_macro["indic_id"]==2160000481) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["公路货运量"]=df_macro[(df_macro["indic_id"]==2160000489) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["铁路客运量"]=df_macro[(df_macro["indic_id"]==2160000875) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["铁路货运量"]=df_macro[(df_macro["indic_id"]==2160000883) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["广东水运"]=df_macro[(df_macro["indic_id"]==2160001002) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["港口吞吐"]=df_macro[(df_macro["indic_id"]==2160001523) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["民航客运量"]=df_macro[(df_macro["indic_id"]==2160002772) & (df_macro["PERIOD_DATE"]>='2010-12-31'
				                                      ) & (df_macro["PERIOD_DATE"]<'2011-06-30')]["DATA_VALUE"].mean()
		X_train_2011["label"]=df_2011["REVENUE"]


		# In[14]:


		df_train=pd.concat([X_train_2011,X_train_2012,X_train_2013,X_train_2014,X_train_2015,
				  X_train_2016,X_train_2017],axis=0)


		# In[15]:


		df_train.to_csv("../data/train_all.csv",index=False)
		X_train_2018.to_csv("../data/test_all.csv",index=False)


		# ###000776,000750,000712,证券  600816应该是银行,应该从此内容中剔除
		# ##indu中的 000563去哪里了?  由于000563的收入表 写在了bank模块里  所以造成了数据丢失,需要进行处理

		# In[16]:


		df_train=pd.read_csv("../data/train_all.csv")
		df_test=pd.read_csv("../data/test_all.csv")


		# In[17]:


		df_all=pd.concat([df_train,df_test],axis=0).reset_index().drop("index",axis=1)


		# ### 根据各自特点填充缺失值,均值填充的时候按照type分组后再填充

		# 去掉没么有market信息的股票

		# In[18]:


		## print(set(df_all["TICKER_SYMBOL"].unique())-set(df_map.index))
		df_all=df_all[(df_all["TICKER_SYMBOL"]!=2720) & (df_all["TICKER_SYMBOL"]!=300361)&(df_all["TICKER_SYMBOL"]!=300728)
				      &(df_all["TICKER_SYMBOL"]!=601206)&(df_all["TICKER_SYMBOL"]!=603302) ]


		# In[19]:


		temp=[]
		for i in df_all.index:
			ticker=df_all.loc[i,"TICKER_SYMBOL"]
			temp.append(df_map.loc[ticker,"TYPE_NAME_CN"])
		df_all["type"]=temp

		## 防止出现除以零的情况,将无穷大替换为未知
		df_all=df_all.replace({np.inf:np.nan,-np.inf:np.nan})


		# ### 由于xgboost 与lightgbm可以自动处理缺失值,先不处理(此处注释掉)

		# In[20]:


		# ## 净资产利润率用当年行业均值填充
		# df_all["profit_asset"]=df_all.groupby(['END_DATE',"type"])["profit_asset"].transform(lambda x: x.fillna(x.mean()))
		# ## 总资产增长率用当年行业均值填充
		# df_all["T_ASSETS_rate"]=df_all.groupby(['END_DATE',"type"])["T_ASSETS_rate"].transform(lambda x: x.fillna(x.mean()))
		# ## 股价增长相对幅度用当年行业均值填充
		# df_all["market_incre"]=df_all.groupby(['END_DATE',"type"])["market_incre"].transform(lambda x: x.fillna(x.mean()))
		# ## 所有者权益比率用当年行业均值填充
		# df_all["pure_asset_rate"]=df_all.groupby(['END_DATE',"type"])["pure_asset_rate"].transform(lambda x: x.fillna(x.mean()))
		#  ## 营业利润率,用当年行业均值填充
		# df_all["profit_rate"]=df_all.groupby(['END_DATE',"type"])["profit_rate"].transform(lambda x: x.fillna(x.mean()))
		# ## 增长率用当年行业均值填充
		# df_all["incre_rate"]=df_all.groupby(['END_DATE',"type"])["incre_rate"].transform(lambda x: x.fillna(x.mean()))  ##
		# ## 非流动负债比率当年行业均值填充
		# df_all["ncl_rate"]=df_all.groupby(['END_DATE',"type"])["ncl_rate"].transform(lambda x: x.fillna(x.mean()))  ##
		# df_all["ncl_rate"]=df_all["ncl_rate"].fillna(df_all["ncl_rate"].mean())
		# ## 流动负债比率当年行业均值填充
		# df_all["cl_rate"]=df_all.groupby(['END_DATE',"type"])["cl_rate"].transform(lambda x: x.fillna(x.mean()))  ##
		# df_all["cl_rate"]=df_all["cl_rate"].fillna(df_all["cl_rate"].mean())
		# ## 速动比率当年行业均值填充
		# df_all["fast_rate"]=df_all.groupby(['END_DATE',"type"])["fast_rate"].transform(lambda x: x.fillna(x.mean()))  ##
		# df_all["fast_rate"]=df_all["fast_rate"].fillna(df_all["fast_rate"].mean())
		# ## 现金比率当年行业均值填充
		# df_all["cash_rate"]=df_all.groupby(['END_DATE',"type"])["cash_rate"].transform(lambda x: x.fillna(x.mean()))  ##
		# df_all["cash_rate"]=df_all["cash_rate"].fillna(df_all["cash_rate"].mean())
		# ## 流动资产比率当年行业均值填充
		# df_all["T_CA_rate"]=df_all.groupby(['END_DATE',"type"])["T_CA_rate"].transform(lambda x: x.fillna(x.mean()))  ##
		# df_all["T_CA_rate"]=df_all["T_CA_rate"].fillna(df_all["T_CA_rate"].mean())
		# ## 固定占净资产比率当年行业均值填充
		# df_all["固定比率"]=df_all.groupby(['END_DATE',"type"])["固定比率"].transform(lambda x: x.fillna(x.mean()))  ##
		# ## 固定占总资产比率当年行业均值填充
		# df_all["FIXED_ASSETS_rate"]=df_all.groupby(['END_DATE',"type"])["FIXED_ASSETS_rate"].transform(lambda x: x.fillna(x.mean()))  ##
		# ## 应收账款比率当年行业均值填充
		# df_all["AR"]=df_all.groupby(['END_DATE',"type"])["AR"].transform(lambda x: x.fillna(x.mean()))  ##
		# ## 债券比率当年行业均值填充
		# df_all["bond_rate"]=df_all.groupby(['END_DATE',"type"])["bond_rate"].transform(lambda x: x.fillna(x.mean()))  ##
		# ## 应缴税费当年行业均值填充
		# df_all["TAXES_PAYABLE"]=df_all.groupby(['END_DATE',"type"])["TAXES_PAYABLE"].transform(lambda x: x.fillna(x.mean()))
		# ## 权益乘数当年行业均值填充
		# df_all["权益乘数"]=df_all.groupby(['END_DATE',"type"])["权益乘数"].transform(lambda x: x.fillna(x.mean()))
		# ## 利润率当年行业均值填充
		# df_all["profit_rate"]=df_all.groupby(['END_DATE',"type"])["profit_rate"].transform(lambda x: x.fillna(x.mean()))
		# ## 商誉占比当年行业均值填充
		# df_all["goodwill_rate"]=df_all.groupby(['END_DATE',"type"])["goodwill_rate"].transform(lambda x: x.fillna(x.mean()))
		# ## 上季度值营业收入当年行业均值填充(只有三个)
		# df_all["last_season"]=df_all.groupby(['END_DATE',"type"])["last_season"].transform(lambda x: x.fillna(x.mean()))
		# ## 上季度值营业收入当年行业均值填充(只有三个)
		# df_all["资本周转率"]=df_all.groupby(['END_DATE',"type"])["资本周转率"].transform(lambda x: x.fillna(x.mean()))
		# df_all["资本周转率"]=df_all["资本周转率"].fillna(df_all["资本周转率"].mean())
		# ## 在建工程当年行业均值填充
		# df_all["CIP_2"]=df_all.groupby(['END_DATE',"type"])["CIP_2"].transform(lambda x: x.fillna(x.mean()))
		# df_all["CIP_1"]=df_all.groupby(['END_DATE',"type"])["CIP_1"].transform(lambda x: x.fillna(x.mean()))
		# df_all["CIP_0"]=df_all.groupby(['END_DATE',"type"])["CIP_0"].transform(lambda x: x.fillna(x.mean()))

		# ## 营业收入当年行业均值填充
		# df_all["REVENUE_2"]=df_all.groupby(['END_DATE',"type"])["REVENUE_2"].transform(lambda x: x.fillna(x.mean()))
		# df_all["REVENUE_1"]=df_all.groupby(['END_DATE',"type"])["REVENUE_1"].transform(lambda x: x.fillna(x.mean()))
		# df_all["REVENUE_0"]=df_all.groupby(['END_DATE',"type"])["REVENUE_0"].transform(lambda x: x.fillna(x.mean()))

		# df_all["T_ASSETS"]=df_all.groupby(['END_DATE',"type"])["T_ASSETS"].transform(lambda x: x.fillna(x.mean()))
		# df_all["再投资率"]=df_all.groupby(['END_DATE',"type"])["再投资率"].transform(lambda x: x.fillna(x.mean()))
		# df_all["资本周转率"]=df_all["资本周转率"].fillna(df_all["资本周转率"].mean())
		# df_all["利润留存率"]=df_all.groupby(['END_DATE',"type"])["利润留存率"].transform(lambda x: x.fillna(x.mean()))
		# df_all["资本周转率"]=df_all["资本周转率"].fillna(df_all["资本周转率"].mean())
		# df_all["每股收益增长率"]=df_all.groupby(['END_DATE',"type"])["每股收益增长率"].transform(lambda x: x.fillna(x.mean()))
		# df_all["总资产增长率"]=df_all.groupby(['END_DATE',"type"])["总资产增长率"].transform(lambda x: x.fillna(x.mean()))
		# df_all["固定资产扩张率"]=df_all.groupby(['END_DATE',"type"])["固定资产扩张率"].transform(lambda x: x.fillna(x.mean()))
		# df_all["净利润增长率"]=df_all.groupby(['END_DATE',"type"])["净利润增长率"].transform(lambda x: x.fillna(x.mean()))

		# ##宏观指标用均值填充
		# df_all["商品房成交"]=df_all["商品房成交"].fillna(df_all["商品房成交"].mean())
		# df_all["SUV产量"]=df_all["SUV产量"].fillna(df_all["SUV产量"].mean())
		# df_all["SUV销量"]=df_all["SUV销量"].fillna(df_all["SUV销量"].mean())
		# df_all["原油进口量"]=df_all["原油进口量"].fillna(df_all["原油进口量"].mean())
		# df_all["汽车库存"]=df_all["汽车库存"].fillna(df_all["汽车库存"].mean())
		# df_all["液晶出货"]=df_all["液晶出货"].fillna(df_all["液晶出货"].mean())
		# df_all["火车出境"]=df_all["火车出境"].fillna(df_all["火车出境"].mean())
		# df_all["商品房待售面积"]=df_all["商品房待售面积"].fillna(df_all["商品房待售面积"].mean())


		##重要指标用上年的指标进行填充

		# df_all['liab_ago_half']=df_all.groupby(['END_DATE',"type"])["liab_ago_half"].transform(lambda x: x.fillna(x.mean())) ##半年前资产负债率,均值填充
		# df_all["recommend2"]=df_all.groupby(['END_DATE',"type"])["recommend2"].transform(lambda x: x.fillna(x.mean()))
		# for i in tqdm(df_all.index):
		#     if np.isnan(df_all['liab_ago_last'][i]):
		#         df_all['liab_ago_last'][i]=df_all['liab_ago_half'][i]
		#     if np.isnan(df_all['liab_ago1'][i]):
		#         df_all['liab_ago1']=df_all['liab_ago_half'][i]
		#     if np.isnan(df_all['liab_ago2'][i]):
		#         df_all['liab_ago2']=df_all['liab_ago1'][i]
		#     if np.isnan(df_all['liab_ago3'][i]):
		#         df_all['liab_ago3']=df_all['liab_ago2'][i]
				
				
		# for i in tqdm(df_all.index):
		#     if np.isnan(df_all["recommend1"][i]):
		#         df_all["recommend1"][i]=df_all["recommend2"][i]
		#     if np.isnan(df_all["recommend3"][i]):
		#         df_all["recommend3"][i]=df_all["recommend2"][i]
				


		# In[21]:


		training_data=df_all[df_all["label"].notnull()]
		testing_data=df_all[df_all["label"].isnull()]


		# In[22]:


		training_data.to_csv("../data/train_all_na_done.csv",index=False)
		testing_data.to_csv("../data/test_all_na_done.csv",index=False)

