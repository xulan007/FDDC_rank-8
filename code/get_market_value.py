

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile


def Get_Market_Value():
	all_zip = zipfile.ZipFile('../data/[Add June July] FDDC_financial_data_20180711.zip','r')
	all_zip.extractall(path=r"../data")
	all_zip.close()

	df_market=pd.read_excel('../data/[Add June July] FDDC_financial_data_20180711/[New] Market Data_20180711.xlsx',sheet_name=[0],parse_dates=['END_DATE_'])[0]

	df_map=df_market[df_market["END_DATE_"]=="2018-05-31"][['TICKER_SYMBOL',"MARKET_VALUE","TYPE_NAME_CN"]]
	df_map["MARKET_VALUE"]=df_map["MARKET_VALUE"]/100000000
	df_map=df_map.set_index("TICKER_SYMBOL")

	###希望消除大盘影响,获取每个股票跑赢大盘的情况
	temp21=df_market[df_market["END_DATE_"]=="2018-05-31"].set_index("TICKER_SYMBOL")
	temp21["CLOSE_PRICE_2018_05"]=temp21.pop("CLOSE_PRICE")
	temp22=df_market[df_market["END_DATE_"]=="2017-11-30"].set_index("TICKER_SYMBOL")
	temp22["CLOSE_PRICE_2017_11"]=temp22.pop("CLOSE_PRICE")
	temp23=df_market[df_market["END_DATE_"]=="2017-05-31"].set_index("TICKER_SYMBOL")
	temp23["CLOSE_PRICE_2017_05"]=temp23.pop("CLOSE_PRICE")
	temp24=df_market[df_market["END_DATE_"]=="2016-11-30"].set_index("TICKER_SYMBOL")
	temp24["CLOSE_PRICE_2016_11"]=temp24.pop("CLOSE_PRICE")
	temp25=df_market[df_market["END_DATE_"]=="2016-05-31"].set_index("TICKER_SYMBOL")
	temp25["CLOSE_PRICE_2016_05"]=temp25.pop("CLOSE_PRICE")
	temp26=df_market[df_market["END_DATE_"]=="2015-11-30"].set_index("TICKER_SYMBOL")
	temp26["CLOSE_PRICE_2015_11"]=temp26.pop("CLOSE_PRICE")
	temp27=df_market[df_market["END_DATE_"]=="2015-05-29"].set_index("TICKER_SYMBOL")
	temp27["CLOSE_PRICE_2015_05"]=temp27.pop("CLOSE_PRICE")
	temp28=df_market[df_market["END_DATE_"]=="2014-11-28"].set_index("TICKER_SYMBOL")
	temp28["CLOSE_PRICE_2014_11"]=temp28.pop("CLOSE_PRICE")
	temp29=df_market[df_market["END_DATE_"]=="2014-05-30"].set_index("TICKER_SYMBOL")
	temp29["CLOSE_PRICE_2014_05"]=temp29.pop("CLOSE_PRICE")
	temp210=df_market[df_market["END_DATE_"]=="2013-11-29"].set_index("TICKER_SYMBOL")
	temp210["CLOSE_PRICE_2013_11"]=temp210.pop("CLOSE_PRICE")
	temp211=df_market[df_market["END_DATE_"]=="2013-05-31"].set_index("TICKER_SYMBOL")
	temp211["CLOSE_PRICE_2013_05"]=temp211.pop("CLOSE_PRICE")
	temp212=df_market[df_market["END_DATE_"]=="2012-11-30"].set_index("TICKER_SYMBOL")
	temp212["CLOSE_PRICE_2012_11"]=temp212.pop("CLOSE_PRICE")
	temp213=df_market[df_market["END_DATE_"]=="2012-05-31"].set_index("TICKER_SYMBOL")
	temp213["CLOSE_PRICE_2012_05"]=temp213.pop("CLOSE_PRICE")
	temp214=df_market[df_market["END_DATE_"]=="2011-11-30"].set_index("TICKER_SYMBOL")
	temp214["CLOSE_PRICE_2011_11"]=temp214.pop("CLOSE_PRICE")
	temp215=df_market[df_market["END_DATE_"]=="2011-05-31"].set_index("TICKER_SYMBOL")
	temp215["CLOSE_PRICE_2011_05"]=temp215.pop("CLOSE_PRICE")
	temp216=df_market[df_market["END_DATE_"]=="2010-11-30"].set_index("TICKER_SYMBOL")
	temp216["CLOSE_PRICE_2010_11"]=temp216.pop("CLOSE_PRICE")


	df_temp2=pd.concat([temp216["CLOSE_PRICE_2010_11"],
		       temp215["CLOSE_PRICE_2011_05"],temp214["CLOSE_PRICE_2011_11"],
		       temp213["CLOSE_PRICE_2012_05"],temp212["CLOSE_PRICE_2012_11"],
		       temp211["CLOSE_PRICE_2013_05"],temp210["CLOSE_PRICE_2013_11"],
		       temp29["CLOSE_PRICE_2014_05"],temp28["CLOSE_PRICE_2014_11"],
		       temp27["CLOSE_PRICE_2015_05"],temp26["CLOSE_PRICE_2015_11"],
		       temp25["CLOSE_PRICE_2016_05"],temp24["CLOSE_PRICE_2016_11"],
		       temp23["CLOSE_PRICE_2017_05"],temp22["CLOSE_PRICE_2017_11"],
		       temp21["CLOSE_PRICE_2018_05"]],axis=1)


	# In[5]:


	###希望消除大盘影响,获取每个股票跑赢大盘的情况
	temp1=df_market[df_market["END_DATE_"]=="2018-05-31"].set_index("TICKER_SYMBOL")
	temp1["MARKET_VALUE_2018_05"]=temp1.pop("MARKET_VALUE")
	temp2=df_market[df_market["END_DATE_"]=="2017-11-30"].set_index("TICKER_SYMBOL")
	temp2["MARKET_VALUE_2017_11"]=temp2.pop("MARKET_VALUE")
	temp3=df_market[df_market["END_DATE_"]=="2017-05-31"].set_index("TICKER_SYMBOL")
	temp3["MARKET_VALUE_2017_05"]=temp3.pop("MARKET_VALUE")
	temp4=df_market[df_market["END_DATE_"]=="2016-11-30"].set_index("TICKER_SYMBOL")
	temp4["MARKET_VALUE_2016_11"]=temp4.pop("MARKET_VALUE")
	temp5=df_market[df_market["END_DATE_"]=="2016-05-31"].set_index("TICKER_SYMBOL")
	temp5["MARKET_VALUE_2016_05"]=temp5.pop("MARKET_VALUE")
	temp6=df_market[df_market["END_DATE_"]=="2015-11-30"].set_index("TICKER_SYMBOL")
	temp6["MARKET_VALUE_2015_11"]=temp6.pop("MARKET_VALUE")
	temp7=df_market[df_market["END_DATE_"]=="2015-05-29"].set_index("TICKER_SYMBOL")
	temp7["MARKET_VALUE_2015_05"]=temp7.pop("MARKET_VALUE")
	temp8=df_market[df_market["END_DATE_"]=="2014-11-28"].set_index("TICKER_SYMBOL")
	temp8["MARKET_VALUE_2014_11"]=temp8.pop("MARKET_VALUE")
	temp9=df_market[df_market["END_DATE_"]=="2014-05-30"].set_index("TICKER_SYMBOL")
	temp9["MARKET_VALUE_2014_05"]=temp9.pop("MARKET_VALUE")
	temp10=df_market[df_market["END_DATE_"]=="2013-11-29"].set_index("TICKER_SYMBOL")
	temp10["MARKET_VALUE_2013_11"]=temp10.pop("MARKET_VALUE")
	temp11=df_market[df_market["END_DATE_"]=="2013-05-31"].set_index("TICKER_SYMBOL")
	temp11["MARKET_VALUE_2013_05"]=temp11.pop("MARKET_VALUE")
	temp12=df_market[df_market["END_DATE_"]=="2012-11-30"].set_index("TICKER_SYMBOL")
	temp12["MARKET_VALUE_2012_11"]=temp12.pop("MARKET_VALUE")
	temp13=df_market[df_market["END_DATE_"]=="2012-05-31"].set_index("TICKER_SYMBOL")
	temp13["MARKET_VALUE_2012_05"]=temp13.pop("MARKET_VALUE")
	temp14=df_market[df_market["END_DATE_"]=="2011-11-30"].set_index("TICKER_SYMBOL")
	temp14["MARKET_VALUE_2011_11"]=temp14.pop("MARKET_VALUE")
	temp15=df_market[df_market["END_DATE_"]=="2011-05-31"].set_index("TICKER_SYMBOL")
	temp15["MARKET_VALUE_2011_05"]=temp15.pop("MARKET_VALUE")
	temp16=df_market[df_market["END_DATE_"]=="2010-11-30"].set_index("TICKER_SYMBOL")
	temp16["MARKET_VALUE_2010_11"]=temp16.pop("MARKET_VALUE")


	temp=pd.concat([temp16["MARKET_VALUE_2010_11"],
		       temp15["MARKET_VALUE_2011_05"],temp14["MARKET_VALUE_2011_11"],
		       temp13["MARKET_VALUE_2012_05"],temp12["MARKET_VALUE_2012_11"],
		       temp11["MARKET_VALUE_2013_05"],temp10["MARKET_VALUE_2013_11"],
		       temp9["MARKET_VALUE_2014_05"],temp8["MARKET_VALUE_2014_11"],
		       temp7["MARKET_VALUE_2015_05"],temp6["MARKET_VALUE_2015_11"],
		       temp5["MARKET_VALUE_2016_05"],temp4["MARKET_VALUE_2016_11"],
		       temp3["MARKET_VALUE_2017_05"],temp2["MARKET_VALUE_2017_11"],
		       temp1["MARKET_VALUE_2018_05"]],axis=1)
	temp["market_incre_2018"]=(temp["MARKET_VALUE_2018_05"]-temp["MARKET_VALUE_2017_11"])/temp["MARKET_VALUE_2017_11"]
	temp["market_incre_2017"]=(temp["MARKET_VALUE_2017_05"]-temp["MARKET_VALUE_2016_11"])/temp["MARKET_VALUE_2016_11"]
	temp["market_incre_2016"]=(temp["MARKET_VALUE_2016_05"]-temp["MARKET_VALUE_2015_11"])/temp["MARKET_VALUE_2015_11"]
	temp["market_incre_2015"]=(temp["MARKET_VALUE_2015_05"]-temp["MARKET_VALUE_2014_11"])/temp["MARKET_VALUE_2014_11"]
	temp["market_incre_2014"]=(temp["MARKET_VALUE_2014_05"]-temp["MARKET_VALUE_2013_11"])/temp["MARKET_VALUE_2013_11"]
	temp["market_incre_2013"]=(temp["MARKET_VALUE_2013_05"]-temp["MARKET_VALUE_2012_11"])/temp["MARKET_VALUE_2012_11"]
	temp["market_incre_2012"]=(temp["MARKET_VALUE_2012_05"]-temp["MARKET_VALUE_2011_11"])/temp["MARKET_VALUE_2011_11"]
	temp["market_incre_2011"]=(temp["MARKET_VALUE_2011_05"]-temp["MARKET_VALUE_2010_11"])/temp["MARKET_VALUE_2010_11"]


	# In[6]:


	df_map=pd.concat([df_map,
		              temp["market_incre_2018"]-temp["market_incre_2018"].mean(),
		              temp["market_incre_2017"]-temp["market_incre_2017"].mean(),
		              temp["market_incre_2016"]-temp["market_incre_2016"].mean(),
		              temp["market_incre_2015"]-temp["market_incre_2015"].mean(),
		              temp["market_incre_2014"]-temp["market_incre_2014"].mean(),
		              temp["market_incre_2013"]-temp["market_incre_2013"].mean(),
		              temp["market_incre_2012"]-temp["market_incre_2012"].mean(),
		              temp["market_incre_2011"]-temp["market_incre_2011"].mean(),
		              df_temp2],axis=1)


	# In[7]:


	df_map.to_csv("../data/maket_value_type_map.csv")


# ### 以下代码为从网易财经获取各个公司流动股比和一二股东持股比例的代码
# (运行时间长,不再运行)

# In[8]:


# import time
# holder1=[]
# holder2=[]
# flow_share=[]

# for i in tqdm(df_map.reset_index()["TICKER_SYMBOL"]):
#     ticker=(6-len(str(i)))*"0"+str(i)
#     dfs=pd.read_html("http://quotes.money.163.com/f10/gdfx_{}.html#01d02".format(ticker))
#     ##第一二股东持股比例
#     try:
#         temp1=dfs[6].loc[0,"持有比例"].split("%")[0]
#         temp2=dfs[6].loc[1,"持有比例"].split("%")[0]
#         holder1.append(temp1)
#         holder2.append(temp2)
#     except:
#         holder1.append(dfs[5].loc[0,"持有比例"].split("%")[0])
#         holder2.append(dfs[5].loc[1,"持有比例"].split("%")[0])
        
#     ##流通股占比(A股,B股,H股)
#     flow_share.append(float(dfs[4].loc[1,2].split("%")[0])+float(dfs[4].loc[3,2].split("%")[0])+
#                       float(dfs[4].loc[4,2].split("%")[0]))
#     time.sleep(5)

# df_holder=pd.DataFrame({"TICKER_SYMBOL":df_map["TICKER_SYMBOL"],
#                        "holder1":holder1,
#                        "holder2":holder2,
#                        "flow_share":flow_share})

# df_holder.to_csv("../data/holder_share.csv",index=False)

