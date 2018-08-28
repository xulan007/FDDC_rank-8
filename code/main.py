from get_market_value import Get_Market_Value
from preprocess_3sheets import Preprocess_3Sheets
from FE import Feature_Engineer
from train_and_predict import Train_Model

def main():
	print("##############################################################")
	print("Loading Data...")
	#Get_Market_Value()
	print("##############################################################")
	print("Preprocessing 3 sheets......")
	#Preprocess_3Sheets()
	print("##############################################################")
	print("Feature Engineering......")
	#Feature_Engineer()
	print("##############################################################")
	print("Trainging model.....")
	Train_Model()
	print("All Done!!!")
	
	
if __name__ == '__main__':
  main()
  

	
