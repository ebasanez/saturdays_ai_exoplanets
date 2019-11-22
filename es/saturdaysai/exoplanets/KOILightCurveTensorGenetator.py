import pandas as pd
from lightKurveApi.lightKurveClient import LightKurveClient 

class KOILightCurveTensorGenerator:

	DEFAULT_GLOBAL_LEN = 2001 	
	DEFAULT_LOCAL_LEN = 201
	DEFAULT_LOCAL_VIEW_WIDTH = 4 
	
	def __init__(self, sourceFileName, global_tensor_len = DEFAULT_GLOBAL_LEN, local_tensor_len = DEFAULT_LOCAL_LEN, local_view_witdh = DEFAULT_LOCAL_VIEW_WIDTH):
		self.df = pd.read_csv(sourceFileName)
		self.global_tensor_len = global_tensor_len
		self.local_tensor_len = local_tensor_len
		
	def getTensors(self, window):
		print(window)
		df = self.df.loc[window[0]:window[1]]
		lkClient = LightKurveClient()
		for index, row in df.iterrows():
			print(row)
			mission = row.mission
			koi_id = row.koi_id
			duration = row.koi_duration
			period = row.koi_period
			foldedLightCurveDataFrames = lkClient.getKOILightKurve(koi_id, mission = mission, fold_period = period)
			print(f"Obtained {len(foldedLightCurveDataFrames)} folds. Creating local views:")
			for flc in foldedLightCurveDataFrames:
				llc = createLocalView(flc, duration, period)	
				print(f"Generated local view with {len(llc)} items for period with {len(flc)} items")
				
	def createLocalView(self, flc, duration, period):
		# With of period in folded, multiplied by DEFAULT_LOCAL_VIEW_WIDTH to also include points close to transit init
		semiPeriod = duration /(period * 2) * local_view_witdh
		return flcDataFrame.loc[(flcDataFrame['time'] >= -semiPeriod) & (flc['time'] <= semiPeriod)]
	

# Execute only if script run standalone (not imported)	
if __name__ == '__main__':
	import sys

	(script, sourceFileName, window_init, window_end) = sys.argv
	tensorGenerator = KOILightCurveTensorGenerator(sourceFileName)
	tensorGenerator.getTensors([int(window_init),int(window_end)])
	