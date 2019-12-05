import pandas as pd

class KOIMaxCalculator:
	
	def __init__(self, sourceFileName):
		self.sourceFileName = sourceFileName
		
	def extractMaxPeriodAndDuration(self):
		df = pd.read_csv(self.sourceFileName)
		print(df.max())
		#return (df['koi_period'].max(), df['koi_duration'].max())
				
	
					
# Execute only if script run standalone (not imported)	
if __name__ == '__main__':
	import sys
	(script, sourceFileName) = sys.argv
	extractor = KOIMaxCalculator(sourceFileName)
	(period,duration) = extractor.extractMaxPeriodAndDuration()
	print(period,duration)