import os
import sys
import re

class KOIFeatureExtractor:	
	
	COLUMN_ID = 0
	COLUMN_LABEL = 3
	COLUMN_PERIOD = 10
	COLUMN_T0 = 13
	COLUMN_DURATION = 19
	
	LABEL_TRUE = 'CONFIRMED'
	LABEL_FALSE = 'FALSE POSITIVE'
	VALID_LABELS = [LABEL_TRUE, LABEL_FALSE]
	
	def __init__(self, sourceFileName):
		self.sourceFileName = sourceFileName
 	
	def extractFeatures(self, destinationFileName):
		print("Extracting features in dataset ", self.sourceFileName, " to file ", destinationFileName)
		self.deleteFile(destinationFileName) # Delete destination file to be able to recreate it
		with open(destinationFileName,'a+') as destinationFile:
			# Header row for generated file
			destinationFile.write("mission, koi_id, koi_time0bk, koi_period, koi_duration, koi_is_planet\n") 
			with open(self.sourceFileName,'r') as sourceFile:
				# Skip header row
				next(sourceFile) 
				# Read each line in dataset
				for line in sourceFile:
					lineItems = line.split(',')
					koiId = lineItems[self.COLUMN_ID]
					koiLabel = lineItems[self.COLUMN_LABEL]
					koiPeriod = float(lineItems[self.COLUMN_PERIOD])
					koiT0 = float(lineItems[self.COLUMN_T0])
					koiDuration = float(lineItems[self.COLUMN_DURATION])/24
					# We are only interested in KOIs with CONFIRMED of FALSE POSITIVE TCEs
					if koiLabel in self.VALID_LABELS: 
						destinationFile.write('Kepler, %s,%f,%f,%f,%d\n' % (koiId,koiT0,koiPeriod,koiDuration,self.toDummy(koiLabel)))
	
	def toDummy(self, label):
		if label == self.LABEL_TRUE:
			return 1
		if label == self.LABEL_FALSE:
			return 0
		print("Error: label ", label, "not recognized")	
		return -1			
	
	def deleteFile(self, fileName):
		if os.path.exists(fileName):
			os.remove(fileName)
		
# Execute only if script run standalone (not imported)						
if __name__ == '__main__':
	(script, sourceFileName, destinationFileName) = sys.argv
	extractor = KOIFeatureExtractor(sourceFileName)
	extractor.extractFeatures(destinationFileName)