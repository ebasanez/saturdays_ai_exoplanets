import os
import sys
import re

class KOIFeatureExtractor:	
	
	COLUMN_ID = 0
	COLUMN_LABEL = 3
	COLUMN_PERIOD = 10

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
			destinationFile.write("koi_id, koi_period, koi_is_planet\n") 
			with open(self.sourceFileName,'r') as sourceFile:
				# Skip header row
				next(sourceFile) 
				# Read each line in dataset
				for line in sourceFile:
					lineItems = line.split(';')
					koiId = lineItems[self.COLUMN_ID]
					koiLabel = lineItems[self.COLUMN_LABEL]
					koiPeriod = lineItems[self.COLUMN_PERIOD]
					koiPeriod = self.cleanNumber(koiPeriod) # Remove dots present in dataset field
					# We are only interested in KOIs with CONFIRMED of FALSE POSITIVE TCEs
					if koiLabel in self.VALID_LABELS: 
						destinationFile.write('%s,%s,%d\n' % (koiId,koiPeriod,self.toDummy(koiLabel)))
	
	def toDummy(self, label):
		if label == self.LABEL_TRUE:
			return 1
		if label == self.LABEL_FALSE:
			return 0
		print("Error: label ", label, "not recognized")	
		return -1			
	
	def cleanNumber(self, number):
		return re.sub('[^0-9]','', number)
		
	def deleteFile(self, fileName):
		if os.path.exists(fileName):
			os.remove(fileName)
		
# Execute only if script run standalone (not imported)						
if __name__ == '__main__':
	(script, sourceFileName, destinationFileName) = sys.argv
	extractor = KOIFeatureExtractor(sourceFileName)
	extractor.extractFeatures(destinationFileName)