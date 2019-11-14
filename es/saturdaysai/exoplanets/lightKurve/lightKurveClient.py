#conda install --channel conda-forge lightkurve
import pandas as pd
try:
	import lightkurve as lk
	from lightkurve import KeplerTargetPixelFile
	from lightkurve import search_targetpixelfile
	from lightkurve import search_lightcurvefile
except:
	import lightKurve as lk
	from lightKurve import KeplerTargetPixelFile
	from lightKurve import search_targetpixelfile
	from lightkurve import search_lightcurvefile

class LightKurveClient():
	
	def getKOIPixel(self,  koi_kic):
		tpf = search_targetpixelfile('KIC ' + str(koi_kic)).download_all(quality_bitmask='hardest')
		lightcurve = tpf.to_lightcurve()
		print(lightcurve.centroid_col)
		return tpf
	
	def getKOILightKurve(self,  koi_kic):
		lkf = search_lightcurvefile('KIC ' + str(koi_kic)).download_all(quality_bitmask='hardest')
		# TODO Obtain: dataframe with timestamp, flux, pixelMatrix and centroid
		time = lkf.PDCSAP_FLUX.time
		flux  = lkf.PDCSAP_FLUX.flux
		print(time)
		return lkf
	
	
	def getKOINames(self):
		# TODO
		pass
	
if __name__ == '__main__':
	import sys
	(script, koi_name) = sys.argv
	lkc = LightKurveClient()
	lkc.getKOIPixel(koi_name)
	#10797460

