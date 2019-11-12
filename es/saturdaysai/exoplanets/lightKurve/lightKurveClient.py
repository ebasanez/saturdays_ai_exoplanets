#conda install --channel conda-forge lightkurve
try:
	import lightkurve as lk
	from lightkurve import KeplerTargetPixelFile
	from lightkurve import search_targetpixelfile
except:
	import lightKurve as lk
	from lightKurve import KeplerTargetPixelFile
	from lightKurve import search_targetpixelfile

class LightKurveClient():
	
	def __init__(self):
		#self.tpf = KeplerTargetPixelFile("https://archive.stsci.edu/pub/kepler/target_pixel_files/0069/006922244/kplr006922244-2010078095331_lpd-targ.fits.gz")
	
	def getKOIData(self,  koi_kic):
		# TODO Obtain: dataframe with timestamp, flux, pixelMatrix and centroid
		tpf = search_targetpixelfile('KIC ' + str(koi_kic)).download_all(quality_bitmask='hardest')
		return tpf
	
	def getKOINames(self):
		# TODO
		pass
	
	
#lkc = LightKurveClient()
#lkc.loadStarLightCurves(9787239)

