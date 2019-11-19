#conda install --channel conda-forge lightkurve
import pandas as pd
try:
	import lightkurve as lk
	from lightkurve import LightCurveCollection
	from lightkurve import search_lightcurvefile
except:
	import lightKurve as lk
	from lightKurve import LightCurveCollection
	from lightkurve import search_lightcurvefile

class LightKurveClient():
	
	def getKOILightKurve(self, koi_kic, quarter = None, fold_period = None, flatten = True, t0 = 0, mission = 'Kepler', fold_mode = 'split'):
		if quarter == None:
			print(f"Obtaining all {mission} light curves for KOI {koi_kic}.")
			lcs = search_lightcurvefile('KIC ' + str(koi_kic)).download_all(quality_bitmask='hardest').PDCSAP_FLUX
			lcc = LightCurveCollection([lc for lc in lcs if lc.mission == mission])
			lkf = lcc.stitch()
		else:
			print(f"Obtaining {mission} light curve for KOI {koi_kic} (quarter {quarter})")
			lkf = search_lightcurvefile('KIC ' + str(koi_kic),quarter=quarter).download(quality_bitmask='hardest').PDCSAP_FLUX
		if flatten: 
			print("Flattening light curve")
			lkf = lkf.flatten()		
		
		# TODO Obtain: dataframe with timestamp, flux, centroid_col, centroid_row, transit_timestamp, transit_duration
		
		
		if fold_mode == 'fold':
			return self.toFoldedDataFrame(lkf, t0, fold_period)
			
		if fold_mode == 'split':
			return self.toSplittedDataFrame(lkf, t0, fold_period)
		
		return [pd.DataFrame({'time':lkf.time,'flux':lkf.flux})]
	
	def toFoldedDataFrame(self, lightCurve, t0, period):
		print(f"Folding light curve by period = {period} and start = {t0}")
		foldedLightCurve = lightCurve.fold(period, t0)
		df =  pd.DataFrame({'time' : foldedLightCurve.time, 'flux' : foldedLightCurve.flux})
		return [df]
	
	def toSplittedDataFrame(self, lightCurve, t0, period):
		print(f"Spliting light curve by period = {period} and start = {t0}")
		init_time = t0 - period / 2
		df = pd.DataFrame({'time' : lightCurve.time, 'flux' : lightCurve.flux})
		df['period'] = df['time'].map(lambda t: int((t - t0 - period / 2) // period) )
		# Remove first and last period (could be incomplete)
		max_period = df.period.max()
		df = df[df.period != -1]
		df = df[df.period != max_period]	
		
		dfs = []
		for i in range(max_period - 1):
			# Transform time to fold time (from -0.5 to +0.5)
			toPeriod = lambda t: toFoldTime(t, t0,period, i)
			dfs.append(df[['time','flux']][df.period == i].reset_index(drop = True))

		print(f"Generated {len(dfs)} folds")
		return dfs
			
		
	def toFoldTime(time, t0, period_duration, period_num):
		return (time - t0 - period_duration / 2 - (period_duration * period_num))/period_duration  - .5

		
if __name__ == '__main__':
	import sys
	(script, koi_name, period) = sys.argv
	lkc = LightKurveClient()
	lkc.getKOILightKurve(koi_name, fold_period = float(period), fold_mode = 'split')

