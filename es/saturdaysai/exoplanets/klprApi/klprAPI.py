#pip install numpy
#pip install astropy
#pip install -e git+https://github.com/dfm/kplr#egg=kplr-dev
import kplr
import numpy as np
import pandas as pd

class KlprAPIClient():

	def __init__(self):
		self.client = kplr.API()

	def loadStarLightCurves(self, star_kic):
		star = self.client.star(star_kic)
		time, flux = [], []
		df =  pd.DataFrame(columns = ['time', 'flux']) 
		light_curves = star.get_light_curves()
		for light_curve in light_curves:
			with light_curve.open() as f:
				hdu_data = f[1].data
				time.append(hdu_data["time"])
				flux.append(hdu_data["sap_flux"])
				
		df =  pd.DataFrame(data=np.column_stack((np.concatenate(time),np.concatenate(flux))),columns=['time','flux'])
		df.set_index('time',inplace = True)
		return df
			
df = KlprAPIClient().loadStarLightCurves(9787239)

print("Data frame:",df.shape) 
print(df.head()) 
