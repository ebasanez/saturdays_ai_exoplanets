#pip install numpy
#pip install astropy
#pip install -e git+https://github.com/dfm/kplr#egg=kplr-dev
import kplr
import matplotli.pyplot as plt
client = kplr.API()

# Search info about Kepler Objects of Interest:
koi_id = "952.01"
koi = client.koi(koi_id)
print(koi.koi_period, koi.koi_period_err1, koi.koi_period_err2)

# Search information about stars:
#stars = client.stars(kic_teff="5700..5800") # Max 100 per request
star = client.star(9787239)
print("Light curves for star ", star.kic_teff)

time, flux, ferr, quality = [], [], [], []

light_curves = star.get_light_curves()
for light_curve in light_curves:
	print(light_curve.filename)
	with light_curve.open() as f:
		hdu_data = f[1].data
		time.append(hdu_data["time"])
		flux.append(hdu_data["sap_flux"])
		ferr.append(hdu_data["sap_flux_err"])
		quality.append(hdu_data["sap_quality"])

		

