# catalog.py

# Henrik Ruh, 20/07/2017


'''

This scrip will read in source catalogs of NGC628 of radio sources in S-, C-, and X- bands.
It will search for counterpart and write out a joint catalog of sources.
It will compute the spectral index and compactness, and append them to the catalog.

This version uses the primary beam corrected images.

'''
#%% imports

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import LogFormatterMathtext
import seaborn as sns
from adjustText import adjust_text

# Uncertainties
import uncertainties
from uncertainties import ufloat
from uncertainties import unumpy

# Astrophysics
import astropy
from astropy.io import fits
from astropy.wcs import WCS

# Fitting
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------
#%% definitions

def distance (right_ascension_1, declination_1,right_ascension_2, declination_2):
    # computes the distance between two positions
    
    # convert dagree to radians
    right_ascension_1 = np.pi*180**-1*right_ascension_1
    right_ascension_2 = np.pi*180**-1*right_ascension_2
    declination_1 = np.pi*180**-1*declination_1
    declination_2 = np.pi*180**-1*declination_2
    
    # compute distance
    distance = (((right_ascension_1-right_ascension_2)*unumpy.cos(declination_1))**2+(declination_1-declination_2)**2)**0.5
    
    # convert distance back to degree
    distance = 180*np.pi**-1*distance
    
    return distance

def values (series,index,cols):
	# converts specified columns of a pandas series to a numpy array
 
	vals=[]
	for i in range(np.size(cols)):
	 	vals.append(series.ix[index,cols[i]])
   
	return vals
		

def flux_error(flux,E_fit,rms):
    # function to compute total flux error
    
	E_scale = 0.02 * flux # the error of the scale is defined as 0.02 of the flux

	# error propagation
	E_tot = (E_scale**2+E_fit**2+rms**2)**0.5

	return E_tot

def specfit2(x,*p):
    # fitting function for the spectral index
    
	a,b = p
	return b*x**a 

def general_fit(f, xdata, ydata, p0, sigma=None, x_range=[0]):
    """
    Pass all arguments to curve_fit, which uses non-linear least squares
    to fit a function, f, to data.  Calculate the uncertaities in the
    fit parameters from the covariance matrix.
    """

    popt, pcov = curve_fit(f, xdata, ydata, p0=p0, sigma=sigma,maxfev=1000)

    # The uncertainties are the square roots of the diagonal elements
    perr = np.sqrt(np.diag(pcov))

    if len(x_range) != 1:
        y_fit = f(x_range,*popt)	
    else:
        y_fit = None
    return popt, perr, y_fit

def mplspecs (mult=1,tex=True):
    # specifications for plotting with matplotlib

    mpl.rc('font', serif='CM')
    mpl.rc('font', size=16)
    mpl.rc('text', usetex=tex)
    mpl.rc('xtick', labelsize=14)
    mpl.rc('ytick', labelsize=14)
    mpl.rc('legend',fontsize=16)
    mpl.rc('axes', labelsize=16)
    mpl.rc('figure',figsize = [6.4*mult, 4.8*mult])

# ----------------------------------------------------------------------------
#%% matplotlib specifications

plt.close('all')
mplspecs(1, False)

# -----------------------------------------------------------------------
#%% inputs

# observed frequencies
nu = np.array([2.9995, 6.349, 9.999]) # in GHz (S,C,X)

# noise level
S_band_rms = 13.4e-6 # Jansky 
C_band_rms = 9.7e-6
X_band_rms = 2.0e-6

# flux correction factors
C_flux_corr = 1.058
E_C_flux_corr = 0.015
X_flux_corr = 2.666
E_X_flux_corr = 0.063

# beam size
beam_size= 1.6*3600**-1 # degree

# source detection threshold
sigma_thresh  = 5.5

 
# gauss catalogs
S_band_catalog_name = 'NGC628_S_band_25arcmin.pbc.image.pybdsm.gaul'
C_band_catalog_name = 'NGC628_C_band_25arcmin.pbc.image.pybdsm.gaul'
X_band_catalog_name = 'NGC628_X_band_25arcmin.pbc.image.pybdsm.gaul'

# rms images
s_rms_image_name = 'NGC628_S_band_25arcmin.pbc.image.pybdsm.rmsd_I.fits'
c_rms_image_name = 'NGC628_C_band_25arcmin.pbc.image.pybdsm.rmsd_I.fits'
x_rms_image_name = 'NGC628_X_band_25arcmin.pbc.image.pybdsm.rmsd_I.fits'

# images
simage_name = 'NGC628_S_band_25arcmin.pbc.image.fits' 
cimage_name = 'NGC628_C_band_25arcmin.pbc.image.fits' 
ximage_name = 'NGC628_X_band_25arcmin.pbc.image.fits' 

# define image for plotting
image_name = ximage_name

image_Ha_name = 'NGC_0628-I-Ha-hwb2001.fits'


#%% load catalogs

# S- band
s_catalog = pd.read_csv(S_band_catalog_name,header=4)
#print(s_catalog.head(2))  

# C-band
c_catalog = pd.read_csv(C_band_catalog_name,header=4)

# X-band
x_catalog = pd.read_csv(X_band_catalog_name,header=4)

# -----------------------------------------------------------------------
#%% flagging & flux corrections

# S-band ----------------------------------------------------------------

# drop sources close to image frame

# isl id
isl_id = [0,42,47,48,2,3]

# find indices
index =[]

for i in isl_id:
	index.append(np.where(s_catalog[' Isl_id']==i)[0][0])

# drop sources
s_catalog =  s_catalog.drop(index,axis=0)

s_catalog = s_catalog.reset_index(drop=True)

# C-band ----------------------------------------------------------------

# flux correction
c_catalog[' Total_flux']=c_catalog[' Total_flux']*C_flux_corr
c_catalog[' E_Total_flux']=np.sqrt((c_catalog[' Total_flux']*E_C_flux_corr)**2+(c_catalog[' E_Total_flux']*C_flux_corr)**2)

c_catalog[' Peak_flux']=c_catalog[' Peak_flux']*C_flux_corr
c_catalog[' E_Peak_flux']=np.sqrt((c_catalog[' Peak_flux']*E_C_flux_corr)**2+(c_catalog[' E_Peak_flux']*C_flux_corr)**2)


# drop souces close to image frame

# island id of sources to drop
isl_id = [1,9,16,4]

# find indices of those sources
index =[]

for i in isl_id:
	index.append(np.where(c_catalog[' Isl_id']==i)[0][0])

# drop sources
c_catalog =  c_catalog.drop(index,axis=0)

# reset the index
c_catalog = c_catalog.reset_index(drop=True)


# X-band ----------------------------------------------------------------

# flux correction
x_catalog[' Total_flux']=x_catalog[' Total_flux']*X_flux_corr
x_catalog[' E_Total_flux']=np.sqrt((x_catalog[' Total_flux']*E_X_flux_corr)**2+(x_catalog[' E_Total_flux']*X_flux_corr)**2)

x_catalog[' Peak_flux']=x_catalog[' Peak_flux']*X_flux_corr
x_catalog[' E_Peak_flux']=np.sqrt((x_catalog[' Peak_flux']*E_X_flux_corr)**2+(x_catalog[' E_Peak_flux']*X_flux_corr)**2)


# drop souces close to image frame

# isl id
isl_id = [0,2,3,38,40,39,37,36,24,17,13,6]

# find indices
index =[]

for i in isl_id:
	index.append(np.where(x_catalog[' Isl_id']==i)[0][0])

# drop sources
x_catalog =  x_catalog.drop(index,axis=0)

x_catalog = x_catalog.reset_index(drop=True)

# -----------------------------------------------------------------------
#%% RMS
# find the rms at the source positions from the rms maps

# s-band ----------------------------------------------------------------

# read in fits rms image
hdu_list = fits.open(s_rms_image_name)

# extract image data
image_data = hdu_list[0].data
image_data = image_data[0,0]

# get information on wcs projection
s_wcs = WCS(hdu_list[0].header)
s_wcs = WCS.dropaxis(s_wcs,2)
s_wcs = WCS.dropaxis(s_wcs,2)

# close imagefile
hdu_list.close()

image_data = np.array(image_data)


# source positions
coord = np.round([s_catalog[' Xposn'].values,s_catalog[' Yposn'].values])

s_rms = []

# find rms at source positions and append to list
for i in range(np.size(coord[0])):
	s_rms.append(image_data[int(coord[1][i])][int(coord[0][i])])

# make new column with rms
s_catalog[' rms'] = pd.Series(s_rms, index=s_catalog.index)

# rename data
s_rms = image_data


# x-band ----------------------------------------------------------------

# read in fits rms image
hdu_list = fits.open(x_rms_image_name)

# extract image data
image_data = hdu_list[0].data
image_data = image_data[0,0]

# get information on wcs projection
x_wcs = WCS(hdu_list[0].header)
x_wcs = WCS.dropaxis(x_wcs,2)
x_wcs = WCS.dropaxis(x_wcs,2)

# close imagefile
hdu_list.close()

image_data = np.array(image_data)

coord = np.round([x_catalog[' Xposn'].values,x_catalog[' Yposn'].values])

x_rms = []

for i in range(np.size(coord[0])):
	x_rms.append(image_data[int(coord[1][i])][int(coord[0][i])])

# make new column with rms
x_catalog[' rms'] = pd.Series(x_rms, index=x_catalog.index)

# rename data
x_rms = image_data


fig = plt.figure()
#ax = fig.add_subplot(111, projection=x_wcs)
plt.imshow(image_data, origin='lower',cmap='autumn')

# c-band ----------------------------------------------------------------

# read in fits rms image
hdu_list = fits.open(c_rms_image_name)

# extract image data
image_data = hdu_list[0].data
image_data = image_data[0,0]

# get information on wcs projection
c_wcs = WCS(hdu_list[0].header)
c_wcs = WCS.dropaxis(c_wcs,2)
c_wcs = WCS.dropaxis(c_wcs,2)

# close imagefile
hdu_list.close()

image_data = np.array(image_data)

coord = np.round([c_catalog[' Xposn'].values,c_catalog[' Yposn'].values])

c_rms = []

for i in range(np.size(coord[0])):
	c_rms.append(image_data[int(coord[1][i])][int(coord[0][i])])

# make new column with rms
c_catalog[' rms'] = pd.Series(c_rms, index=c_catalog.index)

# rename data
c_rms = image_data

# -----------------------------------------------------------------------
#%% Total errors

# Compute the total error on the flux including scale error, rms noise, and fitting error
# append as new column

s_catalog[' Err_Peak_flux'] = flux_error(s_catalog.ix[:,' Peak_flux'],s_catalog.ix[:,' E_Peak_flux'],s_catalog.ix[:,' rms'])
s_catalog[' Err_Total_flux'] = flux_error(s_catalog.ix[:,' Total_flux'],s_catalog.ix[:,' E_Total_flux'],s_catalog.ix[:,' rms'])
c_catalog[' Err_Peak_flux'] = flux_error(c_catalog.ix[:,' Peak_flux'],c_catalog.ix[:,' E_Peak_flux'],c_catalog.ix[:,' rms'])
c_catalog[' Err_Total_flux'] = flux_error(c_catalog.ix[:,' Total_flux'],c_catalog.ix[:,' E_Total_flux'],c_catalog.ix[:,' rms'])
x_catalog[' Err_Peak_flux'] = flux_error(x_catalog.ix[:,' Peak_flux'],x_catalog.ix[:,' E_Peak_flux'],x_catalog.ix[:,' rms'])
x_catalog[' Err_Total_flux'] = flux_error(x_catalog.ix[:,' Total_flux'],x_catalog.ix[:,' E_Total_flux'],x_catalog.ix[:,' rms'])

# -----------------------------------------------------------------------
#%% Cross-matching

# find counterparts of sources observed in different bands

# (i) S- and C-band counterparts ----------------------------------------

# create new dataframe for joint sources in s and c bands
# name columns matching column names of existing catalogs (will be renamed later)
s_c_matches = pd.DataFrame(columns=['# Gaus_id', ' Source_id', ' Isl_id', ' RA', ' E_RA', ' DEC',' E_DEC', ' Peak_flux', ' S_Code', '# Gaus_id', ' Source_id',' Isl_id', ' RA', ' E_RA', ' DEC', ' E_DEC', ' Peak_flux',' S_Code', ' E_Separation', ' Separation'])

# temp counter
temp_count = 0

# go through s-band catalog
for i in range(0,len(s_catalog.index),1):

	# compute distances between source positions
	phi = distance(unumpy.uarray([c_catalog.ix[:,' RA'],c_catalog.ix[:,' E_RA']]),unumpy.uarray([c_catalog.ix[:,' DEC'],c_catalog.ix[:,' E_DEC']]),unumpy.uarray([s_catalog.ix[i,' RA'],s_catalog.ix[i,' E_RA']]),unumpy.uarray([s_catalog.ix[i,' DEC'],s_catalog.ix[i,' E_DEC']]))

	# array with nominal values and with error
	phi_nom = unumpy.nominal_values(phi)
	phi_std = unumpy.std_devs(phi)

	# for one source in the s-band, check distances to each x band source
	for j in range(0,len(phi_nom)):

		# set limit for counterpart
		if phi_nom[j] <= beam_size: 

			temp_count=temp_count+1

			# new entry in catalog of matches

			# entries in parts for readability
			A = s_catalog.ix[i:i+1,['# Gaus_id',' Source_id',' Isl_id',' RA',' E_RA',' DEC',' E_DEC',' Peak_flux',' S_Code']]
			B = c_catalog.ix[j:j+1,['# Gaus_id',' Source_id',' Isl_id',' RA',' E_RA',' DEC',' E_DEC',' Peak_flux',' S_Code']]
			C = pd.DataFrame({' Separation':[phi_nom[j],np.nan],' E_Separation':[phi_std[j],np.nan]})

			# reset indices, so all parts have the same indices
			A = A.reset_index(drop=True)
			B = B.reset_index(drop=True)

			# join parts into new entry
			entry = pd.concat([A,B,C],axis=1)

			# append to dataframe of joint sources
			s_c_matches = pd.concat([s_c_matches,entry])

	
		else:
			tmp=0

# drop tmp rows
s_c_matches = s_c_matches.drop(s_c_matches.index[1])

# reset indices
s_c_matches = s_c_matches.reset_index(drop=True)

# rename columns
s_c_matches.columns =[' Gaus_id_S',' Source_id_S',' Isl_id_S',' RA_S',' E_RA_S',' DEC_S',' E_DEC_S',' Peak_flux_S',' S_Code_S',' Gaus_id_C',' Source_id_C',' Isl_id_C',' RA_C',' E_RA_C',' DEC_C',' E_DEC_C',' Peak_flux_C',' S_Code_C',' Separation',' E_Separation']


# (ii) S- and X-band counterparts ---------------------------------------------

# create new dataframe for joint sources in s and x bands
# name columns matching column names of existing catalogs (will be renamed later)

s_x_matches = pd.DataFrame(columns=['# Gaus_id', ' Source_id', ' Isl_id', ' RA', ' E_RA', ' DEC',' E_DEC', ' Peak_flux', ' S_Code', '# Gaus_id', ' Source_id',' Isl_id', ' RA', ' E_RA', ' DEC', ' E_DEC', ' Peak_flux',' S_Code', ' E_Separation', ' Separation'])

# temp counter
temp_count = 0

# go through s-band catalog
for i in range(0,len(s_catalog.index),1):

	# compute distances between source positions
	phi = distance(unumpy.uarray([x_catalog.ix[:,' RA'],x_catalog.ix[:,' E_RA']]),unumpy.uarray([x_catalog.ix[:,' DEC'],x_catalog.ix[:,' E_DEC']]),unumpy.uarray([s_catalog.ix[i,' RA'],s_catalog.ix[i,' E_RA']]),unumpy.uarray([s_catalog.ix[i,' DEC'],s_catalog.ix[i,' E_DEC']]))

	# array with nominal values and with error
	phi_nom = unumpy.nominal_values(phi)
	phi_std = unumpy.std_devs(phi)

	# for one source in the s-band, check distances to each x band source
	for j in range(0,len(phi_nom)):

		# set limit for counterpart
		if phi_nom[j] <= beam_size: 

			temp_count=temp_count+1

			# new entry in catalog of matches

			# entries in parts for readability
			A = s_catalog.ix[i:i+1,['# Gaus_id',' Source_id',' Isl_id',' RA',' E_RA',' DEC',' E_DEC',' Peak_flux',' S_Code']]
			B = x_catalog.ix[j:j+1,['# Gaus_id',' Source_id',' Isl_id',' RA',' E_RA',' DEC',' E_DEC',' Peak_flux',' S_Code']]
			C = pd.DataFrame({' Separation':[phi_nom[j],np.nan],' E_Separation':[phi_std[j],np.nan]})

			# reset indices, so all parts have the same indices
			A = A.reset_index(drop=True)
			B = B.reset_index(drop=True)

			# join parts into new entry
			entry = pd.concat([A,B,C],axis=1)

			# append to dataframe of joint sources
			s_x_matches = pd.concat([s_x_matches,entry])

	
		else:
			tmp=0
   
# drop tmp rows
s_x_matches = s_x_matches.drop(s_x_matches.index[1])

# reset indices
s_x_matches = s_x_matches.reset_index(drop=True)

# rename columns
s_x_matches.columns =[' Gaus_id_S',' Source_id_S',' Isl_id_S',' RA_S',' E_RA_S',' DEC_S',' E_DEC_S',' Peak_flux_S',' S_Code_S',' Gaus_id_X',' Source_id_X',' Isl_id_X',' RA_X',' E_RA_X',' DEC_X',' E_DEC_X',' Peak_flux_X',' S_Code_X',' Separation',' E_Separation']


# (iii) X- and C-band counterparts ---------------------------------------------

# create new dataframe for joint sources in x and c bands
# name columns matching column names of existing catalogs (will be renamed later)

x_c_matches = pd.DataFrame(columns=['# Gaus_id', ' Source_id', ' Isl_id', ' RA', ' E_RA', ' DEC',' E_DEC', ' Peak_flux', ' S_Code', '# Gaus_id', ' Source_id',' Isl_id', ' RA', ' E_RA', ' DEC', ' E_DEC', ' Peak_flux',' S_Code', ' E_Separation', ' Separation'])

# temp counter
temp_count = 0

# go through s-band catalog
for i in range(0,len(x_catalog.index),1):

	# compute distances between source positions
	phi = distance(unumpy.uarray([c_catalog.ix[:,' RA'],c_catalog.ix[:,' E_RA']]),unumpy.uarray([c_catalog.ix[:,' DEC'],c_catalog.ix[:,' E_DEC']]),unumpy.uarray([x_catalog.ix[i,' RA'],x_catalog.ix[i,' E_RA']]),unumpy.uarray([x_catalog.ix[i,' DEC'],x_catalog.ix[i,' E_DEC']]))

	# array with nominal values and with error
	phi_nom = unumpy.nominal_values(phi)
	phi_std = unumpy.std_devs(phi)

	# for one source in the s-band, check distances to each x band source
	for j in range(0,len(phi_nom)):

		# set limit for counterpart
		if phi_nom[j] <= beam_size: 

			temp_count=temp_count+1

			# new entry in catalog of matches

			# entries in parts for readability
			A = x_catalog.ix[i:i+1,['# Gaus_id',' Source_id',' Isl_id',' RA',' E_RA',' DEC',' E_DEC',' Peak_flux',' S_Code']]
			B = c_catalog.ix[j:j+1,['# Gaus_id',' Source_id',' Isl_id',' RA',' E_RA',' DEC',' E_DEC',' Peak_flux',' S_Code']]
			C = pd.DataFrame({' Separation':[phi_nom[j],np.nan],' E_Separation':[phi_std[j],np.nan]})

			# reset indices, so all parts have the same indices
			A = A.reset_index(drop=True)
			B = B.reset_index(drop=True)

			# join parts into new entry
			entry = pd.concat([A,B,C],axis=1)

			# append to dataframe of joint sources
			x_c_matches = pd.concat([x_c_matches,entry])

	
		else:
			tmp=0
# drop tmp rows
x_c_matches = x_c_matches.drop(x_c_matches.index[1])

# reset indices
x_c_matches = x_c_matches.reset_index(drop=True)

# rename columns
x_c_matches.columns =[' Gaus_id_X',' Source_id_X',' Isl_id_X',' RA_X',' E_RA_X',' DEC_X',' E_DEC_X',' Peak_flux_X',' S_Code_X',' Gaus_id_C',' Source_id_C',' Isl_id_C',' RA_C',' E_RA_C',' DEC_C',' E_DEC_C',' Peak_flux_C',' S_Code_C',' Separation',' E_Separation']


# -----------------------------------------------------------------------
#%% write sources into new catalog

# create catalog
source_catalog = pd.DataFrame(columns=['Gaus_id_S','Gaus_id_C','Gaus_id_X','Counterparts'])

# (i) S-band sources ------ ---------------------------------------------
# go through sources in S-band
for i in range(0,len(s_catalog.index),1):
	
	# variable to check for counterparts
	x_counterpart=0
	c_counterpart=0

	# compare with catalogs of counterparts and find indices
	s_x_index = np.where(s_x_matches.ix[:,' Gaus_id_S']==s_catalog.ix[i,'# Gaus_id'])[0]
	s_c_index = np.where(s_c_matches.ix[:,' Gaus_id_S']==s_catalog.ix[i,'# Gaus_id'])[0]

	# check if it has a X-band counterpart
	if np.size(s_x_index) != 0:
		x_counterpart=1

	# check if it has a C-band counterpart
	if np.size(s_c_index) != 0: 
		c_counterpart=1

	# write entry
	if x_counterpart==1 and c_counterpart==1:


		# print warning if there are more than one counterparts
		if np.size(s_x_index) > 1 or np.size(s_c_index > 1):
			print('Warning: s-band gauss %i has more than one counterpart.' %(s_catalog.ix[i,'# Gaus_id']))

		for j in range(np.size(s_c_index)):
			for k in range(np.size(s_x_index)):

				# make new entry
				entry = pd.DataFrame([[s_catalog.ix[i,'# Gaus_id'],s_c_matches.ix[s_c_index[j],' Gaus_id_C'],s_x_matches.ix[s_x_index[k],' Gaus_id_X'],2]],columns=['Gaus_id_S','Gaus_id_C','Gaus_id_X','Counterparts'])
				# append to catalog
				source_catalog=pd.concat([source_catalog,entry])

	elif x_counterpart==1:

		# print warning if there are more than one counterparts
		if np.size(s_x_index) > 1 or np.size(s_c_index > 1):
			print('Warning: s-band gauss %i has more than one counterpart.' %(s_catalog.ix[i,'# Gaus_id']))

		for k in range(np.size(s_x_index)):

			# make new entry
			entry = pd.DataFrame([[s_catalog.ix[i,'# Gaus_id'],np.nan,s_x_matches.ix[s_x_index[k],' Gaus_id_X'],1]],columns=['Gaus_id_S','Gaus_id_C','Gaus_id_X','Counterparts'])
			# append to catalog
			source_catalog=pd.concat([source_catalog,entry])

	elif c_counterpart==1:

		# print warning if there are more than one counterparts
		if np.size(s_x_index) > 1 or np.size(s_c_index > 1):
			print('Warning: s-band gauss %i has more than one counterpart.' %(s_catalog.ix[i,'# Gaus_id']))

		for j in range(np.size(s_c_index)):

			# make new entry
			entry = pd.DataFrame([[s_catalog.ix[i,'# Gaus_id'],np.nan,s_x_matches.ix[s_x_index[k],' Gaus_id_X'],1]],columns=['Gaus_id_S','Gaus_id_C','Gaus_id_X','Counterparts'])
			# append to catalog
			source_catalog=pd.concat([source_catalog,entry])

	else:
	# if not, write entry for single source
		# make new entry
		entry = pd.DataFrame([[s_catalog.ix[i,'# Gaus_id'],np.nan,np.nan,0]],columns=['Gaus_id_S','Gaus_id_C','Gaus_id_X','Counterparts'])
		# append to catalog
		source_catalog=pd.concat([source_catalog,entry])


# (ii) X-band Sources ------ ---------------------------------------------
# go through sources in X-band
for i in range(0,len(x_catalog.index),1):
	# check if it already is in the catalog
	if x_catalog.ix[i,'# Gaus_id'] in source_catalog.ix[:,'Gaus_id_X'].values:
		# do nothing
		tmp=0
	else: 		
	    # variable to check for counterparts
	    c_counterpart=0

	    # compare with catalog of counterparts and find indices
	    x_c_index = np.where(x_c_matches.ix[:,' Gaus_id_X']==x_catalog.ix[i,'# Gaus_id'])[0]

	    # check if it has a C-band counterpart
	    if np.size(x_c_index) != 0: 
		    c_counterpart=1

	    # write entry
	    if c_counterpart==1:

		    # print warning if there are more than one counterparts
		    if np.size(x_c_index > 1):
			    print('Warning: x-band gauss %i has more than one counterpart.' %(x_catalog.ix[i,'# Gaus_id']))

		    for j in range(np.size(x_c_index)):

			    # make new entry
			    entry = pd.DataFrame([[np.nan,x_c_matches.ix[x_c_index[j],' Gaus_id_C'],x_catalog.ix[i,'# Gaus_id'],1]],columns=['Gaus_id_S','Gaus_id_C','Gaus_id_X','Counterparts'])
			    # append to catalog
			    source_catalog=pd.concat([source_catalog,entry])

	    else:
	    # if not, write entry for single source
		    # make new entry
		    entry = pd.DataFrame([[np.nan,np.nan,x_catalog.ix[i,'# Gaus_id'],0]],columns=['Gaus_id_S','Gaus_id_C','Gaus_id_X','Counterparts'])
		    # append to catalog
		    source_catalog=pd.concat([source_catalog,entry])


# (iii) C-band Sources ------ ---------------------------------------------
# go through sources in C-band
for i in range(0,len(c_catalog.index)):
	# check if it already is in the catalog
	if c_catalog.ix[i,'# Gaus_id'] in source_catalog.ix[:,'Gaus_id_C'].values:
		# do nothing
		tmp=0
	else: 			    
	# if not, write entry for single source
		# make new entry
		entry = pd.DataFrame([[np.nan,c_catalog.ix[i,'# Gaus_id'],np.nan,0]],columns=['Gaus_id_S','Gaus_id_C','Gaus_id_X','Counterparts'])
		# append to catalog
		source_catalog=pd.concat([source_catalog,entry])

source_catalog = source_catalog.reset_index(drop=True)


# (iv) Catalog ----------------------------------------------------------
# write data into a joint catalog

# save reduced, simplified catalog
source_catalog_simp = source_catalog

scx_cols = ['Gaus_id_S', 'Source_id_S', 'Isl_id_S', 'RA_S', 'E_RA_S', 'DEC_S','E_DEC_S', 'Peak_flux_S', 'E_Peak_flux_S','Total_flux_S','E_Total_flux_S','DC_Maj_S','E_DC_Maj_S','DC_Min_S','E_DC_Min_S','DC_PA_S','E_DC_PA_S','Gaus_id_C', 'Source_id_C', 'Isl_id_C', 'RA_C', 'E_RA_C', 'DEC_C','E_DEC_C', 'Peak_flux_C', 'E_Peak_flux_C','Total_flux_C','E_Total_flux_C','DC_Maj_C','E_DC_Maj_C','DC_Min_C','E_DC_Min_C','DC_PA_C','E_DC_PA_C','Gaus_id_X', 'Source_id_X', 'Isl_id_X', 'RA_X', 'E_RA_X', 'DEC_X','E_DEC_X', 'Peak_flux_X', 'E_Peak_flux_X','Total_flux_X','E_Total_flux_X','DC_Maj_X','E_DC_Maj_X','DC_Min_X','E_DC_Min_X','DC_PA_X','E_DC_PA_X','Number of bands','S-band','C-band','X-band']

scx_data_cols = ['# Gaus_id', ' Source_id', ' Isl_id', ' RA', ' E_RA', ' DEC',' E_DEC', ' Peak_flux', ' Err_Peak_flux',' Total_flux',' Err_Total_flux',' DC_Maj',' E_DC_Maj',' DC_Min',' E_DC_Min',' DC_PA',' E_DC_PA']
# note: here we use the total error and rename it as E_Peak_flux, which was just the fitting error before

source_catalog = pd.DataFrame(columns=scx_cols)

for i in range(0,len(source_catalog_simp.index)):
# for each line, find the indice of gauss entry in the band catalog
	
	s_index = np.where(s_catalog.ix[:,'# Gaus_id']==source_catalog_simp.ix[i,'Gaus_id_S'])[0]
	c_index = np.where(c_catalog.ix[:,'# Gaus_id']==source_catalog_simp.ix[i,'Gaus_id_C'])[0]
	x_index = np.where(x_catalog.ix[:,'# Gaus_id']==source_catalog_simp.ix[i,'Gaus_id_X'])[0]
	
	# variable to indicate if there is band information at this entry
	s_band = 1
	c_band = 1
	x_band = 1

	if np.size(s_index) == 0:
		s_data = np.nan*np.ones(len(scx_data_cols))
		s_band = 0
	else:
		s_index = s_index[0]
		s_data = values(s_catalog,s_index,scx_data_cols)
	if np.size(c_index) == 0:
		c_data = np.nan*np.ones(len(scx_data_cols))	
		c_band = 0
	else:
		c_index = c_index[0]
		c_data = values(c_catalog,c_index,scx_data_cols)
	if np.size(x_index) == 0:
		x_data = np.nan*np.ones(len(scx_data_cols))
		x_band = 0	
	else:
		x_index = x_index[0]
		x_data = values(x_catalog,x_index,scx_data_cols)

	# make a entry with the relevant data
	entry = pd.DataFrame([np.concatenate([s_data,c_data,x_data,[source_catalog_simp.ix[i,'Counterparts']+1],[s_band,c_band,x_band]])],columns=scx_cols)
	
	# append to the catalog
	source_catalog=pd.concat([source_catalog,entry])

source_catalog = source_catalog.reset_index(drop=True)

# -----------------------------------------------------------------------
# Sorting

# drop non-relevant information
source_catalog =  source_catalog.drop(['Source_id_S', 'Isl_id_S', 'Source_id_C', 'Isl_id_C','Source_id_X', 'Isl_id_X'],axis=1)




# -----------------------------------------------------------------------
#%% Spectral index
spec_ind_cols = ['Spectral_index','E_Spectral_index','Upper_limit_spectral_index','Lower_limit_spectral_index']
spec_ind = pd.DataFrame(columns=spec_ind_cols)

# define the initial fitting parameters
func = specfit2
p0 = [1.,1.]

# set a frequency range to obtain the fitting values from the best model
nu_range = np.arange(3,10.1,0.1)

# open figure to plot results
plt.figure()

plt.xscale('log',subsx=[1.2,1.4,1.6,1.8])
plt.yscale('log')

plt.xlabel('log[Frequency/GHz]')
plt.ylabel('log[Normalised Flux Density/Jy]')
plt.title('Spectral index of sources in S,C and X')


for i in range(0,len(source_catalog.index)):

	# sources with S-, C-, X-band
	if source_catalog.ix[i,'Number of bands'] == 3:
	
		S_true =  [(source_catalog.ix[i,'Total_flux_S']+source_catalog.fillna(0).ix[i,'Total_flux_S']),(source_catalog.ix[i,'Total_flux_C']+source_catalog.fillna(0).ix[i,'Total_flux_C']),(source_catalog.ix[i,'Total_flux_X']+source_catalog.fillna(0).ix[i,'Total_flux_X'])]
		S_err =  [(source_catalog.ix[i,'E_Total_flux_S']**2+source_catalog.fillna(0).ix[i,'E_Total_flux_S']**2)**0.5,(source_catalog.ix[i,'E_Total_flux_C']**2+source_catalog.fillna(0).ix[i,'E_Total_flux_C']**2)**0.5,(source_catalog.ix[i,'E_Total_flux_X']**2+source_catalog.fillna(0).ix[i,'E_Total_flux_X']**2)**0.5]

		# Execute the fitting function
		popt, perr, S_fit = general_fit(func, nu, S_true, p0, sigma=S_err , x_range = nu_range)

		entry = pd.DataFrame([[popt[0],perr[0],np.nan,np.nan]],columns=spec_ind_cols)
		
		# plot
		plt.plot(nu_range,S_fit,'r',ls='-')

	# sources with 2 bands
	elif source_catalog.ix[i,'Number of bands'] == 2:
         
         # select the two bands where data is avaible to compute the spectral index
		if source_catalog.ix[i,'S-band'] == 1 and source_catalog.ix[i,'C-band'] == 1:
			S_true =  [source_catalog.ix[i,'Total_flux_S'],source_catalog.ix[i,'Total_flux_C']]
			S_err =  [source_catalog.ix[i,'E_Total_flux_S'],source_catalog.ix[i,'E_Total_flux_C']]
			V = nu[[0,1]]
		elif source_catalog.ix[i,'C-band'] == 1 and source_catalog.ix[i,'X-band'] == 1:
			S_true =  [source_catalog.ix[i,'Total_flux_C'],source_catalog.ix[i,'Total_flux_X']]
			S_err =  [source_catalog.ix[i,'E_Total_flux_C'],source_catalog.ix[i,'E_Total_flux_X']]
			V = nu[[1,2]]
		else:
			S_true =  [source_catalog.ix[i,'Total_flux_S'],source_catalog.ix[i,'Total_flux_X']]
			S_err =  [source_catalog.ix[i,'E_Total_flux_S'],source_catalog.ix[i,'E_Total_flux_X']]
			V = nu[[0,2]]
        
          # compute the spectral index
		alpha = np.log(V[0]*V[1]**-1)**-1*unumpy.log(ufloat(S_true[0],S_err[0])*ufloat(S_true[1],S_err[1])**-1)	

          # append to the list of new entries
		entry = pd.DataFrame([[alpha.n,alpha.s,np.nan,np.nan]],columns=spec_ind_cols)

		# plot the spectrum for sources in two bands
		plt.plot(V,S_true, '--b')

	# sources with in one band
	elif source_catalog.ix[i,'Number of bands'] == 1:
		
          # array for the coordinates of a source
		coord = []

		# use position of one gaussian in band that is avaible
		for j in source_catalog.index:
			if source_catalog.ix[j,'S-band'] == 1:
				coord.append([s_catalog.ix[np.where(s_catalog['# Gaus_id']==source_catalog.ix[j,'Gaus_id_S'])[0][0],' RA'],s_catalog.ix[np.where(s_catalog['# Gaus_id']==source_catalog.ix[j,'Gaus_id_S'])[0][0],' DEC']])
			elif source_catalog.ix[j,'C-band'] == 1:
				coord.append([c_catalog.ix[np.where(c_catalog['# Gaus_id']==source_catalog.ix[j,'Gaus_id_C'])[0][0],' RA'],c_catalog.ix[np.where(c_catalog['# Gaus_id']==source_catalog.ix[j,'Gaus_id_C'])[0][0],' DEC']])
			else:
				coord.append([x_catalog.ix[np.where(x_catalog['# Gaus_id']==source_catalog.ix[j,'Gaus_id_X'])[0][0],' RA'],x_catalog.ix[np.where(x_catalog['# Gaus_id']==source_catalog.ix[j,'Gaus_id_X'])[0][0],' DEC']])
          
          # arrays to hold the rms at the coordinates
		S_band_rms = []
		C_band_rms = []
		X_band_rms = []

		# convert world coordinates to pxl
		s_coord = s_wcs.wcs_world2pix(coord,1)
		c_coord = c_wcs.wcs_world2pix(coord,1)
		x_coord = x_wcs.wcs_world2pix(coord,1)


		# find rms at source positions and append to list
		for j in range(int(np.size(coord)*0.5)):
			S_band_rms.append(s_rms[int(s_coord[j][1])][int(s_coord[j][0])])
			C_band_rms.append(C_flux_corr*c_rms[int(c_coord[j][1])][int(c_coord[j][0])]) # multiply by flux correction factor
			X_band_rms.append(X_flux_corr*x_rms[int(x_coord[j][1])][int(x_coord[j][0])])
          
          # compute the flux density
		S_true =  [source_catalog.fillna(sigma_thresh*S_band_rms[i]).ix[i,'Total_flux_S'],source_catalog.fillna(sigma_thresh*C_band_rms[i]).ix[i,'Total_flux_C'],source_catalog.fillna(sigma_thresh*X_band_rms[i]).ix[i,'Total_flux_X']]
		S_err =  [source_catalog.fillna(0).ix[i,'E_Total_flux_S'],source_catalog.fillna(0).ix[i,'E_Total_flux_C'],source_catalog.fillna(0).ix[i,'E_Total_flux_X']]

		V=nu

		# compute spectral index between two bands (fill nans with threshold)
		alpha = [np.log(V[0]*V[1]**-1)**-1*unumpy.log(ufloat(S_true[0],S_err[0])*ufloat(S_true[1],S_err[1])**-1),np.log(V[1]*V[2]**-1)**-1*unumpy.log(ufloat(S_true[1],S_err[1])*ufloat(S_true[2],S_err[2])**-1),np.log(V[0]*V[2]**-1)**-1*unumpy.log(ufloat(S_true[0],S_err[0])*ufloat(S_true[2],S_err[2])**-1)]

		
		# convert to np.array for slicing
		alpha = np.array(alpha)

		if source_catalog.ix[i,'S-band'] == 1:

			# set limit indicator
			if alpha[0]<100:
				lim = 1
			if alpha[2]<100:
				lim = 1
			else:
				lim = 0

			# find upper limit, indicate in error column
			entry = pd.DataFrame([[np.min(alpha[[0,2]]).n,np.min(alpha[[0,2]]).s,lim,0]],columns=spec_ind_cols)			      

			# plot
			plt.plot(np.array(V)[[0,1]],np.array(S_true)[[0,1]], '-.y')
			plt.plot(np.array(V)[[0,2]],np.array(S_true)[[0,2]], '--y')

		elif source_catalog.ix[i,'X-band'] == 1:
	
			# set limit indicator
			if alpha[1]<100:
				lim = 1
			if alpha[2]<100:
				lim = 1
			else:
				lim = 0

			# find upper limit, indicate in error column
			entry = pd.DataFrame([[np.max(alpha[[1,2]]).n,np.max(alpha[[1,2]]).s,0,lim]],columns=spec_ind_cols)

			# plot
			#plt.errorbar(V,S_true,S_err,ecolor='k',fmt='.',color='k')
			plt.plot(np.array(V)[[1,2]],np.array(S_true)[[1,2]], '-.g')
			plt.plot(np.array(V)[[0,2]],np.array(S_true)[[0,2]], '--g')

		else:
			entry = pd.DataFrame([[np.max(alpha[[1,2]]).n,np.max(alpha[[1,2]]).s,np.nan,np.nan]],columns=spec_ind_cols)
			print(i)
			print('error in spectral index computation. C-band')
	else:
			print('error in spectral index computation.')
     
     # append entry to spectral indices
	spec_ind = pd.concat([spec_ind,entry])

# reset indices of spectral index data frame
spec_ind  = spec_ind.reset_index(drop=True)

# append the calumn with spectral indices to the source cataloge
source_catalog = pd.concat([source_catalog,spec_ind],axis=1)

# plot stacked histogram of spectral indices

# sort data by limitation of spectral index
upper = source_catalog.ix[np.where(source_catalog['Upper_limit_spectral_index']==1)[0],'Spectral_index'].reset_index(drop=True)
lower = source_catalog.ix[np.where(source_catalog['Lower_limit_spectral_index']==1)[0],'Spectral_index'].reset_index(drop=True)
other = source_catalog.ix[np.where(source_catalog['Number of bands']>=2)[0],'Spectral_index'].reset_index(drop=True)

spec_ind_plotting_frame = pd.DataFrame({'upper limit':upper, 'lower limit':lower, 'Detected in two or three bands':other})

# histogram plotting
spec_ind_plotting_frame.plot.hist(stacked=True,bins=50)

plt.xlabel('Spectral index')
plt.ylabel('Number of sources')

# -----------------------------------------------------------------------
#%% Compactness

# name columns for the compactness
comp_columns=['Compactness_S','E_Compactness_S','Compactness_C','E_Compactness_C','Compactness_X','E_Compactness_X']

# create data frame for the compactness
comp_frame = pd.DataFrame(columns=comp_columns)

for i in range(0,len(source_catalog.index)):
    # check if there is data for a band and compute the compactness
    
	if source_catalog.ix[i,'S-band'] == 1:
		s_comp = ufloat(source_catalog.ix[i,'Peak_flux_S'],source_catalog.ix[i,'E_Peak_flux_S'])**-1*ufloat(source_catalog.ix[i,'Total_flux_S'],source_catalog.ix[i,'E_Total_flux_S'])
	else:													    
		s_comp = ufloat(np.nan,np.nan)								       	
	if source_catalog.ix[i,'C-band'] == 1:
		c_comp = ufloat(source_catalog.ix[i,'Peak_flux_C'],source_catalog.ix[i,'E_Peak_flux_C'])**-1*ufloat(source_catalog.ix[i,'Total_flux_C'],source_catalog.ix[i,'E_Total_flux_C'])
	else:													    
		c_comp = ufloat(np.nan,np.nan)	    
	if source_catalog.ix[i,'X-band'] == 1:
		x_comp = ufloat(source_catalog.ix[i,'Peak_flux_X'],source_catalog.ix[i,'E_Peak_flux_X'])**-1*ufloat(source_catalog.ix[i,'Total_flux_X'],source_catalog.ix[i,'E_Total_flux_X'])
	else:													    
		x_comp = ufloat(np.nan,np.nan)
  
     # create a new entry from the data
	entry=pd.DataFrame([[s_comp.n,s_comp.s,c_comp.n,c_comp.s,x_comp.n,x_comp.s]],columns=comp_columns)
     
     # append entry to compactness columns
	comp_frame=pd.concat([comp_frame,entry])


# reset indices
comp_frame = comp_frame.reset_index(drop=True)

# append columns to source list
source_catalog = pd.concat([source_catalog,comp_frame],axis=1)

# plot histogram of compactnes data
comp_frame[['Compactness_S','Compactness_C','Compactness_X']].plot.hist(alpha=0.5,bins=50)
plt.xlabel('Compactness')
plt.ylabel('Number of sources')

# -----------------------------------------------------------------------
#%% save the cource catalog as csv

source_catalog.to_csv('source_catalog.csv')


# -----------------------------------------------------------------------
#%% create array with source positions and markers for plotting

# source positions in pxl

positions = []
markers=[]

# use position of one gaussian in band that is avaible
for i in source_catalog.index:

	if source_catalog.ix[i,'S-band'] == 1:
		positions.append(values(source_catalog,i,['RA_S','DEC_S']))
	elif source_catalog.ix[i,'C-band'] == 1:
		positions.append(values(source_catalog,i,['RA_C','DEC_C']))
	else:
		positions.append(values(source_catalog,i,['RA_X','DEC_X']))	
		
	# give different markers depending on upper or lower limit
	if source_catalog.ix[i,'Upper_limit_spectral_index'] == 1:
		markers.append('v')
	elif source_catalog.ix[i,'Lower_limit_spectral_index'] == 1:
		markers.append('^')
	elif source_catalog.ix[i,'Spectral_index'] < 100:
		markers.append('s')
	else:
		markers.append('o')

# convert list to np array
positions=np.array(positions)
markers=np.array(markers)

# mask data to show special sources
mask=[]

for i in source_catalog.index:
	if source_catalog.ix[i,'Number of bands']>=0:
		mask.append(True)
	else:
		mask.append(False)
		
mask = np.array(mask)
positions=positions[mask]
markers=markers[mask]


# -----------------------------------------------------------------------
#%% plot data

# get s-band fits image

# read in fits image
hdu_list = fits.open(image_name)

# extract image data
image_data = hdu_list[0].data
image_data = image_data[0,0]

# close imagefile
hdu_list.close()

# get information on wcs projection
wcs = WCS(hdu_list[0].header)
wcs = WCS.dropaxis(wcs,2)
wcs = WCS.dropaxis(wcs,2)


# get optical fits image

# read in fits image
hdu_list = fits.open(image_Ha_name)

# extract image data
image_Ha_data = hdu_list[0].data

# close imagefile
hdu_list.close()

# get information on wcs projection
wcs_Ha = WCS(hdu_list[0].header)


# -----------------------------------------------------------------------
#%% plot map without source markers for comparison

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection=wcs_Ha)
plt.imshow(image_Ha_data, origin='lower', norm = mpl.colors.SymLogNorm(linthresh=1, linscale=0.1,vmax=np.max(image_Ha_data),vmin=np.min(image_Ha_data)),cmap='autumn')
plt.xlabel('Right Ascension')
plt.ylabel('Declination')

# plot colorbar
formatter = LogFormatterMathtext(10, labelOnlyBase=False,linthresh=1) 
    
# specify ticks to avoid overlaps
cb = plt.colorbar(ticks=[1e4,1e3,1e2,1e1,0,-1e1,-1e2,-1e3,-1e4],format=formatter)
cb.set_label(label='Jy/beam')

#plt.savefig("haim0.5.pdf",bbox_inches='tight',format='pdf',dpi=600)

# -----------------------------------------------------------------------
#%% plot the source position and spectral index


# plot the map
fig2 = plt.figure()
ax = fig2.add_subplot(211, projection=wcs_Ha)

# cut out image frames
centre_pos = [24.1739458, 15.7836619]
centre_pos = wcs_Ha.wcs_world2pix([centre_pos], 1)[0].astype(int)

plt.imshow(image_Ha_data, origin='lower', norm = mpl.colors.SymLogNorm(linthresh=1, linscale=0.1,vmax=np.max(image_Ha_data),vmin=np.min(image_Ha_data)),cmap='autumn')
plt.xlim(900,1150)
plt.ylim(1100,1250)
plt.xlabel('Right Ascension')
plt.ylabel('Declination')

lon = ax.coords[0]
lon.set_ticks(number=4)

# plot colorbar
formatter = LogFormatterMathtext(10, labelOnlyBase=False,linthresh=1) 
    
# specify ticks to avoid overlaps
cb = plt.colorbar(ticks=[1e4,1e3,1e2,1e1,0,-1e1,-1e2,-1e3,-1e4],format=formatter)
cb.set_label(label='Jy/beam')


# plot source position with spectral indice
sct_id = ax.scatter(positions[:,0],positions[:,1],marker='o',c=source_catalog['Spectral_index'][mask],cmap='jet',transform=ax.get_transform('world'),s=75,vmin=-3.1,vmax=3.1,alpha=1)

for i in range(len(markers)):
	ax.scatter(positions[i,0],positions[i,1],marker=markers[i], color='k',facecolor='None',transform=ax.get_transform('world'),s=15,alpha=1)

 
xy=[]
texts=[]
s=[]

# annotate source numbers
for i, txt in enumerate(source_catalog.index[mask]):
    txt = int(txt)


    xy.append([positions[i,0],positions[i,1]])
    s.append(txt)

xy = wcs_Ha.wcs_world2pix(xy,1)
x = xy[:,0]
y = xy[:,1]

for i in range(len(x)):
    texts.append(plt.text(x[i], y[i], s[i], size=10, color='k'))

adjust_text(texts,x,y,autoalign='xy',force_points=0.7,force_text=0.5)
 
# plot colorbar
cb = plt.colorbar(sct_id)
cb.set_label(label='Spectral index')

lon = ax.coords[0]
lon.set_ticks(number=4)


#plt.savefig("galarm.pdf",bbox_inches='tight',format='pdf',dpi=1200)


# -----------------------------------------------------------------------
#%% plot the source position with spectral index on observational map X

# plot the map
fig3 = plt.figure()
ax = fig3.add_subplot(211, projection=wcs)

# cut out image frames
centre_pos = [24.1739458, 15.7836619]
centre_pos = wcs.wcs_world2pix([centre_pos], 1)[0].astype(int)

plt.imshow(image_data, origin='lower', norm = mpl.colors.SymLogNorm(linthresh=1e-6, linscale=0.1,vmax=1e-5,vmin=-1e-5),cmap='winter')
plt.xlim(centre_pos[0]-1500,centre_pos[0]+1500)
plt.ylim(centre_pos[1]-1500,centre_pos[1]+1500)
plt.xlabel('Right Ascension')
plt.ylabel('Declination')


# plot colorbar
formatter = LogFormatterMathtext(10, labelOnlyBase=False,linthresh=1e-6) 
    
# specify ticks to avoid overlaps
cb = plt.colorbar(format=formatter,ticks=[1e-4,1e-5,0,-1e-5,-1e-4])
cb.set_label(label='Jy/beam')

 
xy=[]
texts=[]
s=[]

# annotate source numbers
for i, txt in enumerate(source_catalog.index[mask]):
    txt = int(txt)

    xy.append([positions[i,0],positions[i,1]])
    s.append(txt)

xy = wcs.wcs_world2pix(xy,1)
x = xy[:,0]
y = xy[:,1]

for i in range(len(x)):
    texts.append(plt.text(x[i], y[i], s[i], size=10, color='k'))


# plot colorbar
cb = plt.colorbar(sct_id)
cb.set_label(label='Spectral index')


lon = ax.coords[0]
lon.set_ticks(number=4)

# -----------------------------------------------------------------------
#%% plot the source position with source numbers

# plot the map
fig4=plt.figure()
ax = fig4.add_subplot(111, projection=wcs_Ha)

plt.imshow(image_Ha_data, origin='lower', norm = mpl.colors.SymLogNorm(linthresh=1, linscale=0.1,vmax=np.max(image_Ha_data),vmin=np.min(image_Ha_data)),cmap='autumn')
plt.xlabel('Right Ascension')
plt.ylabel('Declination')



# plot colorbar
formatter = LogFormatterMathtext(10, labelOnlyBase=False,linthresh=1e-6) 
    
# specify ticks to avoid overlaps
cb = plt.colorbar(format=formatter)
cb.set_label(label='Jy/beam')


# plot source positions
ax.scatter(positions[:,0],positions[:,1],marker='.',c='k',transform=ax.get_transform('world'),s=25,vmin=-3.1,vmax=3.1,alpha=1)

xy=[]
texts=[]
s=[]
# annotate source numbers
for i, txt in enumerate(source_catalog.index[mask]):
    txt = int(txt)


    xy.append([positions[i,0],positions[i,1]])
    s.append(txt)

xy = wcs_Ha.wcs_world2pix(xy,1)
x = xy[:,0]
y = xy[:,1]

for i in range(len(x)):
    texts.append(plt.text(x[i], y[i], s[i], size=10, color='k'))

adjust_text(texts,x,y,autoalign='xy',force_points=0.8,force_text=0.8,arrowprops=dict(arrowstyle="-", color='k', lw=1,alpha=0.5))

#fig4.savefig("srcid.pdf",bbox_inches='tight',format='pdf',dpi=1200)

# -----------------------------------------------------------------------