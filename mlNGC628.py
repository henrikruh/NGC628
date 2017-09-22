# mlNGC628.py

# Henrik Ruh, 08/08/2017


'''
This script was used to experiment with different sorts of machine learning to classify radio sources in the spiral galaxy NGC 628.
Especialally, clustering of sources with using different parameters and algorithms was tested.
'''

# -----------------------------------------------------------------------
#%% imports

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Unsupervised Algorithms
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
from scipy.stats import gaussian_kde
from sklearn.preprocessing import normalize 

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import LogFormatterMathtext
import seaborn as sns

# Uncertainties

import uncertainties
from uncertainties import ufloat
from uncertainties import unumpy as unp
from uncertainties.unumpy import nominal_values as nom 
from uncertainties.unumpy import std_devs as std 

# Astrophysics

import astropy
from astropy.io import fits
from astropy.wcs import WCS

# Fitting
from scipy.optimize import curve_fit


# -----------------------------------------------------------------------
#%% definitions

def tabfmt ( s ):
    # formats string output of pd.to_latex 
    
    # replace strings by nice output
    s = s.replace('NaN','...')
    s = s.replace('nan+/-nan','...')
    s = s.replace('+/-nan','')
    s = s.replace('+/-',' $\pm$ ')
    s = s.replace('\$','$')

    return s
    
def distance (right_ascension_1, declination_1,right_ascension_2, declination_2):
    # compute distance of to positions in degree
	
    # convert degree to radians
    right_ascension_1 = np.pi*180**-1*right_ascension_1
    right_ascension_2 = np.pi*180**-1*right_ascension_2
    declination_1 = np.pi*180**-1*declination_1
    declination_2 = np.pi*180**-1*declination_2
    
    # compute distance
    distance = (((right_ascension_1-right_ascension_2)*unp.cos(declination_1))**2+(declination_1-declination_2)**2)**0.5
	
	# convert distance back to degree
    distance = 180*np.pi**-1*distance

    return distance

def values (series,index,cols):
	# converts a pandas series to a numpy array
	vals=[]
	for i in range(np.size(cols)):
	 	vals.append(series.ix[index,cols[i]])
	vals = np.array(vals)
	return vals

def plot_galaxy (imagename):
    # plots fits image of the galaxy

    # read in fits image
    hdu_list = fits.open(imagename)
    
    # extract image data
    imagedata = hdu_list[0].data
    
    # close imagefile
    hdu_list.close()
    
    # get information on wcs projection
    wcs = WCS(hdu_list[0].header)
    
    # plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=wcs)
    plt.imshow(imagedata, origin='lower', norm = mpl.colors.SymLogNorm(linthresh=1, linscale=0.1,vmax=np.max(imagedata),vmin=np.min(imagedata)),cmap='gray')
    plt.xlabel('RA')
    plt.ylabel('Dec')
    
    # plot colorbar
    formatter = LogFormatterMathtext(10, labelOnlyBase=False,linthresh=1) 
    
    # specify ticks to avoid overlaps
    cb = plt.colorbar(ticks=[1e4,1e3,1e2,1e1,0,-1e1,-1e2,-1e3,-1e4],format=formatter)
    cb.set_label(label='Jy/beam')
    
    return ax,cb,wcs

def plot_cluster ():
	# plots clustered data on galaxy map with a density plot
 
     # data frame that contains the source positions of sources selected for clustering
	plotting_frame = pd.DataFrame([nom(positions)[ind,0],nom(positions)[ind,1]]).transpose()
 
     # specify colormaps
	cmaps = ['Blues','Oranges','Greens','Reds','Purples']

	for i in range(max(cluster)+1):
         # for all clusters...    
     
          # cluster indices
		cind = np.where(cluster==i)[0]

		if len(cind) >=3:
              # check if cluster has more than two entries
              # otherwise can not create the density plot
             
               # plot the galaxy
			ax,cb,wcs = plot_galaxy(image_Ha_name)

			# kernel density
			sns.kdeplot(plotting_frame.ix[cind,0],plotting_frame.ix[cind,1],transform=ax.get_transform('world'),kind='kde',ax=ax,alpha=0.8,shade=False,cmap=cmaps[i])

			# kde centre
			for path in ax.collections[-1].get_paths():
                   # for all clusters
				x, y = path.vertices.mean(axis=0)
                    # plot the cluster centre x,y
				ax.plot(x, y,color = mpl.cm.get_cmap(cmaps[i])(0.99),marker = '.',transform=ax.get_transform('world'),markersize=8,mew=3,alpha=0.8)

			# source positions
			ax.scatter(nom(positions)[ind,0],nom(positions)[ind,1],transform=ax.get_transform('world'),c=cluster,cmap=plt.cm.Paired,s=80,edgecolors='k',alpha=0.8)

			# galactic centre
			ax.scatter(centre_pos[0],centre_pos[1],transform=ax.get_transform('world'),color = 'w',edgecolors='k',marker = 'D',s=40)

			plt.xlabel('Right Ascension')
			plt.ylabel('Declination')

			plt.xlim(0,2000)
			plt.ylim(0,2000)
   
           # save the plot
           #plt.savefig("cluster_plot_" + str(i) +".pdf",bbox_inches='tight',format='pdf',dpi=1200)

           
def plot_cluster_hist (data,cluster,ax):
    # plot a satcked histogram of the specified pd.DataFrame with indication of cluster membership

     # create a new dataframe 
	plotting_frame = pd.DataFrame()
     # specify the color
	colors = ['lightgray','k']

	for i in range(max(cluster)+1):
         # for all clusters..
         
         # find the indices of cluster members
		cind = np.where(cluster==i)[0]

         # append data from this cluster as a new column to the plotting data frame
		plotting_frame = pd.concat([plotting_frame,data.ix[cind].reset_index(drop=True)],axis=1)

	# plot the histogram 
	plotting_frame.plot.hist(ax=ax,stacked=True,bins=30,legend=False, edgecolor="k",color=colors[:max(cluster)+1],alpha=0.5)
	plt.ylabel('')
 

# machine learning tools (from kaggle) -----------------------------------------------------------


def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 12 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
    ax.set_xticklabels(df.corr().columns,rotation=90)
    ax.set_yticklabels(df.corr().columns,rotation=0)


def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    #plot_model_var_imp( tree , X , y )
    

# ----------------------------------------------------------------------------
def mplspecs (mult=1,tex=True):
    # specifications for plotting with matplotlib
    
    mpl.rc('font', family='serif', serif='CM')
    mpl.rc('font', size=16)
    mpl.rc('text', usetex=tex)
    mpl.rc('xtick', labelsize=14)
    mpl.rc('ytick', labelsize=14)
    mpl.rc('legend',fontsize=16)
    mpl.rc('axes', labelsize=16)
    mpl.rc('figure',figsize = [6.4*mult, 4.8*mult])

# ----------------------------------------------------------------------------
#%% matplotlib specifications

mplspecs(1, False)
plt.close('all')

# -----------------------------------------------------------------------
#%% inputs

# fits image hydrogen
image_Ha_name = 'NGC_0628-I-Ha-hwb2001.fits'

# centre position
centre_pos = [24.1739458, 15.7836619]

# source catalog
catalog_name = 'source_catalog2.csv'

# load catalog
catalog = pd.DataFrame.from_csv(catalog_name,header=0)

# -----------------------------------------------------------------------
'''
Data Preparation
'''
#%% data reduction

# take extended objects out of the source catalog
index=[6,7,8,9,11,12,13,14,15,16,17,18,40,41,42,48,49,50,51,52]


catalog = catalog.drop(index,axis=0)
catalog = catalog.reset_index(drop=True)
# ------------------------------------------------------------------------
#%% positions of radio sources

# define position of the source by the position of one gaussian

# position array
positions = []

# use position of one gaussian in band that is avaible
for i in catalog.index:
    # for all sources...
    
	if catalog.ix[i,'S-band'] == 1:
         # append position of the S-band observation
		positions.append([ufloat(values(catalog,i,['RA_S']),values(catalog,i,['E_RA_S'])),ufloat(values(catalog,i,['DEC_S']),values(catalog,i,['E_DEC_S']))])
	elif catalog.ix[i,'C-band'] == 1:
         # else c-band position
		positions.append([ufloat(values(catalog,i,['RA_C']),values(catalog,i,['E_RA_C'])),ufloat(values(catalog,i,['DEC_C']),values(catalog,i,['E_DEC_C']))])	
	else:
         # elsex-band position
		positions.append([ufloat(values(catalog,i,['RA_X']),values(catalog,i,['E_RA_X'])),ufloat(values(catalog,i,['DEC_X']),values(catalog,i,['E_DEC_X']))])	

# convert list to np array
positions=np.array(positions)

# -----------------------------------------------------------------------
#%% distance of sources to the galactic centre

# compute the distance
dist_to_centre = distance(centre_pos[0],centre_pos[1],positions[:,0],positions[:,1])

# new column in catalog
catalog['Distance_to_centre'] = pd.Series(nom(dist_to_centre), index=catalog.index)
catalog['E_Distance_to_centre'] = pd.Series(std(dist_to_centre), index=catalog.index)

# -----------------------------------------------------------------------
#%% steepness of the spectrum

# columns with bolean 'steepness' of sources
a = pd.Series( np.where( catalog['Spectral_index'] <= -0.5 , 1 , 0 ) , name = 'steep' )
catalog['steep'] = a
a = pd.Series( np.where( np.logical_and(catalog['Spectral_index'] >= -0.5, catalog['Spectral_index'] <= 0 ), 1 , 0 ) , name = 'flat' )
catalog['flat'] = a
a = pd.Series( np.where( catalog['Spectral_index'] > 0 , 1 , 0 ) , name = 'inverse' )
catalog['inverse'] = a

# -----------------------------------------------------------------------
#%% error columns
# find columns with errors

E_cols = []
val_cols = []

for col in catalog.columns:
	if col[:2] == 'E_':
         # write names of columns with errors to array
		E_cols.append(col)
	else:
         # write names of columns with values to array
		val_cols.append(col)


# -----------------------------------------------------------------------
#%% correlation map

# choose columns to plot in the correlation map
corr_cols = ['Total_flux_S', 'Total_flux_C', 'Total_flux_X', 'Peak_flux_S', 'Peak_flux_C', 'Peak_flux_X', 'Spectral_index', 'Compactness_S', 'Compactness_C', 'Compactness_X', 'Distance_to_centre','steep']


# plot correlation map 
plot_correlation_map(catalog[corr_cols])

#%% -----------------------------------------------------------------------
'''
Supervised Machine Learning
'''
# perform supervised machine learning on sources detected in two or three bands

# choose data columns for the supervised learning
sml_cols = ['RA_S', 'E_RA_S', 'DEC_S', 'E_DEC_S','Peak_flux_S', 'E_Peak_flux_S', 'Total_flux_S', 'E_Total_flux_S','DC_Maj_S', 'E_DC_Maj_S', 'DC_Min_S', 'E_DC_Min_S', 'DC_PA_S','E_DC_PA_S','Distance_to_centre']

# create catalog with detection in three, two and one band
cat_a = catalog.ix[np.where( catalog['Number of bands'] >= 2 )[0] , :]
cat_b = catalog.ix[np.where( catalog['Number of bands'] == 2 )[0] , :]
cat_c = catalog.ix[np.where( np.logical_and(catalog['S-band'] == 1, catalog['Number of bands'] == 1 ))[0] , :]

# Create all datasets to train, validate and test models
train_valid_X = cat_a[sml_cols]
train_valid_y = cat_a['steep']
test_X = cat_b[sml_cols]

train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )

print (catalog.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)

# plot model importance
plot_variable_importance(train_X, train_y)


# choose model
model = RandomForestClassifier(n_estimators=100)
#model = LogisticRegression()
#model = KNeighborsClassifier(n_neighbors = 6)
#model = SVC()
#model = GaussianNB()

# fit the model on the dataset
model.fit( train_X , train_y )

# Score the model
print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))

# -----------------------------------------------------------------------
#%% evaluation

rfecv = RFECV( estimator = model , step = 1 , cv = StratifiedKFold( train_y , 2 ) , scoring = 'accuracy' )
rfecv.fit( train_X , train_y )

print( "Optimal number of features : %d" % rfecv.n_features_ )
print (rfecv.score( train_X , train_y ) , rfecv.score( valid_X , valid_y ))

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel( "Number of features selected" )
plt.ylabel( "Cross validation score (nb of correct classifications)" )
plt.plot( range( 1 , len( rfecv.grid_scores_ ) + 1 ) , rfecv.grid_scores_ )

# -----------------------------------------------------------------------
#%% predictions


# make predictions on the spectral index of sources detected in only in one band
mod = model.predict(cat_c[sml_cols])

do_agree = 0
num = 0

for i in range(len(mod)):

	if catalog.ix[np.where( np.logical_and(catalog['S-band'] == 1, catalog['Number of bands'] == 1 ))[0][i] ,'Spectral_index'] < 100:
		num = num +1

		if mod[i] >= cat_c.reset_index(drop=True).ix[i,'steep']:
              # check if predictions are in conlict with limits on the spectral index or not
			do_agree=do_agree+1


print('Agreement with limits:')
print(do_agree*(num)**-1)

# -----------------------------------------------------------------------
'''
Unsupervised Machine Learning
'''

#%% choose clustering algorithm

# (1) kmeans --------------------------------------------------------------

clustering = KMeans(n_clusters=2)

# (2) Agglomerative -----------------------------------------------------

#clustering = AgglomerativeClustering(n_clusters=2)

# (3) DBSCAN ------------------------------------------------------------

#clustering = DBSCAN(eps=0.5, min_samples=2)

# (3) DBSCAN ------------------------------------------------------------

#clustering = SpectralClustering(n_clusters=2)


# -----------------------------------------------------------------------
#%% S-properties clustering

# parameter
usm_cols = [ u'Peak_flux_S', u'E_Peak_flux_S', u'Total_flux_S', u'E_Total_flux_S',
       u'DC_Maj_S', u'E_DC_Maj_S', u'DC_Min_S', u'E_DC_Min_S', u'Compactness_S', u'E_Compactness_S','Distance_to_centre']

#cat_spc_ind = catalog.ix[np.where( catalog['Spectral_index'] < 100 )[0] , :]
cat_sb = catalog.ix[np.where( catalog['S-band'] == 1 )[0] , :]
ind = np.where( catalog['S-band'] == 1 )[0]

clustering.fit(normalize(cat_sb[usm_cols]))

cluster = clustering.labels_

plot_cluster()

# -----------------------------------------------------------------------
#%% S-properties clustering reduced parameters

# parameter
usm_cols = [ u'Peak_flux_S',  u'Total_flux_S',
       u'DC_Maj_S', u'DC_Min_S',  u'Compactness_S', 'Distance_to_centre']

#cat_spc_ind = catalog.ix[np.where( catalog['Spectral_index'] < 100 )[0] , :]
cat_sb = catalog.ix[np.where( catalog['S-band'] == 1 )[0] , :]
ind = np.where( catalog['S-band'] == 1 )[0]

clustering.fit(normalize(cat_sb[usm_cols]))

cluster = clustering.labels_

plot_cluster()

# -----------------------------------------------------------------------
#%% position clustering

cat = pd.DataFrame([nom(positions)[ind,0],nom(positions)[ind,1]]).transpose()
ind = cat.index

clustering.fit(cat)

cluster = clustering.labels_

plot_cluster()

# -----------------------------------------------------------------------
#%% Spectral index and compactness, 2 and 3 bands

usm_cols = ['Spectral_index','Compactness_S','E_Compactness_S','Compactness_X','E_Compactness_X']

cat = catalog.ix[np.where( catalog['Number of bands'] >= 2 )[0] , :]
ind = cat.index

clustering.fit(normalize(cat[usm_cols]))

cluster = clustering.labels_

plot_cluster()

# -----------------------------------------------------------------------
#%% Spectral index

usm_cols = ['Spectral_index']

cat = catalog.ix[np.where( catalog['Spectral_index'] < 100 )[0] , :]
ind = cat.index

clustering.fit(cat[usm_cols])

cluster = clustering.labels_

plot_cluster()

# -----------------------------------------------------------------------
#%% distance

usm_cols = ['Distance_to_centre']

cat = catalog

ind = cat.index

clustering.fit(cat[usm_cols])

cluster = clustering.labels_


plot_cluster()

# -----------------------------------------------------------------------
#%% distance and spectral index

usm_cols = ['Distance_to_centre','Spectral_index']

cat = catalog.ix[np.where( catalog['Spectral_index'] < 100 )[0] , :]

ind = cat.index

clustering.fit(cat[usm_cols])

cluster = clustering.labels_


plot_cluster()

#%% histograms

# plot histograms of the source clusters

# select histogram columns 

hist_cols = ['Spectral_index','Compactness_S','Peak_flux_S', 'Distance_to_centre']
xlabels = ['(a)','(b)','(Jy/beam)\n(c)','(degree)\n(d)']

# plot histograms
fig = plt.figure()
comax = fig.add_subplot(111) 


for i in range(len(hist_cols)):
    # plot histograms on subfigures
    ax = fig.add_subplot(2,2,i+1)
    c = hist_cols[i]
    xl = xlabels[i]
    hist_data = catalog.ix[ind,c]
    plot_cluster_hist(hist_data,cluster,ax)
    plt.xlabel(xl)

    
# Turn off axis lines and ticks of the common plot
comax.spines['top'].set_color('none')
comax.spines['bottom'].set_color('none')
comax.spines['left'].set_color('none')
comax.spines['right'].set_color('none')
comax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

comax.set_ylabel('Number of Sources')

plt.tight_layout()

#plt.savefig("clushists.pdf",bbox_inches='tight',format='pdf',dpi=1200)


#%% statistics

# create masks for sources in cluster A,B
maskA=np.where(cluster==1,False,True)
maskB=np.where(cluster==1,True,False)

# print out cluster statistics
print('\nStatistics cluster A')
statA=cat_sb[maskA][hist_cols].describe()
print(statA)

print('\nStatistics cluster B')
statB=cat_sb[maskB][hist_cols].describe()
print(statB)


# latex

# create latex readable tables as strings 
ltxA = statA.ix[['mean', 'std'],:].T.to_latex()
ltxA = tabfmt(ltxA)

ltxB = statB.ix[['mean', 'std'],:].T.to_latex()
ltxB = tabfmt(ltxB)
