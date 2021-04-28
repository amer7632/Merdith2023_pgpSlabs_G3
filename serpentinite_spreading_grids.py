import numpy as np
from scipy.stats import truncnorm

def crust_characterisation(time, spreading_rate_array):

    '''
    Characterises the crust at each time step using a plate model.
    '''
    samples=1
    peridotite_thickness = SR_and_peridotite(samples,spreading_rate_array)
    print(peridotite_thickness)

    return
def SR_and_peridotite(samples, spreading_rate_array, nans_array, normal=True):

    '''
    Return peridotite proportion given a spreading rate. Optional flag is for
    normal or uniform distribution. Default is normal.
    '''

    # with vectors
    #print len(spreading_rate)
    empty_peridotite = np.zeros_like(spreading_rate_array)


    SR1 = np.where(spreading_rate_array<=10)
    SR2 = np.where((spreading_rate_array>10) & (spreading_rate_array<=20))
    SR3 = np.where((spreading_rate_array>20) & (spreading_rate_array<=35))
    SR4 = np.where(spreading_rate_array>35)

    if normal:
        #print 'normal'
            empty_peridotite[SR1] = np.mean(truncnorm.rvs(-1.65,1.65, scale=6,size=samples) + 90)
            empty_peridotite[SR2] = np.mean(truncnorm.rvs(-1.71,1.71, scale=19.6,size=samples) + 46)
            empty_peridotite[SR3] = np.mean(truncnorm.rvs(-1.57,1.57, scale=2.87,size=samples) + 9.5)
            empty_peridotite[SR4] = np.mean(truncnorm.rvs(-1.57,1.57, scale=2.87,size=samples) + 5)

    else:
        #print 'uniform'
            empty_peridotite[SR1] = np.mean(np.random.uniform(80, 100, size=samples))
            empty_peridotite[SR2] = np.mean(np.random.uniform(12.5, 80, size=samples))
            empty_peridotite[SR3] = np.mean(np.random.uniform(5, 15, size=samples))
            empty_peridotite[SR4] = np.mean(np.random.uniform(0, 10, size=samples))

    empty_peridotite[nans_array] = np.nan
    peridotite = empty_peridotite


    return peridotite

def SR_and_thickness(samples, spreading_rate_array, nans_array):
    # with vectors
    '''
    this function takes a spreading rate and returns a thickness for slow or
    ultraslow spreading (less than 40 mm/a) and returns a 'thickness' which in
    this case is the maximum depth of water penetration giving the maximum
    depth of serpentinisation. i.e. this returns the depth to the
    unserpentinised mantle lithosphere
    '''
    #truncated normal distributions for thickness

    empty_crustal_thickness = np.zeros_like(spreading_rate_array)
    SR1 = np.where(spreading_rate_array<=10)
    SR2 = np.where((spreading_rate_array>10) & (spreading_rate_array<=20))
    SR3 = np.where(spreading_rate_array>20)

    #slow and ultraslow
    empty_crustal_thickness[SR1] = np.mean(truncnorm.rvs(-0.6, 0.6, scale=0.4, size=samples) + 3.6)
    empty_crustal_thickness[SR2] = np.mean(truncnorm.rvs(-0.8, 0.8, scale=0.8, size=samples) + 3.6)
    empty_crustal_thickness[SR3] = np.mean(truncnorm.rvs(-0.6, 0.6, scale=0.4, size=samples) + 7.0)

    empty_crustal_thickness[nans_array] = np.nan

    crustal_thickness = empty_crustal_thickness

    return crustal_thickness

def volcanic_component(samples, spreading_rate_array, thickness_array, volcanic_percent_array, nans_array):

   #spreading_rate_array = ocean_crust_values['spreading_rate']
   #thickness_array = ocean_crust_values['crustal_thickness_kms']
    #olcanic_percent_array = ocean_crust_values['volcanic_percent']
    #ans_array = ocean_crust_values['NaN_array']

    #divide spreading rate
    SR1 = np.where(spreading_rate_array<=10)
    SR2 = np.where((spreading_rate_array>10) & (spreading_rate_array<=20))
    SR3 = np.where(spreading_rate_array>20)

    #create five empty arrays
    z = 5
    x = np.shape(nans_array.values)[0]
    y = np.shape(nans_array.values)[1]
    empty_thickness_arrays = np.zeros((z,x,y))

    #set nans
    for empty_array in empty_thickness_arrays:
        empty_array[nans_array] = np.nan

    #layers thicknesses
    upper_volc = 0.3
    lower_volc = 0.3
    transition = 0.2
    sheeted_dykes = 1.2
    gabbros = 5

    #fast/intermediate
    empty_thickness_arrays[0][SR3] = upper_volc
    empty_thickness_arrays[1][SR3] = lower_volc
    empty_thickness_arrays[2][SR3] = transition
    empty_thickness_arrays[3][SR3] = sheeted_dykes
    empty_thickness_arrays[4][SR3] = gabbros

    #slow, assume volcanic component is half basalt half gabbro
    empty_thickness_arrays[0][SR2] = thickness_array.values[SR2]*volcanic_percent_array.values[SR2]/100*0.5
    empty_thickness_arrays[1][SR2] = 0
    empty_thickness_arrays[2][SR2] = 0
    empty_thickness_arrays[3][SR2] = 0
    empty_thickness_arrays[4][SR2] = thickness_array.values[SR2]*volcanic_percent_array.values[SR2]/100*0.5
    #
    ##ultraslow, assume volcanic component is 0.2 basalt, 0.8 gabbro
    empty_thickness_arrays[0][SR1] = thickness_array.values[SR1]*volcanic_percent_array.values[SR1]/100*0.2
    empty_thickness_arrays[1][SR1] = 0
    empty_thickness_arrays[2][SR1] = 0
    empty_thickness_arrays[3][SR1] = 0
    empty_thickness_arrays[4][SR1] = thickness_array.values[SR1]*volcanic_percent_array.values[SR1]/100*0.8

    filled_thickness_arrays = empty_thickness_arrays

    return filled_thickness_arrays

def SlowUltraslow_DS(dsSurf, spreading_rate_array, thickness):

    empty_DS_SlowUltraslow = np.zeros_like(thickness)
    dInflex = 1.1

    SR1 = np.where(spreading_rate_array<=20)
    SlowUltraslow_dBot = thickness[SR1] #as below is unaltered mantle peridotites
    SlowUltraslow_ds = dsSurf[SR1]


    area_total = SlowUltraslow_dBot*100 #(max area of dsSurf)
    serp_area = (SlowUltraslow_ds * dInflex) + (((SlowUltraslow_dBot - dInflex) * SlowUltraslow_ds)/2.)
    serp_total = serp_area/area_total * 100

    empty_DS_SlowUltraslow[SR1] = serp_total

    DS_SlowUltraslow = empty_DS_SlowUltraslow

    return DS_SlowUltraslow

def degree_of_serpentinisation_of_system(spreading_rate_array, crustal_thickness_array, nans_array):


    empty_DS_array = np.zeros_like(spreading_rate_array)

    spreading_rate_array[spreading_rate_array > 50] = 50

    SR1 = np.where(spreading_rate_array<=10)
    SR2 = np.where((spreading_rate_array>10) & (spreading_rate_array<=20))
    SR3 = np.where((spreading_rate_array>20) & (spreading_rate_array<=35))
    SR4 = np.where(spreading_rate_array>35)

    y_ultraslow = np.arange(100,79,-1) # min depending on spreading rate
    y1_ultraslow = 0.0449*spreading_rate_array[SR1]**2 - 1.899*spreading_rate_array[SR1] + 98.47 #max

    #we need to map our spreading rates (ranging 1–10) to peridotite 80–100
    ind_ultraslow = np.round(spreading_rate_array[SR1]*20/10).astype(int)
    dsSurf_ultraslow = np.mean((np.floor(y1_ultraslow),y_ultraslow[ind_ultraslow]), axis=0)

    #slow
    y_slow = np.arange(80,19,-1)
    y1_slow = 0.2193*spreading_rate_array[SR2]**2 - 17.63*spreading_rate_array[SR2] + 369.03

    #we need to map our spreading rates (ranging 11–20) to peridotite 20–80
    ind_slow = np.round((spreading_rate_array[SR2]-10)*60/10).astype(int)
    dsSurf_slow = np.mean((np.floor(y1_slow),y_slow[ind_slow]), axis=0)

    #intermeidate
    y_intermediate = np.arange(20,-1,-1) # min DS depending on spreading rate
    y1_intermediate = 0.0231*spreading_rate_array[SR3]**2 - 3.247*spreading_rate_array[SR3] + 112.77
    y1_intermediate[y1_intermediate < 0] = 0

    #we need to map our spreading rates (ranging 21–35) to peridotite 0-20
    ind_intermediate = np.round((spreading_rate_array[SR3]-20)*20/15).astype(int)
    dsSurf_intermediate = np.mean((np.floor(y1_intermediate),y_intermediate[ind_intermediate]), axis=0)

    #fast
    y_fast = np.arange(10,-1,-1)
    y1_fast = 0.0115*spreading_rate_array[SR4]**2 - 2.3162*spreading_rate_array[SR4] + 115.48
    y1_fast[y1_fast < 0] = 0

    #we need to map our spreading rates (ranging 35–50) to peridotite 0-10
    ind_fast = np.round((spreading_rate_array[SR4]-35)*10/15).astype(int)
    dsSurf_fast = np.mean((np.floor(y1_fast),y_fast[ind_fast]), axis=0)


    empty_DS_array[SR1] = dsSurf_ultraslow
    empty_DS_array[SR2] = dsSurf_slow
    empty_DS_array[SR3] = dsSurf_intermediate
    empty_DS_array[SR4] = dsSurf_fast

    #finish slow and ultraslow
    slow_ultraslow_DS = SlowUltraslow_DS(empty_DS_array, spreading_rate_array, crustal_thickness_array)
    slow_ultraslow_DS[nans_array] = np.nan

    return slow_ultraslow_DS

def carbonate_content_serp(spreading_rate_array, nans_array):

    carbonate_max_slow_ultraslow = 0.34 #mean from Kelemen and Manning 2015 NB 0.34%
    max_DS = 100.

    empty_CO2_serp = np.zeros_like(spreading_rate_array)

    SR1 = np.where(spreading_rate_array<=20)
    SR2 = np.where(spreading_rate_array>20)

    empty_CO2_serp[SR1] = carbonate_max_slow_ultraslow/max_DS
    empty_CO2_serp[SR2] = 0.32/100

    full_CO2_serp = empty_CO2_serp
    full_CO2_serp[nans_array] = np.nan

    return full_CO2_serp

def carbon_serp_uncertainty(spreading_rate_array, nans_array, carbon_serpentinite_stds):


    ceil_SR = np.ceil(spreading_rate_array.values)
    #ceil_SR_plus_one = ceil_SR
    ceil_SR_for_index = np.nan_to_num(ceil_SR, copy=False).astype(np.int)
    C_uncertainty = carbon_serpentinite_stds[ceil_SR_for_index]
    #print(C_serp_stds)

    C_uncertainty[nans_array] = np.nan

    return C_uncertainty
