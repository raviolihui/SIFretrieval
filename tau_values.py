import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt


###SAHARA
h = nc.Dataset('Data/S5P_OFFL_L1C_SIFTRS_20240206T123417_20240206T123957_32732_93_010100_20250228T104502_irr.nc')
africa = h.groups['africa']
h1 = nc.Dataset('Data/S5P_OFFL_L1C_SIFTRS_20240206T105346_20240206T105827_32731_93_010100_20250228T104244_irr.nc')
africa1 = h1.groups['africa']

def filter_scanlines(dataset, cloud_fraction_threshold, reflectance_err_threshold, sza_threshold, vza_threshold, surface_classification=None):
    scanlines = []
    for n in range(dataset.variables['CloudFraction'].shape[1]):  # Iterate over scanlines
        if (dataset.variables['CloudFraction'][0, n, 223] < cloud_fraction_threshold).all() and \
            (dataset.variables["Reflectance_err"][0, n, 223] < reflectance_err_threshold).all() and \
            (dataset.variables["SZA"][0, n, 223] < sza_threshold).all() and \
            (dataset.variables["VZA"][0, n, 223] < vza_threshold).all() and \
            (surface_classification is None or (dataset.variables["SurfaceClassification"][0, n, 223] == surface_classification).all()):
            scanlines.append(n)
    return scanlines

scanline_nocloud = filter_scanlines(africa, cloud_fraction_threshold=0.4, reflectance_err_threshold=80, sza_threshold=75, vza_threshold=65, surface_classification=148)
scanline_nocloud1 = filter_scanlines(africa1, cloud_fraction_threshold=0.4, reflectance_err_threshold=80, sza_threshold=75, vza_threshold=65, surface_classification=148)

nocloud_value = np.concatenate((scanline_nocloud, scanline_nocloud1), axis=0)
print(nocloud_value.shape)

#Fixing albedo 
import numpy.polynomial.polynomial as po
def indexate(wl, ranges):
    if isinstance(ranges[0], (int, float)):  # Single range
        start, end = ranges
        return np.where((wl >= start) & (wl <= end))[0]
    else:  # Multiple ranges
        indices = np.concatenate([np.where((wl >= start) & (wl <= end))[0] for start, end in ranges])
        return np.sort(indices)
        
ds = nc.Dataset("Data/wl_per_grpx_sahara_b.nc")
wl_per_gp_sahara_b = ds.variables["Ref_wl"][:]
wl_ground_pixel_224 = wl_per_gp_sahara_b[223, :]  # select ground pixel 224 (index 223)
wl = wl_ground_pixel_224 
retrievalWindow = (734, 758) # retrieval wavelength window [nm]
windowsOfNoAbsorption = ((712, 713), (748, 757), (775, 785)) # windows for no atmospheric absorption [nm]
sb_order = 5 # order of polynomial fit of surface reflectivity (barren)
ind    = indexate(wl, retrievalWindow)
ind_na = indexate(wl, windowsOfNoAbsorption)
ref_na = africa.variables["Reflectance"][0, :, 223, ind_na].data.tolist()

#for each scanline model the albedo with the noabsobtion window of reflectance
surf_albedo = np.zeros((len(scanline_nocloud), len(ind)))
for idx, i in enumerate(scanline_nocloud):
    poly_sa = po.polyfit (wl[ind_na], ref_na[i], sb_order)
    surf_alb = po.polyval (wl[ind], poly_sa)
    surf_albedo[idx, :] = surf_alb

ref_na1 = africa1.variables["Reflectance"][0, :, 223, ind_na].data.tolist()

#for each scanline model the albedo with the noabsobtion window of reflectance
surf_albedo1 = np.zeros((len(scanline_nocloud1), len(ind)))
for idx, i in enumerate(scanline_nocloud1):
    poly_sa = po.polyfit (wl[ind_na], ref_na1[i], sb_order)
    surf_alb = po.polyval (wl[ind], poly_sa)
    surf_albedo1[idx, :] = surf_alb

albedo_value = np.concatenate((surf_albedo, surf_albedo1), axis = 0)
print(albedo_value.shape)

#Computing tau
mu = np.cos(np.radians(africa.variables["VZA"][0,scanline_nocloud,223]))
mu_0 = np.cos(np.radians(africa.variables["SZA"][0,scanline_nocloud,223]))
mu_matrix = np.tile(mu[:, np.newaxis], len(ind))
mu_0_matrix = np.tile(mu_0[:, np.newaxis], len(ind))
reflectance_matrix = africa.variables['Reflectance'][0, scanline_nocloud, 223, ind].data


mu1 = np.cos(np.radians(africa1.variables["VZA"][0,scanline_nocloud1,223]))
mu_01 = np.cos(np.radians(africa1.variables["SZA"][0,scanline_nocloud1,223]))
mu_matrix1 = np.tile(mu1[:, np.newaxis], len(ind))
mu_0_matrix1 = np.tile(mu_01[:, np.newaxis], len(ind))
reflectance_matrix1 = africa1.variables['Reflectance'][0, scanline_nocloud1, 223, ind].data

mu_value =np.concatenate((mu_matrix, mu_matrix1), axis = 0)
mu_0_value = np.concatenate((mu_0_matrix, mu_0_matrix1), axis = 0)
reflectance_value = np.concatenate((reflectance_matrix, reflectance_matrix1), axis = 0)
angle_value = (mu_value + mu_0_value)/mu_value*mu_0_value
tau_value = -np.log(reflectance_value/albedo_value)/angle_value

plt.figure()
for idx, i  in enumerate(nocloud_value):
    plt.plot(wl[ind], tau_value[idx],color='blue', alpha=0.1, linewidth=0.1)
    plt.xlabel("Wavelength - nm")
    plt.ylabel("Optical depth (τ)")
    plt.savefig("tau_value.png")
plt.close()

#SOLAR IRRADIANCE -- The same in all areas of the world 
irradiance_vector = africa.variables["irradiance"][223, ind]
#CONVERTING UNITS --- 
def convert_irradiance(irradiance_mol, wavelength_nm):
    """
    Convert irradiance from mol s⁻¹ m⁻² nm⁻¹ (photon flux) to mW m⁻² nm⁻¹.

    Parameters:
        irradiance_mol (array-like): Irradiance in mol s⁻¹ m⁻² nm⁻¹
        wavelength_nm (array-like): Corresponding wavelengths in nm

    Returns:
        array: Irradiance in mW m⁻² nm⁻¹
    """
    # Constants
    h = 6.62607015e-34  # Planck constant (J·s)
    c = 2.99792458e8    # Speed of light (m/s)
    Na = 6.02214076e23  # Avogadro's number (photons/mol)

    wavelength_m = np.array(wavelength_nm) * 1e-9  # Convert nm to m
    photon_energy = h * c / wavelength_m           # Energy per photon (J)

    irradiance_watts = np.array(irradiance_mol) * Na * photon_energy  # W m⁻² nm⁻¹
    irradiance_mW = irradiance_watts * 1e3  # Convert W to mW

    return irradiance_mW
       
irradiance_value = convert_irradiance(irradiance_vector, wl[ind])
print(irradiance_value.shape)

##AMAZON
h2 = nc.Dataset('Data/S5P_OFFL_L1C_SIFTRS_20240206T172817_20240206T173755_32735_93_010100_20250228T104827_irr.nc')
amazon = h2.groups['amazon']

scanline_nocloud2 = filter_scanlines(amazon, cloud_fraction_threshold=0.4, reflectance_err_threshold=80, sza_threshold=75, vza_threshold=65)
sc_nc2 = len(scanline_nocloud2)
print(sc_nc2)


ref_na2 = amazon.variables["Reflectance"][0, scanline_nocloud2, 223, ind_na].data.tolist()

#for each scanline model the albedo with the noabsobtion window of reflectance
surf_alb2 = np.zeros((sc_nc2, 194))
for i in range(sc_nc2):
    poly_sa = po.polyfit (wl[ind_na], ref_na2[i], sb_order)
    surf_alb = po.polyval (wl[ind], poly_sa) 
    surf_alb2[i, :] = surf_alb

print(surf_alb2.shape)

#Computing tau
mu2 = np.cos(np.radians(amazon.variables["VZA"][0,scanline_nocloud2,223]))
mu_02 = np.cos(np.radians(amazon.variables["SZA"][0,scanline_nocloud2,223]))
mu_matrix2 = np.tile(mu2[:, np.newaxis], (1,194))
mu_0_matrix2 = np.tile(mu_02[:, np.newaxis],(1, 194))
reflectance_matrix2 = amazon.variables['Reflectance'][0, scanline_nocloud2, 223, ind]


tau2 = -np.log(reflectance_matrix2/surf_alb2)/ (np.reciprocal(mu_matrix2)+ np.reciprocal(mu_0_matrix2))
print(tau2.shape)

# Create a directory to save the variables
output_dir = "output_variables"
os.makedirs(output_dir, exist_ok=True)

# Ensure all masked arrays are converted to regular arrays before saving
def save_array(output_path, array):
    if isinstance(array, np.ma.MaskedArray):
        array = array.filled()  
    np.save(output_path, array)

save_array(os.path.join(output_dir, "nocloud_value.npy"), nocloud_value)
save_array(os.path.join(output_dir, "albedo_value.npy"), albedo_value)
save_array(os.path.join(output_dir, "tau_value.npy"), tau_value)
save_array(os.path.join(output_dir, "irradiance_value.npy"), irradiance_value)
save_array(os.path.join(output_dir, "surf_alb2.npy"), surf_alb2)
save_array(os.path.join(output_dir, "tau2.npy"), tau2)
save_array(os.path.join(output_dir, "mu_value.npy"), mu_value)
save_array(os.path.join(output_dir, "mu_0_value.npy"), mu_0_value)
save_array(os.path.join(output_dir, "reflectance_value.npy"), reflectance_value)
save_array(os.path.join(output_dir, "mu_matrix2.npy"), mu_matrix2)
save_array(os.path.join(output_dir, "mu_0_matrix2.npy"), mu_0_matrix2)
save_array(os.path.join(output_dir, "reflectance_matrix2.npy"), reflectance_matrix2)
save_array(os.path.join(output_dir, "scanline_nocloud2.npy"), scanline_nocloud2)
save_array(os.path.join(output_dir, "wl.npy"), wl)
save_array(os.path.join(output_dir, "ind.npy"), ind)
