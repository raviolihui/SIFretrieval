import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from output_variables import *
import netCDF4 as nc
import os
from scipy.stats import gaussian_kde
import math
from functools import partial

h2 = nc.Dataset('Data/S5P_OFFL_L1C_SIFTRS_20240206T172817_20240206T173755_32735_93_010100_20250228T104827_irr.nc')
amazon = h2.groups['amazon']

albedo_value = np.load('output_variables/albedo_value.npy')
irradiance_value = np.load('output_variables/irradiance_value.npy')
mu_0_value = np.load('output_variables/mu_0_value.npy')
mu_matrix2 = np.load('output_variables/mu_matrix2.npy')
mu_0_matrix2 = np.load('output_variables/mu_0_matrix2.npy')
mu_value = np.load('output_variables/mu_value.npy')
nocloud_value = np.load('output_variables/nocloud_value.npy')
reflectance_matrix2 = np.load('output_variables/reflectance_matrix2.npy')
reflectance_value = np.load('output_variables/reflectance_value.npy')
scanline_nocloud2 = np.load('output_variables/scanline_nocloud2.npy')
surf_alb2 = np.load('output_variables/surf_alb2.npy')
tau_value = np.load('output_variables/tau_value.npy')
tau2 = np.load('output_variables/tau2.npy')
wl = np.load('output_variables/wl.npy')
ind = np.load('output_variables/ind.npy')


tau_mean = np.mean(tau_value, axis = 0)
plt.figure()
plt.plot(wl[ind], tau_mean)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Mean τ')
plt.savefig("pic_mean/Mean_tau.png")  
plt.close()

mode_spectrum = np.zeros(tau_value.shape[1])  
for i in range(tau_value.shape[1]):
    kde = gaussian_kde(tau_value[:, i]) 
    x = np.linspace(np.min(tau_value[:, i]), np.max(tau_value[:, i]), 100)
    mode_spectrum[i] = x[np.argmax(kde.pdf(x))] 

tau_median = np.median(tau_value, axis=0)

explained_variance_mean = 1 - (np.var(tau_value - tau_mean) / np.var(tau_value))
explained_variance_median = 1 - np.var(tau_value - tau_median) / np.var(tau_value)
explained_variance_KDE = 1 - np.var(tau_value - mode_spectrum) / np.var(tau_value)
explained_variance_mean = np.mean(explained_variance_mean)
explained_variance_median = np.mean(explained_variance_median)
explained_variance_KDE = np.mean(explained_variance_KDE)

plt.figure()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(wl[ind], tau_mean, label="τ mean")
ax.plot(wl[ind], tau_median, label="τ median")
ax.plot(wl[ind], mode_spectrum, label='Mode τ (KDE)')
textstr = '\n'.join((
    f"Explained Variance:",
    f"Mean: {explained_variance_mean:.2%}",
    f"Median: {explained_variance_median:.2%}",
    f"KDE: {explained_variance_KDE:.2%}"))
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Tau value")
ax.legend()
ax.grid(True)
plt.savefig("pic_mean/Mean_tau_with_explained_variance.png")
plt.close()

tau_tile = np.tile(tau_mean, (len(scanline_nocloud2), 1))
transmitance_matrix = np.exp(-tau_tile)
up_transmitance_matrix = np.exp(-tau_tile*(np.reciprocal(mu_matrix2)/(np.reciprocal(mu_matrix2) + np.reciprocal(mu_0_matrix2))))
irradiance_matrix = np.tile(irradiance_value, (len(scanline_nocloud2), 1))
diff = reflectance_matrix2 - surf_alb2*transmitance_matrix
I = irradiance_matrix*mu_0_matrix2*diff/(up_transmitance_matrix*math.pi) 
plt.plot(wl[ind],I.mean(axis=0))
plt.xlabel('Wavelength (nm)')
plt.ylabel("SIF - mW m⁻² sr⁻¹ nm⁻¹")
plt.savefig("pic_mean/SIF_from_equation.png")
plt.close()


f_matrix = tau_mean.reshape(1, 194)
print(f_matrix.shape)
n = len(f_matrix)
m_values = [1, 2, 3, 4] 
SIF_values_per_m = {}

def reflectance_model(lam, *params, pixel_index, mu_0_matrix2, mu_matrix2):
            a = np.array(params[0:m+1])
            b = np.array(params[m+1:n+m+1])
            c = params[n+m+1]
            attenuation = np.dot(b, f_matrix)
            poly_term = sum(a[j] * lam**j for j in range(m+1))
            baseline = poly_term * np.exp(-attenuation)
            gaussian = np.exp(-0.5 * ((lam - 737) / 34) ** 2)
            geom_factor = (1 / mu_matrix2[pixel_index]) / ((1 / mu_matrix2[pixel_index]) + (1 / mu_0_matrix2[pixel_index]))
            fluorescence = (np.pi * c * gaussian / (mu_0_matrix2[pixel_index] * irradiance_value)) * np.exp(-attenuation * geom_factor)
            return baseline + fluorescence

for m in m_values:
    SIF_values_per_scanline = []
    for pixel_index, i in enumerate(scanline_nocloud2):
        reflectance_observed = amazon.variables['Reflectance'][0, i, 223, ind].data
        
        p0 = [0.5] * (m+1) + [0.5] * n + [0.5]
        model_func = partial(reflectance_model, pixel_index=pixel_index, mu_0_matrix2=mu_0_matrix2, mu_matrix2=mu_matrix2)
        popt, pcov = curve_fit(model_func, wl[ind], reflectance_observed, p0=p0)
        SIF_values_per_scanline.append(popt[-1])

        if m == 2 and pixel_index == 100:
            print("Fitted parameters:")
            print(popt)
            R_fit = reflectance_model(wl[ind], *popt, pixel_index=pixel_index, mu_0_matrix2=mu_0_matrix2, mu_matrix2=mu_matrix2)

            plt.figure()
            plt.figure(figsize=(10, 6))
            plt.plot(wl[ind], reflectance_observed, label="Observed reflectance")
            plt.plot(wl[ind], R_fit, label="Fitted Model Reflectance")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Reflectance")
            plt.legend()
            plt.savefig("pic_mean/Mean_fit_reflectance_m3")
            plt.close()

            b_fit = np.array(popt[m:n+m])
            attenuation_fit = np.dot(b_fit, f_matrix)  
            transmitance_modelled = np.exp(-attenuation_fit)

            print("surface albedo:", np.mean(sum(popt[j] * wl[ind]**j for j in range(m+1))))
            print("transmittance:",np.mean(transmitance_modelled))
            plt.figure()
            plt.plot(wl[ind], transmitance_modelled, label="Modelled transmitance")
            plt.xlabel('Wavelength')
            plt.ylabel('Value')
            plt.legend(loc="best")
            plt.savefig("pic_mean/mean_fit_transmitance_m2")
            plt.close()

            gaussian_full = np.exp(-0.5 * ((wl[ind] - 737) / 34) ** 2)

            plt.figure()  
            plt.plot(wl[ind], popt[-1]*gaussian_full, color='green')
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("SIF Amplitude - mW m⁻² sr⁻¹ nm⁻¹")
            plt.savefig("pic_mean/Mean_fit_SIF_m3")
            plt.close()

            geom_factor = (1 / mu_matrix2[100]) / ((1 / mu_matrix2[100]) + (1 / mu_0_matrix2[100]))
            baseline_fit = sum(popt[j] * wl[ind]**j for j in range(m+1)) * np.exp(-attenuation_fit)
            fluorescence_fit = (np.pi * popt[-1] * gaussian_full / (mu_0_matrix2[100] * irradiance_value)) * np.exp(-attenuation_fit * geom_factor)
            plt.figure()
            plt.plot(wl[ind], baseline_fit, label="Baseline", color='purple')
            plt.plot(wl[ind], fluorescence_fit, label="Fluorescence", color='green')
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Value")
            plt.legend()
            plt.savefig("pic_mean/Mean_fit_baseline")
            plt.close()
      
    SIF_values_per_m[m] = SIF_values_per_scanline
   

    for m in m_values: 
        if m in SIF_values_per_m: 
            SIF_values = SIF_values_per_m[m]
            plt.figure()
            plt.plot(SIF_values, label=f"m={m}")
            plt.xlabel("Scanline Index - Excluding error") 
            plt.ylabel("SIF Value -  mW m⁻² sr⁻¹ nm⁻¹")
            plt.legend()
            plt.savefig(f"pic_mean/SIF_values_m_{m}_Mean.png")
            plt.close()
            print(f"Mean SIF value for m={m}: {np.mean(SIF_values)}")
        else:
            print(f"Warning: No SIF values found for m={m}")
  

    



#fix albedo
f_matrix = tau_mean.reshape(1, 194)
print(f_matrix.shape)
n = len(f_matrix)

def reflectance_model(lam, *params, pixel_index, mu_0_matrix2, mu_matrix2):

        b = np.array(params[0:n])
        c = params[n]
        
        attenuation = np.dot(b, f_matrix) 
        baseline = surf_alb2[pixel_index] * np.exp(-attenuation)
        gaussian = np.exp(-0.5 * ((lam - 737) / 34) ** 2)
        geom_factor = (1 / mu_matrix2[pixel_index]) / ((1 / mu_matrix2[pixel_index]) + (1 / mu_0_matrix2[pixel_index]))
        fluorescence = (np.pi * c * gaussian /( mu_0_matrix2[pixel_index] * irradiance_value)) * np.exp(-attenuation * geom_factor)
        return baseline + fluorescence

SIF_values_per_scanline_A = []
for pixel_index, i in enumerate(scanline_nocloud2):
    reflectance_observed = amazon.variables['Reflectance'][0, i, 223,ind].data

    p0 = [0.5] * n + [0.5]
    model_func = partial(reflectance_model, pixel_index=pixel_index, mu_0_matrix2=mu_0_matrix2, mu_matrix2=mu_matrix2)
    popt, pcov = curve_fit(model_func, wl[ind], reflectance_observed, p0=p0)
    SIF_values_per_scanline_A.append(popt[-1])
    
    if pixel_index == 100:
        print("Fitted parameters:")
        print(popt)
                
        R_fit = reflectance_model(wl[ind], *popt, pixel_index=pixel_index, mu_0_matrix2=mu_0_matrix2, mu_matrix2=mu_matrix2)
        
        plt.figure()
        plt.plot(wl[ind], reflectance_observed, label="Observed reflectance")
        plt.plot(wl[ind], R_fit, label="Fitted Model Reflectance")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.savefig("pic_mean/Mean_fit_reflectance")
        plt.legend()
        plt.close()
        
        gaussian_full = np.exp(-0.5 * ((wl[ind] - 737) / 34) ** 2)
        
        plt.figure()
        plt.plot(wl[ind], popt[-1]*gaussian_full, color='green')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("SIF Amplitude -  mW m⁻² sr⁻¹ nm⁻¹")
        plt.savefig("pic_mean/Mean_fit_SIF")
        plt.close()
        
plt.figure()
plt.plot(SIF_values_per_scanline_A)
plt.xlabel("Scanline Index - Excluding error") 
plt.ylabel("SIF Value -  mW m⁻² sr⁻¹ nm⁻¹")
plt.savefig("pic_mean/Mean_SIF_values_per_scanline_A.png")
plt.close()
print(np.mean(SIF_values_per_scanline_A))  




output_dir = "Mean_SIF_values"
os.makedirs(output_dir, exist_ok=True)
for m, SIF_values in SIF_values_per_m.items():
    np.save(os.path.join(output_dir, f"SIF_values_m_{m}.npy"), SIF_values)

np.save(os.path.join(output_dir, "SIF_values_per_scanline_A.npy"), SIF_values_per_scanline_A)


