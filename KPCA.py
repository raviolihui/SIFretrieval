from sklearn.decomposition import KernelPCA
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from output_variables import *
import netCDF4 as nc
import os
from sklearn.metrics.pairwise import pairwise_distances
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

dists = pairwise_distances(tau_value)
median_dist = np.median(dists)
gamma_guess = 1 / (2 * median_dist**2)
print(f"Gamma guess: {gamma_guess}")

print(tau_value.shape)
kpca = KernelPCA(kernel='rbf', fit_inverse_transform=False, n_components=None, gamma = gamma_guess,)  # not using inverse transform 
tau_reduced = kpca.fit(tau_value)
score_kernel_pca = kpca.transform(tau_value)
eigenvectors = np.dot(tau_value.T, kpca.eigenvectors_) 

eigenvectors = tau_value.T @ kpca.eigenvectors_
print(eigenvectors.shape)
eigenvalues = kpca.eigenvalues_
sklearn_proj = kpca.transform(tau_value)[:, 10]
print(sklearn_proj.shape)


plt.figure()
plt.scatter(score_kernel_pca[:,0],score_kernel_pca[:,1] ,cmap='viridis')
plt.title("Projection onto PCs (kernel)")
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")
plt.savefig("pic_kpca/projection_KPC.png")
plt.close()


explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
sorted_indices = np.argsort(explained_variance_ratio)[::-1][:10]
top10_PC = explained_variance_ratio[sorted_indices]
print(top10_PC.shape)


cumulative_explained_variance = np.cumsum(explained_variance_ratio[sorted_indices[:10]])
print("Cumulative explained variance of the first 10 components:", cumulative_explained_variance)
plt.figure()
plt.bar(range(10), top10_PC, color='skyblue')
plt.xlabel("Component Index")
plt.ylabel("Explained Variance Ratio")
plt.xticks(range(10), range(1,11))
plt.grid(axis='y', linestyle='--')
plt.savefig("pic_kpca/10_PC_KPCA.png")
plt.close()


plt.figure()
components = []
for i, idx in enumerate(sorted_indices):
    components.append(eigenvectors[:, idx])
    if i in [0, 1, 2]:
        plt.figure(figsize=(12, 4))
        plt.bar(range(194), eigenvectors[:, idx])
        plt.title(f"Eigenvector {i+1} (λ = {explained_variance_ratio[idx]:.2f})")
        plt.xlabel("Feature Index")
        plt.ylabel("Component Value")
        plt.grid(axis='y', linestyle='--')
        plt.savefig(f"pic_kpca/Dual_PC_{i+1}_KPCA.png")
        plt.close()


f_matrix = np.array(components)
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
            plt.savefig("pic_kpca/KPCA_fit_reflectance_m2")
            plt.close()

            b_fit = np.array(popt[m:n+m])
            attenuation_fit = np.dot(b_fit, f_matrix)  

            plt.figure()
            transmitance_modelled = np.exp(-attenuation_fit)
            plt.plot(wl[ind], transmitance_modelled, label="Modelled transmitance")
            plt.xlabel('Wavelength')
            plt.ylabel('Value')
            plt.legend(loc="best")
            plt.savefig("pic_kpca/KPCA_fit_transmitance_m2")
            plt.close()

            gaussian_full = np.exp(-0.5 * ((wl[ind] - 737) / 34) ** 2)
            
            
            plt.figure()
            plt.plot(wl[ind], popt[-1]*gaussian_full, color='green')
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("SIF Amplitude - mW m⁻² nm⁻¹")
            plt.savefig("pic_kpca/KPCA_fit_SIF_m2")
            plt.close()
            print("surface albedo:", sum(popt[j] * wl[ind]**j for j in range(m+1)).shape, np.mean(sum(popt[j] * wl[ind]**j for j in range(m+1)), axis=0))

            
            geom_factor = (1 / mu_matrix2[100]) / ((1 / mu_matrix2[100]) + (1 / mu_0_matrix2[100]))
            baseline_fit = sum(popt[j] * wl[ind]**j for j in range(m+1)) * np.exp(-attenuation_fit)
            fluorescence_fit = (np.pi * popt[-1] * gaussian_full / (mu_0_matrix2[100] * irradiance_value)) * np.exp(-attenuation_fit * geom_factor)

            plt.figure()
            plt.plot(wl[ind], baseline_fit, label="Baseline", color='purple')
            plt.plot(wl[ind], fluorescence_fit, label="Fluorescence", color='green')
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Value")
            plt.legend()
            plt.savefig("pic_kpca/KPCA_fit_baseline")
            plt.close()
      
    SIF_values_per_m[m] = SIF_values_per_scanline
    
    for m in m_values: 
        if m in SIF_values_per_m:  
            SIF_values = SIF_values_per_m[m]
            plt.figure()
            plt.plot(SIF_values, label=f"m={m}")
            plt.xlabel("Scanline Index - Excluding error") 
            plt.ylabel("SIF Value - mW m⁻² nm⁻¹")
            plt.legend()
            plt.savefig(f"pic_kpca/SIF_values_m_{m}_KPCA.png")
            plt.close()
            print(f"Mean SIF value for m={m}: {np.mean(SIF_values)}")
        else:
            print(f"Warning: No SIF values found for m={m}")
  

    




#fixed albedo: 
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
    
    p0 = [0] * n + [0]
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
        plt.savefig("pic_kpca/KPCA_fit_reflectance")
        plt.legend()
        plt.close()
    
        gaussian_full = np.exp(-0.5 * ((wl[ind] - 737) / 34) ** 2)
        
        plt.figure()
        plt.plot(wl[ind], popt[-1]*gaussian_full, color='green')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("SIF Amplitude - mW m⁻² nm⁻¹")
        plt.savefig("pic_kpca/KPCA_fit_SIF")
        plt.close()
        
plt.figure()
plt.plot(SIF_values_per_scanline_A)
plt.xlabel("Scanline Index - Excluding error") 
plt.ylabel("SIF Value - mW m⁻² nm⁻¹")
plt.savefig("pic_kpca/KPCA_SIF_values_per_scanline_A.png")
plt.close()
print(np.mean(SIF_values_per_scanline_A))  


output_dir = "KPCA_SIF_values_D1"
os.makedirs(output_dir, exist_ok=True)
for m, SIF_values in SIF_values_per_m.items():
    np.save(os.path.join(output_dir, f"SIF_values_m_{m}.npy"), SIF_values)

np.save(os.path.join(output_dir, "SIF_values_per_scanline_A.npy"), SIF_values_per_scanline_A)