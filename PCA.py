import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from output_variables import *
import os
import netCDF4 as nc
from sklearn.decomposition import PCA
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


mean_spectrum = np.mean(tau_value, axis=0)
tau_centered = tau_value - mean_spectrum
tau_nan = np.isnan(tau_centered)
print("Number of NaN values in tau_centered:", np.sum(tau_nan))
pca = PCA(n_components=20, svd_solver='full').fit(tau_centered)
components = pca.components_
print(components.shape)


plt.figure(figsize=(15, 20))
for i in range(3):
    plt.subplot(4, 4, i+1)  
    plt.plot(wl[ind], components[i], color='blue')
    plt.title(f'PC {i+1}\nExplained Variance: {pca.explained_variance_ratio_[i]*100:.1f}%')
    plt.xlabel("Feature Index" if wl[ind] is None else "Wavelength (nm)")
    plt.ylabel("Loading")
    plt.grid(True)
plt.tight_layout()
#plt.savefig("PCA_tau")
plt.close()

print(f"Explained variance by 20 PCs: {np.sum(pca.explained_variance_ratio_):.2%}")

tau_pca = pca.transform(tau_centered)  
tau_reconstructed = pca.inverse_transform(tau_pca)  
plt.figure()
plt.plot(tau_reconstructed.T,color='blue', alpha = 0.5, linewidth = 0.1)
plt.savefig("PCA_reconstructed_tau")
plt.close()


plt.figure()
plt.scatter(tau_pca[:, 0], tau_pca[:, 1], alpha=0.6, c='green')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} Variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} Variance)')
#plt.savefig("PCA_scores")
plt.close()

plt.figure()
plt.scatter(tau_pca[:, 0], tau_pca[:, 1], alpha=0.6, c='green', label='Observations')

step = max(1, len(pca.components_[0]) // 10)  
for i, (x, y) in enumerate(zip(pca.components_[0][::step], pca.components_[1][::step])):
    plt.arrow(0, 0, x*5, y*5, color='red', alpha=0.5, head_width=0.1)
    plt.text(x*5.2, y*5.2, f'{wl[ind][i*step]:.0f} nm', color='black', fontsize=8)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} Variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} Variance)')
plt.title('Biplot: PCA Scores and Loadings')
plt.grid(True)
plt.tight_layout()
#plt.savefig("PCA_biplot")
plt.close()

explained_variance_ratios = []
n_components_range = range(1, tau_centered.shape[1] + 1)  
for n in n_components_range:
    pca_temp = PCA(n_components=n).fit(tau_centered)
    explained_variance_ratios.append(np.sum(pca_temp.explained_variance_ratio_))

plt.figure()
plt.plot(n_components_range, explained_variance_ratios, marker='o', linestyle='-', color='blue')
plt.axhline(y=0.996, color='red', linestyle='--', label='Threshold (0.996)')

threshold_index = next((i for i, evr in enumerate(explained_variance_ratios) if evr >= 0.996), None)
if threshold_index is not None:
    plt.scatter(threshold_index + 1, explained_variance_ratios[threshold_index], color='green', label=f'{threshold_index + 1} Components')

plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance vs. Number of Principal Components')
plt.grid(True)
#plt.savefig("PCA_sum_explained_variance")
plt.close()

f_matrix = components
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
            print(popt, m)
            R_fit = reflectance_model(wl[ind], *popt, pixel_index=pixel_index, mu_0_matrix2=mu_0_matrix2, mu_matrix2=mu_matrix2)

            plt.figure()
            plt.figure(figsize=(10, 6))
            plt.plot(wl[ind], reflectance_observed, label="Observed reflectance")
            plt.plot(wl[ind], R_fit, label="Fitted Model Reflectance")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Reflectance")
            plt.legend()
            plt.savefig("pic_PCA/PCA_fit_reflectance_m3_n20")
            plt.close()

            b_fit = np.array(popt[m:n+m])
            attenuation_fit = np.dot(b_fit, f_matrix) 
            transmitance_modelled = np.exp(attenuation_fit)
            print("transmitance_modelled", np.mean(transmitance_modelled))
            print("surface albedo", np.mean(sum(popt[j] * wl[ind]**j for j in range(m+1))))
        
            plt.plot(wl[ind], transmitance_modelled, label="Modelled transmitance")
            plt.xlabel('Wavelength')
            plt.ylabel('Value')
            plt.legend(loc="best")
            plt.savefig("pic_PCA/PCA_fit_transmitance_m3_n20")
            plt.close()

            gaussian_full = np.exp(-0.5 * ((wl[ind] - 737) / 34) ** 2)
            plt.figure()
            plt.plot(wl[ind], popt[-1]*gaussian_full, color='green')
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("SIF Amplitude - mW m⁻² sr⁻¹ nm⁻¹")
            plt.savefig("pic_PCA/PCA_fit_SIF_m3_n20")
            plt.close()

            geom_factor = (1 / mu_matrix2[pixel_index]) / ((1 / mu_matrix2[pixel_index]) + (1 / mu_0_matrix2[pixel_index]))
            baseline_fit = sum(popt[j] * wl[ind]**j for j in range(m+1)) * np.exp(-attenuation_fit)
            fluorescence_fit = (np.pi * popt[-1] * gaussian_full / (mu_0_matrix2[pixel_index] * irradiance_value)) * np.exp(-attenuation_fit * geom_factor)
            plt.figure()
            plt.plot(wl[ind], baseline_fit, label="Baseline", color='purple')
            plt.plot(wl[ind], fluorescence_fit, label="Fluorescence", color='green')
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Value")
            plt.legend()
            plt.savefig("pic_PCA/PCA_fit_baseline_n20")
            plt.close()
      
    SIF_values_per_m[m] = SIF_values_per_scanline
    

    for m in m_values: 
        if m in SIF_values_per_m: 
            SIF_values = SIF_values_per_m[m]
            plt.figure()
            plt.plot(SIF_values, label=f"m={m}")
            plt.xlabel("Scanline Index - Excluding error") 
            plt.ylabel("SIF Value - mW m⁻² sr⁻¹ nm⁻¹")
            plt.legend()
            plt.savefig(f"pic_PCA/SIF_values_m_{m}_PCA_n20.png")
            plt.close()
            print(np.mean(SIF_values), m)
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
        plt.savefig("pic_PCA/PCA_fit_reflectance_n20")
        plt.legend()
        plt.close()
        
        b_fit = np.array(popt[0:n])
        attenuation_fit = np.dot(b_fit, f_matrix)  
        
        plt.figure()
        transmitance_modelled = np.exp(-attenuation_fit)
        plt.plot(wl[ind], transmitance_modelled, label="Modelled transmitance")
        plt.xlabel('Wavelength')
        plt.ylabel('Value')
        plt.legend(loc="best")
        plt.savefig("pic_PCA/PCA_fit_transmitance_n20")
        plt.close()
        
        gaussian_full = np.exp(-0.5 * ((wl[ind] - 737) / 34) ** 2)
        plt.figure()
        plt.plot(wl[ind], popt[-1]*gaussian_full, color='green')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("SIF Amplitude - mW m⁻² sr⁻¹ nm⁻¹")
        plt.savefig("pic_PCA/PCA_fit_SIF_n20")
        plt.close()
        
plt.figure()
plt.plot(SIF_values_per_scanline_A)
plt.xlabel("Scanline Index - Excluding error") 
plt.ylabel("SIF Value - mW m⁻² sr⁻¹ nm⁻¹")
plt.savefig("pic_PCA/PCA_SIF_values_per_scanline_A_n20.png")
plt.close()
print(np.mean(SIF_values_per_scanline_A))  


output_dir = "PCA_SIF_values_n20"
os.makedirs(output_dir, exist_ok=True)
for m, SIF_values in SIF_values_per_m.items():
    np.save(os.path.join(output_dir, f"SIF_values_m_{m}.npy"), SIF_values)

np.save(os.path.join(output_dir, "SIF_values_per_scanline_A.npy"), SIF_values_per_scanline_A)

