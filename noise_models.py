import numpy as np
import scipy.stats as stats
#---------------- NEW Version of Error Model -------------

#Magnitude: Mu=[-0.004, 0.006] , Sigma=[0.0025 0.0025] , Weight = [0.4, 0.6]
#Angle : Mu = [-0.2, 0.3] , Sigma= [0.12, 0.12] , Weight = [0.4, 0.6]

def Varying_GMM_Error_vector_Creation(sample_size, Mu_vector, Sigma_vector, weight_vector): ## The GMM error 
    error = sample_size
    nsamp = sample_size
    #np.random.seed(seed=seed_number)
    data1 = stats.norm.rvs(loc=Mu_vector[0], scale=Sigma_vector[0], size=round(weight_vector[0]*sample_size)) 
    data1 = data1.reshape(-1, 1)
    data2 = stats.norm.rvs(loc=Mu_vector[1], scale=Sigma_vector[1], size=round(weight_vector[1]*sample_size)) 
    data2 = data2.reshape(-1, 1)
#    data3 = stats.norm.rvs(loc=Mu_vector[2], scale=Sigma_vector[2], size=round(weight_vector[2]*sample_size)) 
#    data3 = data3.reshape(-1, 1)
#    GMM_vector = np.vstack((data1, data2, data3)) 
    GMM_vector = np.vstack((data1, data2)) 
    error = GMM_vector
    np.random.shuffle(error)
    #error = np.random.shuffle(error)
    return(error)


def NoisyAng(x):
    for i in range (x.shape[1]):
        y = Varying_GMM_Error_vector_Creation(x.shape[0], [-0.2, 0.3], [0.12, 0.12], [0.4, 0.6])
        x[:,i] = x[:,i] + y.reshape((x.shape[0],))
        
    return x

def NoisyMag(x):
    for i in range (x.shape[1]):
        y = Varying_GMM_Error_vector_Creation(x.shape[0], [-0.004, 0.006], [0.0025, 0.0025], [0.4, 0.6])
        y = 1+y
        x[:,i] = np.multiply(y, x[:,i].reshape(-1,1)).reshape(-1,)
        # x[:,i] = x[:,i] + y.reshape((x.shape[0],))
        
        
    return x

def noise_gaussian_mag(X, range1, mu1):
    sigma1 = range1
    l,b  = X.shape
    noise_xtot = np.zeros([l,b])
    for q in range(b):
      noise_cols = np.random.normal(mu1, sigma1, l)
      noise_xtot[:,q] = 1+ noise_cols
    X_noisy = X*noise_xtot 
    return X_noisy

def noise_gaussian_ang(X, range1, mu1):
    sigma1 = range1
    l,b  = X.shape
    noise_xtot = np.zeros([l,b])
    for q in range(b):
      noise_cols = np.random.normal(mu1, sigma1, l)
      noise_xtot[:,q] = noise_cols
    X_noisy = X + noise_xtot 
    return X_noisy

def polar_to_complex(magnitude_array, angle_array_degrees):
    # Convert angle from degrees to radians
    angle_array_radians = np.radians(angle_array_degrees)

    # Calculate real and imaginary components
    real_part = magnitude_array * np.cos(angle_array_radians)
    imaginary_part = magnitude_array * np.sin(angle_array_radians)

    return real_part, imaginary_part

def GMM_real_imag(true_real, true_imag):
    for i in range (true_real.shape[1]):
        mag_error = Varying_GMM_Error_vector_Creation(true_real.shape[0], [-0.2, 0.3], [0.12, 0.12], [0.4, 0.6])
        ang_error = Varying_GMM_Error_vector_Creation(true_real.shape[0], [-0.004, 0.006], [0.0025, 0.0025], [0.4, 0.6])

        real_error, imag_error = polar_to_complex(mag_error, ang_error)

        true_real[:,i] = true_real[:,i] + real_error.reshape((true_real.shape[0],))
        true_imag[:,i] = true_imag[:,i] + imag_error.reshape((true_imag.shape[0],))
        
    return true_real, true_imag



