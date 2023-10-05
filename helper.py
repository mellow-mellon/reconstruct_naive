import numpy as np
import warnings
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift  


  
def integrateFrankot(zx, zy, pad=512):  
    row, col = zx.shape  
  
    # Pad derivatives to size of FFT  
    padded_zx = np.pad(zx, ((0, pad - row), (0, pad - col)), 'constant')  
    padded_zy = np.pad(zy, ((0, pad - row), (0, pad - col)), 'constant')  
  
    # Fourier transform of gradients for projection  
    Zx = fftshift(fft2(padded_zx))  
    Zy = fftshift(fft2(padded_zy))  
    j = 1j  
  
    # Frequency grid  
    [wx, wy] = np.meshgrid(np.linspace(-np.pi, np.pi, pad),  
                           np.linspace(-np.pi, np.pi, pad))  
    absFreq = wx**2 + wy**2  
  
    # Perform the actual projection  
    with warnings.catch_warnings():  
        warnings.simplefilter('ignore')  
        z = (-j*wx*Zx - j*wy*Zy) / absFreq  
  
    # Set (undefined) mean value of the surface depth to 0  
    z[0, 0] = 0.  
    z = ifftshift(z)  
  
    # Invert the Fourier transform for the depth  
    z = np.real(ifft2(z))  
    z = z[:row, :col]  
  
    return z  