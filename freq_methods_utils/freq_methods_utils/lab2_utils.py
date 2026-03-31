from .imports import *
from scipy.integrate import trapezoid
from joblib import Memory
import tempfile

cache_dir = tempfile.gettempdir() + '/freq_methods/lab2/fourier_cache'
memory = Memory(cache_dir, verbose=0)

@memory.cache
def compute_Fourier_image(f, t, V0, dV, V):
    
    ImgArr = []
    vArr = []
    for v in np.arange(V0, V, dV):
        vArr.append(v)
        ImgArr.append(trapezoid(f * np.exp(-1j * 2 * np.pi * v * t), t)) 
    return (np.array(vArr), np.array(ImgArr))