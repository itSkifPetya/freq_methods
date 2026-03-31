import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.signal import find_peaks

import librosa as lb

from joblib import Memory
import tempfile

plt.rcParams.update({'font.size': 16})

def save_separate_figures(extension="pdf", path=""):
    """ Saves each active matplotlib figure to a separate file. """
    # Получаем номера всех созданных фигур
    fig_nums = plt.get_fignums()

    for i in fig_nums:
        plt.figure(i)
        filename = f"figure_{i}.{extension}"
        plt.savefig(path+filename)
        print(f"Сохранен файл: {filename}")


### ЗАДАНИЕ 3
# cache_dir = tempfile.gettempdir() + 'freq_methods/lab2/fourier_cache'
cache_dir = tempfile.gettempdir() + '/freq_methods/lab2/fourier_cache'
memory = Memory(cache_dir, verbose=0)

@memory.cache
def compute_Fourier_image(f, t, dV, V):
    ImgArr = []
    vArr = []
    for v in np.arange(16, V, dV):
        vArr.append(v)
        ImgArr.append(trapezoid(f * np.exp(-1j * 2 * np.pi * v * t), t)) 
    return (np.array(vArr), np.array(ImgArr))

chord = lambda n: f'lab2/chords/Аккорд ({n}).mp3'
chord_num = 6
f, sr = lb.load(chord(chord_num), mono=True)
t = np.linspace(0, len(f)/sr, len(f))
plt.figure(figsize=(14, 6))
plt.title(r"График функции $f(t)\;-\;$" + f"Звуковая волна аккорда №{chord_num}")


plt.plot(t, f, linewidth=0.5)
plt.xlabel(r"$t$, сек")
plt.ylabel(r"$f(t)$")
plt.tight_layout()
dv = 0.1
V = 500
Varr, Yarr = compute_Fourier_image(f, t, dv, V)
ampl = np.abs(Yarr)


# sorted_ampl = np.sort(ampl)
# cumsum = np.cumsum(sorted_ampl)
# cumsum /= cumsum[-1]
# plt.figure()
# plt.grid()
# plt.plot(cumsum*100, sorted_ampl)



# log_ampl = np.log10(ampl + 1e-15)
# plt.figure()
# plt.semilogy(Varr, ampl)




plt.figure(figsize=(12, 8))
plt.title(r"График модуля образа $|\hat f(\nu)$|")
plt.grid()
notes = set()
for p in [1, 2, 3]:
    threshold = np.percentile(ampl, 100 - p)
    peaks, _ = find_peaks(ampl, height=threshold)
    print(f"\nНайдено {len(peaks)} пиков, порог {threshold:.4f}. \nСодержащиеся ноты:")
    plt.plot(Varr, [threshold]*len(Varr), label=f"Порог {threshold:.4f} при перцентиле {100-p}%", linewidth=1.5)
    for freq in sorted(Varr[peaks]):
        note = lb.hz_to_note(freq)
        notes.add(note)
        # print(f"  {freq:7.2f} Hz → {note}")
    print(*notes)

# threshold = (np.quantile(ampl, 0.75) + (np.quantile(ampl, 0.75) - np.quantile(ampl, 0.25))* 1.5)*3
# threshold = np.max(ampl)*0.15
# # threshold = np.percentile(ampl, 0.9)
# print(threshold)
# peaks, props = find_peaks(ampl, height=threshold)
# note_freq = Varr[peaks]
# # print(note_freq)
# freq_values = list(props.values())
# # print(freq_values)
# for i in range(0, len(note_freq)):
#     note = lb.hz_to_note(note_freq[i])
#     print(f"{note_freq[i]:0.2f}Hz: интенсивность {freq_values[0][i]:.3f}; note {note}")

plt.plot(Varr, ampl, label=r"$|\hat f(\nu)|$", linewidth=2.5)

plt.xlabel(r"$\nu$, Гц")
plt.ylabel(r"|$\hat f(\nu)$|")
plt.legend()
plt.tight_layout()
# plt.show()


### ЗАДАНИЕ 2

def card_sin(t, a, b, c): 
    return a * np.sinc(b*t + c)

def analytic_Fourier_card_sin(omega, a, b, c):
    Rect = lambda w, a, b: np.where(np.abs(w) <= b, np.sqrt(np.pi/2) * a/b, 0)
    return np.exp(1j*omega*c)*Rect(omega, a, b)

a, b = 3, 1
c_values = [0, 1, 2]
t = np.linspace(-10, 10, 1000)
omega = np.linspace(-5, 5, 1000)

plt.figure(figsize=(14, 6))
for c in c_values:
    g_t = card_sin(t, a, b, c)
    plt.plot(t, g_t, label=f'c = {c}', linewidth=2)
plt.title(r"Оригиналы $g(t) = \mathrm{sinc}(t + c)$")
plt.xlabel(r"$t$")
plt.ylabel(r"$g(t)$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()


for c in c_values:
    img_g_t = analytic_Fourier_card_sin(omega, a, b, c)

    real_part = np.real(img_g_t)
    imag_part = np.imag(img_g_t)
    modulus = np.abs(img_g_t)
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(omega, real_part, 'b-', linewidth=2)
    plt.title(rf"Фурье-образ $\hat{{g}}(\omega)$ при $c = {c}$")
    plt.ylabel(r"$\mathrm{Re}(\hat{g}(\omega))$")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(omega, imag_part, 'r-', linewidth=2)
    plt.ylabel(r"$\mathrm{Im}(\hat{g}(\omega))$")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(omega, modulus, 'g-', linewidth=2)
    plt.ylabel(r"$|\hat{g}(\omega)|$")
    plt.xlabel(r"$\omega$, рад/с")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()

# plt.show()
save_separate_figures(path="lab2/pete-graphs/")