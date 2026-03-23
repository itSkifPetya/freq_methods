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

plt.plot(Varr, ampl, label=r"|\hat f(\nu)|", linewidth=2.5)

plt.xlabel(r"$\nu$, Гц")
plt.ylabel(r"|$\hat f(\nu)$|")
plt.legend()
plt.tight_layout()
# plt.show()


### ЗАДАНИЕ 2

def card_sin(t, a, b, c): 
    return a * np.sinc(b*t + c)

@memory.cache
def compute_uni_Fourier_unitary(f, t, dw, w_max):
    """
    Вычисляет унитарное преобразование Фурье к угловой частоте ω
    ĝ(ω) = (1/√(2π)) * ∫ g(t) * e^(-iωt) dt
    """
    ImgArr = []
    wArr = []
    for w in np.arange(0, w_max, dw):
        wArr.append(w)
        integrand = f * np.exp(-1j * w * t)
        integral = trapezoid(integrand, t)
        fourier_coeff = integral / np.sqrt(2 * np.pi)
        ImgArr.append(fourier_coeff)
    return (np.array(wArr), np.array(ImgArr))

# Параметры
a, b = 3, 1
c_values = [0, 1, 2]

print("\n" + "="*60)
print("ЗАДАНИЕ 2: Фурье-образ сдвинутой функции")
print("="*60)
print(f"Функция: g(t) = {a}*sinc({b}*t + c)")
print(f"Значения сдвига c: {c_values}\n")

# Временная сетка
t = np.linspace(-5, 5, 1000)
dw = 0.05
w_max = 8

# Аналитическое выражение
print("Аналитическое выражение:")
print("g(t) = sinc(t + c)")
print("ĝ(ω) = e^(iωc) * ŝinc(ω)")
print("Сдвиг на c в области времени => фазовый множитель e^(iωc) в области частот\n")

# 1. График оригиналов
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

# 2. Для каждого c вычисляем и рисуем Фурье-образ
for c in c_values:
    t_calc = np.linspace(-10, 10, 3000)
    g_t = card_sin(t_calc, a, b, c)
    
    w_arr, g_hat = compute_uni_Fourier_unitary(g_t, t_calc, dw, w_max)
    
    real_part = np.real(g_hat)
    imag_part = np.imag(g_hat)
    modulus = np.abs(g_hat)
    
    # Одна фигура с тремя subplots
    plt.figure(figsize=(14, 10))
    
    # Вещественная часть
    plt.subplot(3, 1, 1)
    plt.plot(w_arr, real_part, 'b-', linewidth=2)
    plt.title(rf"Фурье-образ $\hat{{g}}(\omega)$ при $c = {c}$")
    plt.ylabel(r"$\mathrm{Re}(\hat{g}(\omega))$")
    plt.grid(True, alpha=0.3)
    
    # Мнимая часть
    plt.subplot(3, 1, 2)
    plt.plot(w_arr, imag_part, 'r-', linewidth=2)
    plt.ylabel(r"$\mathrm{Im}(\hat{g}(\omega))$")
    plt.grid(True, alpha=0.3)
    
    # Модуль
    plt.subplot(3, 1, 3)
    plt.plot(w_arr, modulus, 'g-', linewidth=2)
    plt.ylabel(r"$|\hat{g}(\omega)|$")
    plt.xlabel(r"$\omega$, рад/с")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()

print("Вывод: сдвиг c меняет фазу Фурье-образа (вещественную и мнимую части),")
print("но модуль |ĝ(ω)| остается неизменным для всех значений c")
# plt.show()
save_separate_figures(path="lab2/graphs/")