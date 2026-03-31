from scipy.fft import fft, fftshift, ifft, ifftshift
import matplotlib.pyplot as plt
import numpy as np

from librosa import load
import sounddevice as sd

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

def setup_plot(figsize=(14, 6)):
    plt.figure(figsize=figsize)
    plt.grid()

muha_f, muha_sr = load("lab3/MUHA.wav", mono=True)
dt = 1/muha_sr
T = len(muha_f)*dt
t = np.linspace(0, T, len(muha_f))

freq = fftshift(np.fft.fftfreq(len(muha_f), dt))

setup_plot()
plt.title("Исходный сигнал MUHA.wav")
plt.plot(t, np.real(muha_f), linewidth=2, label="Исходный сигнал")
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$f(t)$")
img_muha_f = fftshift(fft(muha_f))

setup_plot()
plt.title("Модуль образа исходного сигнала")
plt.plot(freq, np.abs(img_muha_f), linewidth=2, label="Образ исходника")
plt.xlim([-4000, 4000])
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"|$\hat f(t)$|")
HIGH_FREQ = 3400
LOW_FREQ = 300

### Applying freq filter:
mask = np.zeros_like(img_muha_f)
ind = np.where((np.abs(freq) < HIGH_FREQ) & (np.abs(freq) > LOW_FREQ))
mask[ind]=1
filtered_muha_img = img_muha_f * mask
mask[ind]=max(np.abs(filtered_muha_img))
# print(filtered_muha_img)

setup_plot()
plt.title("Модуль образа после применения фильтра")
plt.plot(freq, np.abs(filtered_muha_img), linewidth=2, label="Фильтрованный образ")
plt.plot(freq, mask, linewidth=1.5, color="orange", label="Полосовой фильтр")
plt.xlim([-4000, 4000])
plt.xlabel(r"$t$")
plt.ylabel(r"|$\hat f(t)$|")
plt.legend()


filtered_muha = ifft(ifftshift(filtered_muha_img))

setup_plot()
plt.title("Отфильтрованный сигнал MUHA.wav")
plt.plot(t, np.real(filtered_muha), linewidth=2, label="Отфильтрованный сигнал")
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$f(t)$")


setup_plot()
plt.title("Сравнение сигналов до и после обработки")
plt.xlabel(r"$t$")
plt.ylabel(r"$f(t)$, $f_t(t)$")

plt.plot(t, np.real(muha_f), linewidth=4, color="orange", label="Исходный сигнал")
plt.plot(t, np.real(filtered_muha), linewidth=3, color="green", label="Обработанный сигнал")
plt.legend()
plt.fill_between(t, np.real(filtered_muha), np.real(muha_f), alpha=0.4, color="orange")

sd.play(np.real(muha_f), muha_sr)
sd.wait()
sd.play(np.real(filtered_muha), muha_sr)
sd.wait()


save_separate_figures(path="lab3/pete-graphs/")
# plt.show()


