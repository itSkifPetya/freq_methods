import librosa as lb
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from joblib import Memory
import tempfile

cache_dir = tempfile.gettempdir() + 'freq_methods/lab2/fourier_cache'
memory = Memory(cache_dir, verbose=0)

@memory.cache
def find_frequencies(f, t, dV, V):
    Yarr = []
    Varr = []
    for v in np.arange(0, V, dV):
        Varr.append(v)
        Yarr.append(trapezoid(f * np.exp(-1j * 2 * np.pi * v * t), t)) 

    return (np.array(Varr), np.array(Yarr))

chord = lambda n: f'lab2/chords/Аккорд ({n}).mp3'
chord_num = 6
f, sr = lb.load(chord(chord_num), mono=True)

dt = 1/sr

t = np.linspace(0, len(f)/sr, len(f))
plt.figure(figsize=(14, 5))
plt.title(f"Звуковая волна аккорда №{chord_num}")
plt.tight_layout()
plt.plot(t, f)

dv = 0.1
V = 500
Varr, Yarr = find_frequencies(f, t, dv, V)
ampl = np.abs(Yarr)

threshold = (np.quantile(ampl, 0.75) + (np.quantile(ampl, 0.75) - np.quantile(ampl, 0.25))* 1.5)*3
threshold = np.max(ampl)*0.15
# threshold = np.percentile(ampl, 0.9)
print(threshold)
peaks, props = find_peaks(ampl, height=threshold, distance=dv)
note_freq = Varr[peaks]
# print(note_freq)
freq_values = list(props.values())
# print(freq_values)
for i in range(0, len(note_freq)):
    note = lb.hz_to_note(note_freq[i])
    print(f"{note_freq[i]:0.2f}Hz: интенсивность {freq_values[0][i]:.3f}; note {note}")

plt.figure(figsize=(14, 5))
plt.title("Частоты, содержащиеся в записи")
plt.plot(Varr, ampl)
plt.plot(Varr, [threshold]*len(Varr), color="red", label="Threshold")
plt.xlabel(r"$\nu$, Гц")
plt.ylabel("Интенсивность сигнала")
plt.show()