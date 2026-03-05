from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

def save_separate_figures(extension="pdf", path=""):
    """ Saves each active matplotlib figure to a separate file. """
    # Получаем номера всех созданных фигур
    fig_nums = plt.get_fignums()

    for i in fig_nums:
        plt.figure(i)
        filename = f"figure_{i}.{extension}"
        plt.savefig(path+filename)
        print(f"Сохранен файл: {filename}")

def compute_fourier_coefficients(f, T, N, t0, out=True, disc_points=[]):
    """
    Считает коэффициенты для вещественного и комплексного рядов Фурье
    
    :param f: Функция вида f(t) = lambda t -> f(t, param1, param2 ...)
    :param T: Период функции
    :param N: Точность/количество членов ряда Фурье
    :param t0: Точка нижней границы периода функции (интегрируем от t0 до t0 + T)
    """
    a_arr = []
    b_arr = []
    c_arr_pos =[]
    c_arr_neg = []
    c_arr = []

    a0, _ = quad(f, t0, t0+T, points=disc_points)
    a0 *= 2/T
    c0 = a0/2
    # c_arr.append(c0)
    
    for n in range(1, N+1):
        omega_n = 2 * np.pi * n / T
        f_cos = lambda t: f(t) * np.cos(omega_n * t)
        f_sin = lambda t: f(t) * np.sin(omega_n * t)

        # Интегрируем
        temp_a_n, _ = quad(f_cos, t0, t0+T, points=disc_points, limit=100)
        temp_b_n, _ = quad(f_sin, t0, t0+T, points=disc_points, limit=100)

        c_n_pos = temp_a_n/T - temp_b_n*1j/T

        a_n = temp_a_n * 2/T
        b_n = temp_b_n * 2/T

        a_arr.append(a_n)
        b_arr.append(b_n)
        c_arr_pos.append(c_n_pos)
        c_arr_neg.append(np.conj(c_n_pos))
    c_arr_neg = c_arr_neg[::-1]
    c_arr = c_arr_neg + [c0] + c_arr_pos

    if out: 
        res = "[CFC] For real Fourier:\n"
        res += f"[CFC] a_0={a0:.2f}\n"
        for n in range(1, N+1):
            res += f"[CFC] a_{n}={a_arr[n-1]:.2f}; b_{n}={b_arr[n-1]:.2f}\n"
        res += "[CFC] For complex Fourier:\n"
        for n in range(-N, N+1):
            res += f"[CFC] c_{n}={c_arr[n+N]:.2f}\n"
        print(res)
    return (a0, a_arr, b_arr, c_arr)

def build_real_fourier(a0, a_n, b_n, T):
    """
    Функция для сборки вещественного ряда Фурье
    
    :param a0: коэффициент a_0
    :param a_n: массив коэффициентов a_n
    :param b_n: массив коэффициентов b_n
    :param T: период функции
    """
    if len(a_n) != len(b_n):
        raise Exception("a_n и b_n должны быть одной размерности")
    def F_N(t):
        result = a0/2
        for n in range(1, len(a_n) + 1):
            omega_n = 2 * np.pi * n / T
            result += a_n[n-1] * np.cos(omega_n * t) + b_n[n-1]*np.sin(omega_n*t)
        return result
    return F_N

def build_complex_fourier(c_n, T):
    N = (len(c_n)-1) // 2
    def G_N(t):
        result = 0
        for n in range(-N, N+1):
            omega_n = 2 * np.pi * n / T
            result += c_n[n + N] * np.exp(1j * omega_n * t)
        return np.real(result)
    return G_N

def draw_graphs(t, f, T, t0, N=[], disc_points=[], N2_title=""):
    """
    Рисует графики функций и их разложений Фурье (только если N != 2)
    
    :param t: Описание
    :param f: Описание
    :param T: Описание
    :param t0: Описание
    :param N: Описание
    """
    for n in N:
        out = n == 2
        if out: print(f"\n{N2_title}\nРазложение 2 порядка:")
        (a0, an, bn, cn) = compute_fourier_coefficients(f, T, n, t0, out=out, disc_points=disc_points)
        plt.figure(figsize=(12, 5))
        plt.grid()
        plt.plot(t, f(t), color="purple", label=r"f$(t)$", linewidth=3)
        if not out:
            plt.title(r"Графики $F_N(t)$ и $G_N(t)$" + f" при N={n}", fontsize=16)
            F_N = build_real_fourier(a0,an,bn,T)
            G_N = build_complex_fourier(cn, T)
            plt.plot(t, F_N(t) , color="green", label=r"$F_N(t)$", linewidth=2)
            plt.plot(t, G_N(t), color="lightblue", label=r"$G_N(t)$", linewidth=1)
            plt.legend(loc="best", fontsize=16)
        else:
            plt.title(N2_title, fontsize=16)
        
        # plt.plot(t, t*[0])
        # plt.axvline(0)
        plt.xlabel("t", fontsize=16, labelpad=10)
        plt.grid(True)
        
        # plt.axis('equal')

N_variations = [2, 3, 10, 20, 50]

## Задание 1.1

def f1(t, t0, t1, t2, T, a, b):
    mod = (t - t0) % T + t0
    return np.where((mod >= t0) & (mod < t1), a, b)

a = 12
b = 6
t0 = 1
t1 = 2
t2 = 3
T1 = t2-t0

t_start = t0 - T1
t_end = t2 + T1

lambda_f1 = lambda t: f1(t, t0, t1, t2, T1, a, b)
t = np.linspace(t_start, t_end, 2000)

draw_graphs(t, lambda_f1, T1, t0, N=N_variations, disc_points=[t0, t1, t2], N2_title=f"1.1 Квадратная волна. T={T1:.2f}")


T2 = 6
t0 = -3
t1 = -2
t2 = -1
t3 = 1
t4 = 2
t5 = 3

def f2(tt, t0, t1, t2, t3, t4, t5, T):
    # Определяем, скаляр или массив
    is_scalar = np.isscalar(tt)
    
    # Приводим к массиву для единообразной обработки
    t_array = np.atleast_1d(tt)
    t_norm = ((t_array - t0) % T) + t0
    
    y = np.zeros_like(t_norm)
    
    # Применяем маски
    mask1 = (t_norm >= t0) & (t_norm < t1)
    y[mask1] = 4 * t_norm[mask1] + 12
    
    mask2 = (t_norm >= t1) & (t_norm < t2)
    y[mask2] = -2 * t_norm[mask2]
    
    mask3 = (t_norm >= t2) & (t_norm < t3)
    y[mask3] = 2
    
    mask4 = (t_norm >= t3) & (t_norm < t4)
    y[mask4] = 2 * t_norm[mask4]
    
    mask5 = (t_norm >= t4) & (t_norm < t5)
    y[mask5] = -4 * t_norm[mask5] + 12
    
    # Возвращаем скаляр, если на входе был скаляр
    return y.item() if is_scalar else y


t_start = t0 - T2
t_end = t5 + T2
t = np.linspace(t_start, t_end, 5000)

lambda_f2 = lambda t: f2(t, t0, t1, t2, t3, t4, t5, T2)
draw_graphs(t, lambda_f2, T2, t0, N=N_variations, disc_points=[t0, t1, t2, t3, t4, t5], N2_title=f"Котики T={T2:.2f}")



def f3(t, T, A):
  # + T/2 для того, чтобы остаток от деления был корректным
  # (с отрицательными работает по правилам кольца. здесь не подходит). потом возвращаем обратно
  t_norm = np.mod(t+T/2, T) - T/2
  return A*2/T * t_norm

T3 = 2*np.pi
t = np.linspace(-T3, T3*2, 1000)
a = 7

lambda_f3 = lambda t: f3(t, T3, a)

draw_graphs(t, lambda_f3, T3, t0=0, N=N_variations, N2_title=f"Пилообразная волна T={T3:.2f}")

def f4(t, T, a):
  t = np.mod(t, T)
  return (1-np.pi)**2*np.sin(a*t) + t
T4=4*np.pi
a = 1

lambda_f4 = lambda t: f4(t, T4, a)

t = np.linspace(-T4, T4*2, 1000)
draw_graphs(t, lambda_f4, T4, t[0], N_variations, N2_title=f"Биение T={T4:.2f}")

# save_separate_figures(path="~/dev/freq_methods/lab1/graphs/")
plt.show()



