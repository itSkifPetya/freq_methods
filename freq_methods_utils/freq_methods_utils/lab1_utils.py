from .imports import *
from scipy.integrate import quad

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
            res += f"[CFC] a_{n}={a_arr[n-1]}; b_{n}={b_arr[n-1]}\n"
        res += "[CFC] For complex Fourier:\n"
        for n in range(-N, N+1):
            res += f"[CFC] c_{n}={c_arr[n+N]}\n"

        print(res)
    return (a0, a_arr, b_arr, c_arr)

def check_parseval(f, T, N, t0, out=True, disc_points=[]):
    """
    Проверяет равенство Парсеваля для функции f
    
    :param f: Функция вида f(t) = lambda t -> f(t, param1, param2 ...)
    :param T: Период функции
    :param N: Точность/количество членов ряда Фурье
    :param t0: Точка нижней границы периода функции
    """
    a0, a_arr, b_arr, c_arr = compute_fourier_coefficients(f, T, N, t0, out=False, disc_points=disc_points)
    
    f_squared = lambda t: f(t)**2
    left_part, _ = quad(f_squared, t0, t0+T, points=disc_points, limit=100)
    left_part /= T
    
    right_part_real = a0**2 / 4
    for n in range(len(a_arr)):
        right_part_real += (a_arr[n]**2 + b_arr[n]**2) / 2
    
    right_part_complex = 0
    for n in range(len(c_arr)):
        right_part_complex += np.abs(c_arr[n])**2
    
    if out:
        res = "\n[Parseval] Left part (integral): "
        res += f"{left_part:.6f}\n"
        res += f"[Parseval] Right part (real Fourier): {right_part_real:.6f}\n"
        res += f"[Parseval] Right part (complex Fourier): {right_part_complex:.6f}\n"
        res += f"[Parseval] Difference (real): {abs(left_part - right_part_real):.6e}\n"
        res += f"[Parseval] Difference (complex): {abs(left_part - right_part_complex):.6e}"
        print(res)
    
    return (left_part, right_part_real, right_part_complex)

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
        out = n == 2 or n == max(N)
        if out: print(f"\n{N2_title}\nРазложение 2 порядка:")
        (a0, an, bn, cn) = compute_fourier_coefficients(f, T, n, t0, out=out, disc_points=disc_points)
        plt.figure(figsize=(12, 5))
        plt.grid()
        plt.plot(t, f(t), color="purple", label=r"f$(t)$", linewidth=3)
        if not out:
            plt.title(r"Графики $F_N(t)$ и $G_N(t)$" + f" при N={n}", fontsize=16)
            F_N = build_real_fourier(a0,an,bn,T)
            G_N = build_complex_fourier(cn, T)
            plt.plot(t, F_N(t) , color="green", label=r"Вещественный ряд $F_N(t)$", linewidth=2,)
            plt.plot(t, G_N(t), color="lightblue", label=r"Комплексный ряд $G_N(t)$", linewidth=1)
            
        else:
            plt.title(N2_title)
            plt.xlabel("t", fontsize=15, labelpad=10)
            plt.grid(True)
            plt.axis('equal')

            plt.figure()
            plt.title(r"Графики $F_N(t)$ и $G_N(t)$" + f" при N={n}")
            plt.grid()
            plt.plot(t, f(t), color="purple", label="Исходная функция", linewidth=3)
            F_N = build_real_fourier(a0,an,bn,T)
            G_N = build_complex_fourier(cn, T)
            plt.plot(t, F_N(t) , color="green", label=r"Вещественный ряд $F_N(t)$", linewidth=2,)
            plt.plot(t, G_N(t), color="lightblue", label=r"Комплексный ряд $G_N(t)$", linewidth=1)
        

        
        plt.legend(loc="best")
        # plt.plot(t, t*[0])
        # plt.axvline(0)
        plt.xlabel("t", fontsize=15, labelpad=10)
        plt.grid(True)
        
        # plt.axis('equal')
