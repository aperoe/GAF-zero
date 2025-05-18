import streamlit as st
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

# 设置标题
st.title("GAF 零点模拟器")

# 选择 φ_n(z) 类型
phi_type = st.selectbox("选择 φₙ(z) 的形式：", [
    "φₙ(z) = zⁿ / √(n!)  (平面 GAF)",
    "φₙ(z) = zⁿ  (普通幂级数)",
])

# 设置截断项数 N
N = st.slider("截断项数 N", min_value=10, max_value=100, value=50)

# 设置区域
st.markdown("### 设置展示区域")
x_min = st.number_input("x 最小值", value=-5.0)
x_max = st.number_input("x 最大值", value=5.0)
y_min = st.number_input("y 最小值", value=-5.0)
y_max = st.number_input("y 最大值", value=5.0)

# 设置采样密度
density = st.slider("采样密度（每单位的起始点数）", 2, 10, 4)

# 点击按钮开始模拟
if st.button("生成零点图"):

    st.write("正在计算，请稍候...")

    # 生成 phi_n(z)
    if "√" in phi_type:
        phi_funcs = [lambda z, n=n: z**n / np.sqrt(np.math.factorial(n)) for n in range(N+1)]
    else:
        phi_funcs = [lambda z, n=n: z**n for n in range(N+1)]

    # 随机系数
    X = (np.random.randn(N+1) + 1j*np.random.randn(N+1)) / np.sqrt(2)

    # 定义函数
    def f(z):
        s = 0
        for n in range(N+1):
            s += X[n] * phi_funcs[n](z)
        return s

    # 网格采样初始点
    nx = int((x_max - x_min) * density)
    ny = int((y_max - y_min) * density)
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    seeds = [x + 1j*y for x in xs for y in ys]

    # 找零点
    roots = []
    for z0 in seeds:
        try:
            sol = mp.findroot(
                [lambda u, v: mp.re(f(u + 1j*v)),
                 lambda u, v: mp.im(f(u + 1j*v))],
                (mp.re(z0), mp.im(z0)),
                tol=1e-8, maxsteps=50
            )
            z_root = complex(sol[0], sol[1])
            if x_min <= z_root.real <= x_max and y_min <= z_root.imag <= y_max:
                roots.append(z_root)
        except Exception:
            pass

    # 去重
    unique_roots = []
    eps = 1e-3
    for z in roots:
        if not any(abs(z - w) < eps for w in unique_roots):
            unique_roots.append(z)

    # 绘图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter([z.real for z in unique_roots],
               [z.imag for z in unique_roots], s=10, color='blue')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_title("GAF 零点模拟")
    ax.set_aspect('equal')
    ax.grid(True)
    st.pyplot(fig)
