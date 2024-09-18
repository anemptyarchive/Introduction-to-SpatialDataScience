# 『Pythonで学ぶ空間データサイエンス入門』のノート
# 第4章 地理的加重回帰モデルと最小二乗法
# 4.3.3 GWRモデルの重み関数の可視化

# %%

# ライブラリを読込
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# %%

### 重み関数の実装 --------------------------------------------------------------

# 重み関数を定義
def weight_functions(distance, bandwidth, fnc='bi-square', adjust_flag=False):

    # 変数を設定
    d = distance
    b = bandwidth

    # 対角要素を計算
    if fnc == 'moving window':

        # moving window型
        w = np.ones_like(d)
        w[d >= b] = 0.0

    elif fnc == 'exponential':

        # バンド幅を調整
        if adjust_flag:
            b *= 1.0 / 3.0

        # 指数型
        w = np.exp(- d / b)

    elif fnc == 'Gaussian':

        # バンド幅を調整
        if adjust_flag:
            b *= 1.0 / np.sqrt(6.0)

        # ガウス型
        w = np.exp(-0.5 * (d / b)**2)

    elif fnc == 'bi-square':

        # bi-square型
        w = (1.0 - (d / b)**2)**2
        w[d >= b] = 0.0

    elif fnc == 'tri-cube':

        # tri-cube型
        w = (1.0 - (d / b)**3)**3
        w[d >= b] = 0.0

    return w


# %%

### バンド幅と重み関数の関係 -----------------------------------------------------

# 重み関数を指定
fnc_label_lt = ['moving window', 'exponential', 'Gaussian', 'bi-square', 'tri-cube']

# フレーム数を指定
frame_num = 101

# バンド幅の範囲を指定
b_vals = np.linspace(start=0.0, stop=100.0, num=frame_num)

# 有効バンド幅を設定
adjust_flag = False

# 距離の範囲を指定
dist_line = np.linspace(start=0.0, stop=100.0, num=1001)

# グラフサイズを設定
w_min = -0.1
w_max = 1.1

# グラフオブジェクトを初期化
fig, ax = plt.subplots(figsize=(8, 6), dpi=100, facecolor='white', 
                       constrained_layout=True)

# 作図処理を定義
def update(frame_i):
    
    # 前フレームのグラフを初期化
    ax.cla()

    # バンド幅を取得
    b = b_vals[frame_i]

    # ラベル用の文字列を作成
    if adjust_flag:
        param_label = f'$b^{{*}} = {b:.1f}$'
    else:
        param_label = f'$b = {b:.1f}$'

    # 重み関数を描画
    ax.vlines(x=b, ymin=w_min, ymax=w_max, 
            color='red', linewidth=2.0, linestyle='dashed') # バンド幅
    for fnc_label in fnc_label_lt:
        # 重み関数を計算
        weight_line = weight_functions(dist_line, b, fnc_label, adjust_flag)
        ax.plot(dist_line, weight_line, 
                label=fnc_label) # 重み曲線
    ax.set_xlabel('distance ($d$)')
    ax.set_ylabel('weight ($w$)')
    ax.set_title(param_label, loc='left')
    fig.suptitle('weight function', fontsize=20)
    ax.set_ylim(ymin=w_min, ymax=w_max)
    ax.grid()
    ax.legend(loc='upper right')

# 動画を作成
ani = FuncAnimation(fig=fig, func=update, frames=frame_num, interval=100)

# 動画を書出
ani.save(
    filename='../../figure/GWR/weight_function/weight_functions_b.mp4', 
    progress_callback = lambda i, n: print(f'frame: {i} / {n}')
)


# %%


