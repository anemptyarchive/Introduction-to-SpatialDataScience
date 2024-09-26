# 『Pythonで学ぶ空間データサイエンス入門』のノート
# 第4章 地理的加重回帰モデルと最小二乗法
# 4.3.2 GWRモデルの重み行列の可視化

# %%

# ライブラリを読込
import numpy as np
import libpysal as ps
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# %%

### データの読込 ----------------------------------------------------------------

# 空間データを読込
geom_df = gpd.read_file(ps.examples.get_path('G_utm.shp'))

# 位置情報を取得
coord_arr = np.stack(
    [geom_df['X'].values.astype(np.float64),
     geom_df['Y'].values.astype(np.float64)], 
    axis=1
)


# %%

### 重み関数の実装 --------------------------------------------------------------

# 重み関数を定義
def weight_functions(distance, bandwidth, function='bi-square', adjust_flag=False):

    # 変数を設定
    d = distance
    b = bandwidth

    # 対角要素を計算
    if function == 'moving window':

        # moving window型
        w = np.ones_like(d)
        w[d >= b] = 0.0

    elif function == 'exponential':

        # バンド幅を調整
        if adjust_flag:
            b *= 1.0 / 3.0

        # 指数型
        w = np.exp(- d / b)

    elif function == 'Gaussian':

        # バンド幅を調整
        if adjust_flag:
            b *= 1.0 / np.sqrt(6.0)

        # ガウス型
        w = np.exp(-0.5 * (d / b)**2)

    elif function == 'bi-square':

        # bi-square型
        w = (1.0 - (d / b)**2)**2
        w[d >= b] = 0.0

    elif function == 'tri-cube':

        # tri-cube型
        w = (1.0 - (d / b)**3)**3
        w[d >= b] = 0.0

    # 重み行列を作成
    W = np.diag(w)

    return W


# %%

### 重み関数の設定 --------------------------------------------------------------

# 重み関数を指定
#fnc_label = 'moving window'
#fnc_label = 'exponential'
#fnc_label = 'Gaussian'
fnc_label = 'bi-square'
#fnc_label = 'tri-cube'

# ラベル用の文字列を設定
if fnc_label == 'moving window':

    # moving window型
    fml_label  = '$w_{ijj} = 1 ~ (d_{ij} < b)$\n'
    fml_label += '$w_{ijj} = 0 ~ (d_{ij} \\geq b)$\n'
    fml_label += '$w_{ijl} = 0 ~ (j \\neq l)$'

elif fnc_label == 'exponential':
    
    # 指数型
    fml_label  = '$w_{ijj} = \\exp(- \\frac{d_{ij}}{b})$\n'
    fml_label += '$w_{ijl} = 0 ~ (j \\neq l)$'

elif fnc_label == 'Gaussian':
    
    # ガウス型
    fml_label  = '$w_{ijj} = \\exp(- \\frac{1}{2} (\\frac{d_{ij}}{b})^2)$\n'
    fml_label += '$w_{ijl} = 0 ~ (j \\neq l)$'

elif fnc_label == 'bi-square':

    # bi-square型
    fml_label  = '$w_{ijj} = (1 - (\\frac{d_{ij}}{b})^2)^2 ~ (d_{ij} < b)$\n'
    fml_label += '$w_{ijj} = 0 ~ (d_{ij} \\geq b)$\n'
    fml_label += '$w_{ijl} = 0 ~ (j \\neq l)$'

elif fnc_label == 'tri-cube':

    # tri-cube型
    fml_label  = '$w_{ijj} = (1 - (\\frac{d_{ij}}{b})^3)^3 ~ (d_{ij} < b)$\n'
    fml_label += '$w_{ijj} = 0 ~ (d_{ij} \\geq b)$\n'
    fml_label += '$w_{ijl} = 0 ~ (j \\neq l)$'


# %%

### 各地点と重み行列の関係 -------------------------------------------------------

# 地点数を取得
N = len(geom_df)

# バンド幅を指定
b = 270000.0

# 有効バンド幅を設定
adjust_flag = False

# 地点ごとに計算
dist_lt   = []
weight_lt = []
dist_max  = 0.0
for i in range(N):
    
    # 距離を計算
    dist_vec = np.sqrt(np.sum((coord_arr - coord_arr[i])**2, axis=1))

    # 重み行列を計算
    W = weight_functions(dist_vec, b, fnc_label, adjust_flag)

    # 計算結果を格納
    dist_lt.append(dist_vec.copy())
    weight_lt.append(W.copy())
    dist_max = max(dist_max, dist_vec.max())

# グラフサイズを設定
w_min = -0.1
w_max = 1.1
u = 50000 # 切り下げ・切り上げの単位を指定
u_min = np.floor(geom_df.bounds.minx.min() /u)*u
u_max = np.ceil(geom_df.bounds.maxx.max()  /u)*u
v_min = np.floor(geom_df.bounds.miny.min() /u)*u
v_max = np.ceil(geom_df.bounds.maxy.max()  /u)*u
dist_max = np.ceil(dist_max /u)*u

# 距離の範囲を設定
dist_line = np.linspace(start=0.0, stop=dist_max, num=5001)

# 重み関数を計算
weight_line = np.diag(weight_functions(dist_line, b, fnc_label, adjust_flag))

# ラベル用の文字列を作成
if adjust_flag:
    param_label = f'$b^{{*}} = {b:.1f}$'
else:
    param_label = f'$b = {b:.1f}$'

# グラフオブジェクトを初期化
fig, axes = plt.subplots(nrows=1, ncols=3, 
                         figsize=(15, 5), dpi=100, facecolor='white', 
                         constrained_layout=True)
fig.suptitle('weight matrix: ('+fnc_label+')', fontsize=20)

# 作図処理を定義
def update(i):
    
    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes]

    # 基準地点を抽出
    target_df = geom_df.loc[i:i]
    W         = weight_lt[i]

    # 重み関数を描画
    ax = axes[0]
    ax.vlines(x=b, ymin=w_min, ymax=w_max, 
              color='red', linewidth=2.0, linestyle='dashed') # バンド幅
    ax.plot(dist_line, weight_line, 
            color='C0', label=fml_label) # 重み曲線
    ax.scatter(x=dist_lt[i], y=np.diag(W), 
               c='black', s=10) # 各地点の値
    ax.set_xlabel('distance ($d_{ij}$)')
    ax.set_ylabel('weight ($w_{ijj}$)')
    ax.set_title(param_label, loc='left')
    ax.set_ylim(ymin=w_min, ymax=w_max)
    ax.grid()
    ax.legend(loc='upper right')

    # 重みを格納
    geom_df['weight'] = np.diag(W)

    # 基準地点の位置を取得
    u_i, v_i = coord_arr[i]

    # バンド幅の範囲を計算
    t_vec = np.linspace(start=0.0, stop=2.0*np.pi, num=361) # ラジアン
    u_vec = u_i + b * np.cos(t_vec)
    v_vec = v_i + b * np.sin(t_vec)

    # 重みのコロプレス図を描画
    ax = axes[1]
    geom_df.plot(ax=ax, column='weight', vmin=0.0, vmax=1.0, 
                 edgecolor='white', linewidth=0.5) # 各地点の値
    geom_df.centroid.plot(ax=ax, c='black', markersize=5) # 各地点
    target_df.centroid.plot(ax=ax, c='red', markersize=50) # 基準地点
    ax.plot(u_vec, v_vec, 
            color='red', linewidth=2.0, linestyle='dashed') # バンド幅
    ax.set_xlabel('$u_j$')
    ax.set_ylabel('$v_j$')
    ax.set_title(f'$N = {N}, i = {i+1}, '+'(w_{i11}, \cdots, w_{iNN})$', loc='left')
    ax.grid()
    ax.set_xlim(xmin=u_min, xmax=u_max)
    ax.set_ylim(ymin=v_min, ymax=v_max)
    ax.set_aspect('equal', adjustable='box')

    # 重み行列を描画
    ax = axes[2]
    ax.pcolor(W, vmin=0.0, vmax=1.0) # 行列
    ax.invert_yaxis() # 軸の反転
    ax.set_xlabel('$l$')
    ax.set_ylabel('$j$')
    ax.set_title('$W_i = (w_{i11}, \cdots, w_{ijl}, \cdots, w_{iNN})$', loc='left')
    ax.grid()
    ax.set_aspect('equal', adjustable='box')

# 動画を作成
ani = FuncAnimation(fig=fig, func=update, frames=N, interval=500)

# 動画を書出
ani.save(
    filename='../../figure/GWR/weight_matrix/weight_'+fnc_label+'_i.mp4', 
    progress_callback = lambda i, n: print(f'frame: {i} / {n}')
)


# %%

### バンド幅と重み行列の関係 -----------------------------------------------------

# フレーム数を指定
frame_num = 101

# バンド幅の範囲を指定
b_vals = np.linspace(start=0.0, stop=500000.0, num=frame_num)

# 有効バンド幅を設定
adjust_flag = False

# 地点数を取得
N = len(geom_df)

# 地点を指定
i = 0

# 距離を計算
dist_vec = np.sqrt(np.sum((coord_arr - coord_arr[i])**2, axis=1))

# バンド幅ごとに計算
weight_lt = []
for frame_i in range(frame_num):

    # バンド幅を取得
    b = b_vals[frame_i]

    # 重み行列を計算
    W = weight_functions(dist_vec, b, fnc_label, adjust_flag)

    # 計算結果を格納
    weight_lt.append(W.copy())

# グラフサイズを設定
w_min = -0.1
w_max = 1.1
u = 50000 # 切り下げ・切り上げの単位を指定
u_min = np.floor(geom_df.bounds.minx.min() /u)*u
u_max = np.ceil(geom_df.bounds.maxx.max()  /u)*u
v_min = np.floor(geom_df.bounds.miny.min() /u)*u
v_max = np.ceil(geom_df.bounds.maxy.max()  /u)*u
dist_max = np.ceil(max(dist_vec.max(), b_vals.max()) /u)*u

# 距離の範囲を設定
dist_line = np.linspace(start=0.0, stop=dist_max, num=5001)

# 基準地点の位置を取得
u_i, v_i = coord_arr[i]

# グラフオブジェクトを初期化
fig, axes = plt.subplots(nrows=1, ncols=3, 
                         figsize=(15, 5), dpi=100, facecolor='white', 
                         constrained_layout=True)
fig.suptitle('weight matrix: ('+fnc_label+')', fontsize=20)

# 作図処理を定義
def update(frame_i):
    
    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes]

    # バンド幅を取得
    b = b_vals[frame_i]

    # 重みを取得
    W = weight_lt[frame_i]

    # 重み関数を計算
    weight_line = np.diag(weight_functions(dist_line, b, fnc_label, adjust_flag))
    
    # ラベル用の文字列を作成
    if adjust_flag:
        param_label = f'$b^{{*}} = {b:.1f}$'
    else:
        param_label = f'$b = {b:.1f}$'

    # 重み関数を描画
    ax = axes[0]
    ax.vlines(x=b, ymin=w_min, ymax=w_max, 
              color='red', linewidth=2.0, linestyle='dashed') # バンド幅
    ax.plot(dist_line, weight_line, 
            color='C0', label=fml_label) # 重み曲線
    ax.scatter(x=dist_vec, y=np.diag(W), 
               c='black', s=10) # 各地点の値
    ax.set_xlabel('distance ($d_{ij}$)')
    ax.set_ylabel('weight ($w_{ijj}$)')
    ax.set_title(param_label, loc='left')
    ax.set_ylim(ymin=w_min, ymax=w_max)
    ax.grid()
    ax.legend(loc='upper right')

    # 重みを格納
    geom_df['weight'] = np.diag(W)

    # バンド幅の範囲を計算
    t_vec = np.linspace(start=0.0, stop=2.0*np.pi, num=361) # ラジアン
    u_vec = u_i + b * np.cos(t_vec)
    v_vec = v_i + b * np.sin(t_vec)

    # 重みのコロプレス図を描画
    ax = axes[1]
    geom_df.plot(ax=ax, column='weight', vmin=0.0, vmax=1.0, 
                 edgecolor='white', linewidth=0.5) # 各地点の値
    ax.scatter(x=coord_arr[:, 0], y=coord_arr[:, 1], 
               color='black', s=5) # 各地点
    ax.scatter(x=u_i, y=v_i, color='red', s=50, 
               label='$(u_i, v_i)$') # 基準地点
    ax.plot(u_vec, v_vec, 
            color='red', linewidth=2.0, linestyle='dashed') # バンド幅
    ax.set_xlabel('$u_j$')
    ax.set_ylabel('$v_j$')
    ax.set_title(f'$N = {N}, i = {i+1}, '+'(w_{i11}, \cdots, w_{iNN})$', loc='left')
    ax.grid()
    ax.set_xlim(xmin=u_min, xmax=u_max)
    ax.set_ylim(ymin=v_min, ymax=v_max)
    ax.set_aspect('equal', adjustable='box')

    # 重み行列を描画
    ax = axes[2]
    ax.pcolor(W, vmin=0.0, vmax=1.0) # 行列
    ax.invert_yaxis() # 軸の反転
    ax.set_xlabel('$l$')
    ax.set_ylabel('$j$')
    ax.set_title('$W_i = (w_{i11}, \cdots, w_{ijl}, \cdots, w_{iNN})$', loc='left')
    ax.grid()
    ax.set_aspect('equal', adjustable='box')

# 動画を作成
ani = FuncAnimation(fig=fig, func=update, frames=frame_num, interval=100)

# 動画を書出
ani.save(
    filename='../../figure/GWR/weight_matrix/weight_'+fnc_label+'_b.mp4', 
    progress_callback = lambda i, n: print(f'frame: {i} / {n}')
)


# %%


