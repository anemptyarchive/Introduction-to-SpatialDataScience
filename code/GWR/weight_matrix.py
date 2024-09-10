# 『Pythonで学ぶ空間データサイエンス入門』のノート
# 第4章 地理的加重回帰モデルと最小二乗法
# 4.3.2-3 GWRモデルの重み行列の可視化

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

### 重み関数の設定 --------------------------------------------------------------

# 重み関数を指定
#fnc_label = 'moving window'
#fnc_label = 'Gaussian'
fnc_label = 'bi-square'

# 重み関数を設定
if fnc_label == 'moving window':

    # moving window関数を実装
    def weight_fnc(dist, b):

        # 対角要素を計算
        w_diag = np.ones_like(dist)
        w_diag[dist >= b] = 0.0

        # 重み行列を作成
        W = np.diag(w_diag)

        return W

elif fnc_label == 'Gaussian':

    # Gaussian関数を実装
    def weight_fnc(dist, b):

        # 対角要素を計算
        w_diag = np.exp(-0.5 * (dist / b)**2)

        # 重み行列を作成
        W = np.diag(w_diag)

        return W

elif fnc_label == 'bi-square':

    # bi-square関数を実装
    def weight_fnc(dist, b):

        # 対角要素を計算
        w_diag = (1.0 - (dist / b)**2)**2
        w_diag[dist >= b] = 0.0

        # 重み行列を作成
        W = np.diag(w_diag)

        return W

# ラベル用の文字列を設定
if fnc_label == 'moving window':
    
    fml_label  = '$w_{ijj} = 1 ~ (d_{ij} < b)$\n'
    fml_label += '$w_{ijj} = 0 ~ (d_{ij} \\geq b)$\n'
    fml_label += '$w_{ijl} = 0 ~ (j \\neq l)$'

elif fnc_label == 'Gaussian':
    
    fml_label  = '$w_{ijj} = \\exp(- \\frac{1}{2} (\\frac{d_{ij}}{b})^2)$\n'
    fml_label += '$w_{ijl} = 0 ~ (j \\neq l)$'

elif fnc_label == 'bi-square':

    fml_label  = '$w_{ijj} = (1 - (\\frac{d_{ij}}{b})^2)^2 ~ (d_{ij} < b)$\n'
    fml_label += '$w_{ijj} = 0 ~ (d_{ij} \\geq b)$\n'
    fml_label += '$w_{ijl} = 0 ~ (j \\neq l)$'


# %%

### 各地点と重み行列の関係 -------------------------------------------------------

# 地点数を取得
N = len(geom_df)

# バンド幅を指定
b = 270000.0

# 地点ごとに計算
dist_lt   = []
weight_lt = []
dist_max  = 0.0
for i in range(N):
    
    # 距離を計算
    dist_vec = np.sqrt(np.sum((coord_arr - coord_arr[i])**2, axis=1))

    # 重み行列を計算
    W = weight_fnc(dist_vec, b)

    # 計算結果を格納
    dist_lt.append(dist_vec.copy())
    weight_lt.append(W.copy())
    dist_max = max(dist_max, dist_vec.max())

# グラフサイズを設定
w_min = -0.1
w_max = 1.1
u = 50000 # 切り下げ・切り上げの単位を指定
lng_min = np.floor(geom_df.bounds.minx.min() /u)*u
lng_max = np.ceil(geom_df.bounds.maxx.max()  /u)*u
lat_min = np.floor(geom_df.bounds.miny.min() /u)*u
lat_max = np.ceil(geom_df.bounds.maxy.max()  /u)*u
dist_max = np.ceil(dist_max /u)*u

# 距離の範囲を設定
dist_line = np.linspace(start=0.0, stop=dist_max, num=5001)

# 重み関数を計算
weight_line = np.diag(weight_fnc(dist_line, b))

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
    ax.set_title(f'$b = {b:.1f}$', loc='left')
    ax.set_ylim(ymin=w_min, ymax=w_max)
    ax.grid()
    ax.legend()

    # 重みを格納
    geom_df['weight'] = np.diag(W)

    # 基準地点の位置を取得
    lng0 = target_df.X.values.item()
    lat0 = target_df.Y.values.item()

    # バンド幅の範囲を計算
    t_vec   = np.linspace(start=0.0, stop=2.0*np.pi, num=361) # ラジアン
    lng_vec = lng0 + b * np.cos(t_vec)
    lat_vec = lat0 + b * np.sin(t_vec)

    # 重みのコロプレス図を描画
    ax = axes[1]
    geom_df.plot(ax=ax, column='weight', vmin=0.0, vmax=1.0, 
                 edgecolor='white', linewidth=0.5) # 各地点の値
    geom_df.centroid.plot(ax=ax, c='black', markersize=5) # 各地点
    target_df.centroid.plot(ax=ax, c='red', markersize=50) # 基準地点
    ax.plot(lng_vec, lat_vec, 
            color='red', linewidth=2.0, linestyle='dashed') # バンド幅
    ax.set_xlim(xmin=lng_min, xmax=lng_max)
    ax.set_ylim(ymin=lat_min, ymax=lat_max)
    ax.set_title(f'$N = {N}, i = {i}, '+'(w_{i11}, \cdots, w_{iNN})$', loc='left')
    ax.grid()
    ax.set_aspect('equal', adjustable='box')

    # 重み行列を描画
    ax = axes[2]
    ax.pcolor(W, vmin=0.0, vmax=1.0) # 行列
    ax.invert_yaxis() # 軸の反転
    ax.set_xlabel('$l$')
    ax.set_ylabel('$j$')
    ax.set_title('$W_{i} = (w_{i11}, \cdots, w_{ijl}, \cdots, w_{iNN})$', loc='left')
    ax.grid()
    ax.set_aspect('equal', adjustable='box')

# 動画を作成
ani = FuncAnimation(fig=fig, func=update, frames=N, interval=500)

# 動画を書出
ani.save(
    filename='../../figure/ch4/weight_matrix/weight_'+fnc_label+'_i.mp4', 
    progress_callback = lambda i, n: print(f'frame: {i} / {n}')
)


# %%

### バンド幅と重み行列の関係 -----------------------------------------------------

# フレーム数を指定
frame_num = 101

# バンド幅の範囲を指定
b_vals = np.linspace(start=0.0, stop=500000.0, num=frame_num)

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
    W = weight_fnc(dist_vec, b)

    # 計算結果を格納
    weight_lt.append(W.copy())

# グラフサイズを設定
w_min = -0.1
w_max = 1.1
u = 50000 # 切り下げ・切り上げの単位を指定
lng_min = np.floor(geom_df.bounds.minx.min() /u)*u
lng_max = np.ceil(geom_df.bounds.maxx.max()  /u)*u
lat_min = np.floor(geom_df.bounds.miny.min() /u)*u
lat_max = np.ceil(geom_df.bounds.maxy.max()  /u)*u
dist_max = np.ceil(max(dist_vec.max(), b_vals.max()) /u)*u

# 距離の範囲を設定
dist_line = np.linspace(start=0.0, stop=dist_max, num=5001)

# 基準地点を抽出
target_df = geom_df.loc[i:i]

# 基準地点の位置を取得
lng0 = target_df.X.values.item()
lat0 = target_df.Y.values.item()

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
    weight_line = np.diag(weight_fnc(dist_line, b))

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
    ax.set_title(f'$b = {b:.1f}$', loc='left')
    ax.set_ylim(ymin=w_min, ymax=w_max)
    ax.grid()
    ax.legend()

    # 重みを格納
    geom_df['weight'] = np.diag(W)

    # バンド幅の範囲を計算
    t_vec   = np.linspace(start=0.0, stop=2.0*np.pi, num=361) # ラジアン
    lng_vec = lng0 + b * np.cos(t_vec)
    lat_vec = lat0 + b * np.sin(t_vec)

    # 重みのコロプレス図を描画
    ax = axes[1]
    geom_df.plot(ax=ax, column='weight', vmin=0.0, vmax=1.0, 
                 edgecolor='white', linewidth=0.5) # 各地点の値
    geom_df.centroid.plot(ax=ax, c='black', markersize=5) # 各地点
    target_df.centroid.plot(ax=ax, c='red', markersize=50) # 基準地点
    ax.plot(lng_vec, lat_vec, 
            color='red', linewidth=2.0, linestyle='dashed') # バンド幅
    ax.set_xlim(xmin=lng_min, xmax=lng_max)
    ax.set_ylim(ymin=lat_min, ymax=lat_max)
    ax.set_title(f'$N = {N}, i = {i}, '+'(w_{i11}, \cdots, w_{iNN})$', loc='left')
    ax.grid()
    ax.set_aspect('equal', adjustable='box')

    # 重み行列を描画
    ax = axes[2]
    ax.pcolor(W, vmin=0.0, vmax=1.0) # 行列
    ax.invert_yaxis() # 軸の反転
    ax.set_xlabel('$l$')
    ax.set_ylabel('$j$')
    ax.set_title('$W_{i} = (w_{i11}, \cdots, w_{ijl}, \cdots, w_{iNN})$', loc='left')
    ax.grid()
    ax.set_aspect('equal', adjustable='box')

# 動画を作成
ani = FuncAnimation(fig=fig, func=update, frames=frame_num, interval=100)

# 動画を書出
ani.save(
    filename='../../figure/ch4/weight_matrix/weight_'+fnc_label+'_b.mp4', 
    progress_callback = lambda i, n: print(f'frame: {i} / {n}')
)


# %%



