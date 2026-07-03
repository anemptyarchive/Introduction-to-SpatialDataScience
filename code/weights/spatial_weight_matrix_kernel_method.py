
# 地理空間データの重み -----------------------------------------------------------

# book 2
# ch 6.2.4

# 空間重み行列
# カーネル法
# 隣接関係の可視化


# %%

# ディレクトリの設定 -------------------------------------------------------------

# ライブラリを読込
from pathlib import Path

# ワークスペースを取得
PROJECT_DIR = Path.cwd()
print(PROJECT_DIR)

# 書き出し先を設定
dir_path  = PROJECT_DIR.as_posix()
dir_path += '/figure/weights/' # パスを指定
dir_path += 'spatial_weight_matrix_by_kernel/' # フォルダを指定
print(dir_path)


# %%

# ライブラリの読込 --------------------------------------------------------------

# ライブラリを読込
import geopandas as gpd
import pandas as pd
from pysal.lib import weights
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import japanize_matplotlib
from matplotlib.animation import FuncAnimation


# %%

# データの読込 ------------------------------------------------------------------

# ファイルパスを指定
DISTRICT_PATH = 'data/nlftp/N03-20260101_27_GML/N03-20260101_27.shp' # ポリゴンデータ:大阪府(2026年版)

# データを読込
gdf_district = gpd.read_file(DISTRICT_PATH, encoding='UTF-8') # (2026年版の場合)

# 行政区域データを取得
gdf_district = gdf_district[['N03_004', 'N03_005', 'N03_007', 'geometry']]
gdf_district.columns = ['city1', 'city2', 'd_code', 'geometry']

# データを整形
dst_proj     = 6668
gdf_district = gdf_district.to_crs(epsg=dst_proj) # 空間座標系を再設定

# データを統合
gdf_target = gdf_district.dissolve(by=['d_code'], as_index=False) # 飛び地を統合
gdf_target['centroids'] = gdf_target['geometry'].centroid # 重心座標

# データフレームを整形
gdf_target = gdf_target.reindex(
    columns=['city1', 'city2', 'd_code', 'geometry', 'centroids']
) # (確認用)

# %%

# 地域を指定
city_name = '大阪市'

# データを抽出 
gdf_target = gdf_target[gdf_target['city1'] == city_name]
print(gdf_district)


# %%

# 区域の影響 ----------------------------------------------------------

### パラメータの設定 -----

# 区域数を取得
N = len(gdf_target)

# フレーム数を設定
frame_num = N


# バンド幅の変換係数を作成
km_per_degree = 111.32
degree_per_km = 1.0/km_per_degree

# バンド幅(km)を指定
h_km  = 5.0
h_deg = degree_per_km * h_km

# 空間重み行列を作成
weight_obj = weights.distance.Kernel.from_dataframe(
    df=gdf_target, geom_col='geometry', 
    bandwidth=h_deg, function='gaussian'
)
weight_mat, _ = weight_obj.full()


# %%

### 作図 -----

# カラーマップを作成
cmap = LinearSegmentedColormap.from_list(
    name='white_red',
    colors=['white', 'red']
)

# 軸の範囲を設定
margin_ratio = 0.05
lon_min, lat_min, lon_max, lat_max = gdf_target.total_bounds
lon_min -= (lon_max - lon_min) * margin_ratio
lon_max += (lon_max - lon_min) * margin_ratio
lat_min -= (lat_max - lat_min) * margin_ratio
lat_max += (lat_max - lat_min) * margin_ratio
u = 5.0
d_min = 0.0
d_max = max(
    [km_per_degree * gdf_target['centroids'].distance(gdf_target.iloc[i]['centroids']).max() for i in range(N)]
)
d_max = np.ceil(d_max /u)*u  # u単位で切り上げ
w_min, w_max = 0.0, 1.0 # 最小値・最大値
f_z_max = 0.5


# グラフオブジェクトを初期化
fig, axes = plt.subplots(
    nrows=2, ncols=2, 
    figsize=(16, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
ax2x = axes[1, 0].twiny()
ax2y = axes[1, 1].twinx()
fig.suptitle('spatial weight matrix: kernel method', fontsize=20)

# 装飾用のダミーを作成
ax = axes[0, 1]
dummy_pc = ax.pcolormesh(
    np.zeros(shape=(N, N)), 
    cmap=cmap, vmin=w_min, vmax=w_max, 
    shading='auto'
) # カラーバーの表示用
fig.colorbar(
    mappable=dummy_pc, ax=ax, shrink=1.0, 
    label='$w$'
) # 重み軸

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(frame_i):

    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes.flatten()]
    ax2x.cla()
    ax2y.cla()

    ### パラメータの設定 -----

    # 区域を設定
    n = frame_i

    ### コロプレス図の作図 -----

    # 重みを格納
    gdf_target['weight'] = weight_mat[n]

    # 隣接数を取得
    k = weight_obj.cardinalities[n]

    # 重心座標を取得
    O_x, O_y = gdf_target.iloc[n]['centroids'].coords[0] # 経度, 緯度

    # バンド幅の座標を計算
    t_vec = np.linspace(start=0.0, stop=2.0*np.pi, num=361) # ラジアン
    x_vec = O_x + h_deg * np.cos(t_vec) # 経度
    y_vec = O_y + h_deg * np.sin(t_vec) # 緯度

    # ラベルを作成
    param_lbl = f'$N = {N}, i = {n+1}, k = {k}, h = {h_km:.1f}\ (km)$'

    # コロプレス図を描画
    ax = axes[0, 0]
    gdf_target.boundary.plot(
        ax=ax, 
        edgecolor='black', linewidth=0.5
    ) # 行政区界
    gdf_target.plot(
        ax=ax, column='weight', 
        cmap=cmap, vmin=w_min, vmax=w_max
    ) # 重み
    gdf_target['centroids'].plot(
        ax=ax, 
        color='black', markersize=50, 
        label='centroids\nrepresentative point'
    ) # 重心座標
    for i in range(N):
        adj_idx, = np.where(weight_mat[i] > 0.0) # 隣接区域のインデックス
        if i != n:
            adj_idx = adj_idx[adj_idx > i] # 重複を除去
        for j in adj_idx:
            Q_x, Q_y = gdf_target.loc[i, 'centroids'].coords[0] # 対象区域の座標
            P_x, P_y = gdf_target.loc[j, 'centroids'].coords[0] # 隣接区域の座標
            ax.plot(
                [Q_x, P_x], 
                [Q_y, P_y], 
                color='C0', linewidth=3.0 if i == n else 1.0
            ) # 対象区域 - 隣接区域
    ax.plot(
        x_vec, y_vec, 
        color='black', linewidth=1.5, linestyle='-.'
    ) # バンド幅
    for x, y, area_lbl in zip(gdf_target['centroids'].x, gdf_target['centroids'].y, gdf_target['city2']):
        ax.text(
            x=x, y=y, 
            s=area_lbl, ha='right', va='bottom', 
            size=10
        ) # 区域名
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_title(param_lbl, loc='left')
    ax.legend(loc='upper left')
    ax.grid()
    ax.set_xlim(xmin=lon_min, xmax=lon_max)
    ax.set_ylim(ymin=lat_min, ymax=lat_max)
    ax.set_aspect(aspect='equal', adjustable='box')

    ### ヒートマップの作図 -----

    # 枠線の表示位置を設定
    target_bool_mat    = np.tile(True, reps=weight_mat.shape)
    target_bool_mat[n] = False
    target_masked_mat  = np.ma.masked_array(weight_mat, target_bool_mat) # 対象区域 - 全区域
    adj_idx, = np.where(weight_mat[i] > 0.0) # 隣接区域のインデックス
    adj_bool_mat             = np.tile(True, reps=weight_mat.shape)
    adj_bool_mat[n, adj_idx] = False
    adj_masked_mat           = np.ma.masked_array(weight_mat, adj_bool_mat) # 対象区域 - 隣接区域

    # ヒートマップを描画
    ax = axes[0, 1]
    ax.pcolormesh(
        weight_mat, 
        cmap=cmap, vmin=w_min, vmax=w_max, 
        shading='auto'
    ) # 全区域 - 全区域
    ax.pcolor(
        target_masked_mat, 
        facecolor='none', edgecolor='C0', linewidth=1.0, linestyle='dotted'
    ) # 対象区域 - 全区域
    ax.pcolor(
        adj_masked_mat, 
        facecolor='none', edgecolor='C0', linewidth=1.0, linestyle='solid'
    ) # 対象区域 - 隣接区域
    for j in range(N):
        ax.text(
            x=j+0.5, y=n+0.5, 
            s=f'{weight_mat[n, j]:.2f}', ha='center', va='center', 
            size=6
        ) # 重み
    ax.set_xticks(ticks=np.arange(N)+0.5)
    ax.set_xticklabels(labels=gdf_target['city2'], size=10, rotation=90) # 区域名
    ax.set_yticks(ticks=np.arange(N)+0.5)
    ax.set_yticklabels(labels=gdf_target['city2'], size=10) # 区域名
    ax.set_xlabel('$j$')
    ax.set_ylabel('$i$')
    ax.grid()
    ax.invert_yaxis() # (行番号との対応用)
    ax.set_aspect(aspect='equal', adjustable='box')

    ### カーネル関数の作図：距離 -----

    # 距離を計算
    gdf_target['distance'] = gdf_target['centroids'].distance(gdf_target.iloc[n]['centroids'])

    # 距離を取得
    d_vals   = km_per_degree * gdf_target['distance'].to_numpy()
    f_d_vals = np.exp(-0.5 * (d_vals/h_km)**2) / np.sqrt(2.0*np.pi)

    # カーネル関数を計算
    d_vec   = np.linspace(start=d_min, stop=d_max, num=1001)
    f_d_vec = np.exp(-0.5 * (d_vec/h_km)**2) / np.sqrt(2.0*np.pi)

    # 関数曲線を描画
    ax = axes[1, 0]
    ax.vlines(
        x=d_vals, ymin=w_min, ymax=w_max, 
        colors='black', linewidths=1.0, linestyles=':', 
        zorder=10
    ) # 区域
    ax.scatter(
        x=d_vals, y=f_d_vals, 
        c=cmap(f_d_vals), s=50, 
        zorder=11, clip_on=False
    ) # 重み
    ax.scatter(
        x=d_vals, y=np.zeros(N), 
        c='black', s=50, 
        zorder=12, clip_on=False
    ) # 距離
    ax.plot(
        d_vec, f_d_vec, 
        color='black', linewidth=1.5, 
        label='gaussian function', 
        zorder=20
    ) # 元の関数
    ax.axvline(
        x=h_km, 
        color='black', linewidth=1.5, linestyle='-.', 
        label='bandwidth', 
        zorder=30
    ) # バンド幅
    ax2x.set_xticks(ticks=d_vals) # 距離
    ax2x.set_xticklabels(
        labels=gdf_target['city2'].to_list(), 
        size=10, rotation=60, ha='left'
    ) # 区域名
    ax.set_xlim(xmin=d_min, xmax=d_max)
    ax2x.set_xlim(xmin=d_min, xmax=d_max)
    ax.set_ylim(ymin=w_min, ymax=f_z_max)
    ax.set_xlabel('$d$')
    fnc_lbl = '$w = f(d) = \\frac{1}{\\sqrt{2 \\pi}} \\exp(-\\frac{1}{2} \\frac{d^2}{h^2})$'
    ax.set_ylabel(fnc_lbl)
    ax.legend(loc='upper right')
    ax.grid()

    ### カーネル関数の作図：標準化距離 -----

    # 軸の範囲を設定
    z_min = d_min / h_km
    z_max = d_max / h_km

    # 距離を取得
    z_vals   = km_per_degree * gdf_target['distance'].to_numpy() / h_km
    f_z_vals = np.exp(-0.5 * z_vals**2) / np.sqrt(2.0*np.pi)
    w_vals   = f_z_vals.copy()
    w_vals[z_vals > 1.0] = 0.0

    # カーネル関数を計算
    z_vec   = np.linspace(start=z_min, stop=z_max, num=1001)
    f_z_vec = np.exp(-0.5 * z_vec**2) / np.sqrt(2.0*np.pi)
    w_vec   = f_z_vec.copy()
    w_vec[z_vec > 1.0] = 0.0

    # 関数曲線を描画
    ax = axes[1, 1]
    ax.vlines(
        x=z_vals, ymin=w_min, ymax=w_vals, 
        colors='black', linewidths=1.0, linestyles=':', 
        zorder=10
    ) # 区域
    ax.hlines(
        y=w_vals, xmin=z_vals, xmax=z_max, 
        colors='black', linewidths=1.0, linestyles=':', 
        zorder=10
    ) # 区域
    ax.scatter(
        x=z_vals, y=w_vals, 
        c=cmap(f_z_vals), s=50, 
        zorder=11, clip_on=False
    ) # 重み
    ax.scatter(
        x=z_vals, y=np.zeros(N), 
        c='black', s=50, 
        zorder=12, clip_on=False
    ) # 標準化距離
    ax.plot(
        z_vec, f_z_vec, 
        color='black', linewidth=1.5, linestyle=':', 
        zorder=20
    ) # 元の関数
    ax.plot(
        z_vec, w_vec, 
        color='black', linewidth=1.5, 
        label='kernel function', 
        zorder=20
    ) # カーネル関数
    ax.axvline(
        x=1.0, 
        color='black', linewidth=1.5, linestyle='-.', 
        label='bandwidth', 
        zorder=30
    ) # バンド幅
    ax2y.set_yticks(ticks=w_vals) # 重み
    ax2y.set_yticklabels(
        labels=gdf_target['city2'].to_list(), 
        size=10, rotation=30, va='bottom'
    ) # 区域名
    ax.set_xlim(xmin=z_min, xmax=z_max)
    ax.set_ylim(ymin=w_min, ymax=f_z_max)
    ax2y.set_ylim(ymin=w_min, ymax=f_z_max)
    ax.set_xlabel('$z = \\frac{d}{h}$')
    fnc_lbl = '$w = f(z) = \\frac{1}{\\sqrt{2 \\pi}} \\exp(-\\frac{1}{2} z^2)\ (z < 1)$'
    ax.set_ylabel(fnc_lbl)
    ax.legend(loc='upper right')
    ax.grid()

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=frame_num, interval=1000
)

# 動画を書出
anim.save(
    filename=dir_path+'weight_mat_i.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%

# バンド幅の影響：1区域 ----------------------------------------------------------

### パラメータの設定 -----

# フレーム数を指定
frame_num = 150

# バンド幅(km)を指定
h_km_vals = np.linspace(start=0.0, stop=15.0, num=frame_num+1)[1:] # 0を除外
print(h_km_vals[:5])

# バンド幅の変換係数を作成
km_per_degree = 111.32
degree_per_km = 1.0/km_per_degree


# 区域数を取得
N = len(gdf_target)

# 区域を指定
area_idx = 22

# 距離を計算
gdf_target['distance'] = gdf_target['centroids'].distance(gdf_target.iloc[area_idx]['centroids'])


# %%

### 作図 -----

# カラーマップを作成
cmap = LinearSegmentedColormap.from_list(
    name='white_red',
    colors=['white', 'red']
)

# 軸の範囲を設定
margin_ratio = 0.05
lon_min, lat_min, lon_max, lat_max = gdf_target.total_bounds
lon_min -= (lon_max - lon_min) * margin_ratio
lon_max += (lon_max - lon_min) * margin_ratio
lat_min -= (lat_max - lat_min) * margin_ratio
lat_max += (lat_max - lat_min) * margin_ratio
u = 5.0
d_min = 0.0
d_max = gdf_target['distance'].max() * km_per_degree
d_max = np.ceil(d_max /u)*u  # u単位で切り上げ
w_min, w_max = 0.0, 1.0 # 最小値・最大値
f_z_max = 0.5


# グラフオブジェクトを初期化
fig, axes = plt.subplots(
    nrows=2, ncols=2, 
    figsize=(16, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
ax2x = axes[1, 0].twiny()
ax2y = axes[1, 1].twinx()
fig.suptitle('spatial weight matrix: kernel method', fontsize=20)

# 装飾用のダミーを作成
ax = axes[0, 1]
dummy_pc = ax.pcolormesh(
    np.zeros(shape=(N, N)), 
    cmap=cmap, vmin=w_min, vmax=w_max, 
    shading='auto'
) # カラーバーの表示用
fig.colorbar(
    mappable=dummy_pc, ax=ax, shrink=1.0, 
    label='$w$'
) # 重み軸

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(frame_i):

    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes.flatten()]
    ax2x.cla()
    ax2y.cla()

    ### パラメータの設定 -----

    # バンド幅を取得
    h_km  = h_km_vals[frame_i]   # 度単位の距離
    h_deg = degree_per_km * h_km # キロメートル単位の距離

    # 空間重み行列を作成
    weight_obj = weights.distance.Kernel.from_dataframe(
        df=gdf_target, geom_col='geometry', 
        bandwidth=h_deg, function='gaussian'
    )
    weight_mat, _ = weight_obj.full()

    ### コロプレス図の作図 -----

    # 重みを格納
    gdf_target['weight'] = weight_mat[area_idx]

    # 隣接数を取得
    k = weight_obj.cardinalities[area_idx]

    # 重心座標を取得
    O_x, O_y = gdf_target.iloc[area_idx]['centroids'].coords[0] # 経度, 緯度

    # バンド幅の座標を計算
    t_vec = np.linspace(start=0.0, stop=2.0*np.pi, num=361) # ラジアン
    x_vec = O_x + h_deg * np.cos(t_vec) # 経度
    y_vec = O_y + h_deg * np.sin(t_vec) # 緯度

    # ラベルを作成
    param_lbl = f'$N = {N}, i = {area_idx+1}, k = {k}, h = {h_km:.1f}\ (km)$'

    # コロプレス図を描画
    ax = axes[0, 0]
    gdf_target.boundary.plot(
        ax=ax, 
        edgecolor='black', linewidth=0.5
    ) # 行政区界
    gdf_target.plot(
        ax=ax, column='weight', 
        cmap=cmap, vmin=w_min, vmax=w_max
    ) # 重み
    gdf_target['centroids'].plot(
        ax=ax, 
        color='black', markersize=50, 
        label='centroids\nrepresentative point'
    ) # 重心座標
    adj_idx, = np.where(weight_mat[area_idx] > 0.0) # 隣接区域のインデックス
    for j in adj_idx:
        P_x, P_y = gdf_target.loc[j, 'centroids'].coords[0] # 隣接区域の座標
        ax.plot(
            [O_x, P_x], 
            [O_y, P_y], 
            color='C0', linewidth=1.5
        ) # 対象区域 - 隣接区域
    ax.plot(
        x_vec, y_vec, 
        color='black', linewidth=1.5, linestyle='-.'
    ) # バンド幅
    for x, y, area_lbl in zip(gdf_target['centroids'].x, gdf_target['centroids'].y, gdf_target['city2']):
        ax.text(
            x=x, y=y, 
            s=area_lbl, ha='right', va='bottom', 
            size=10
        ) # 区域名
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_title(param_lbl, loc='left')
    ax.legend(loc='upper left')
    ax.grid()
    ax.set_xlim(xmin=lon_min, xmax=lon_max)
    ax.set_ylim(ymin=lat_min, ymax=lat_max)
    ax.set_aspect(aspect='equal', adjustable='box')

    ### ヒートマップの作図 -----

    # 枠線の表示位置を設定
    target_bool_mat = np.tile(True, reps=weight_mat.shape)
    target_bool_mat[area_idx] = False
    target_masked_mat = np.ma.masked_array(weight_mat, target_bool_mat) # 対象区域 - 全区域
    adj_bool_mat = np.tile(True, reps=weight_mat.shape)
    adj_bool_mat[area_idx, adj_idx] = False
    adj_masked_mat = np.ma.masked_array(weight_mat, adj_bool_mat) # 対象区域 - 隣接区域

    # ヒートマップを描画
    ax = axes[0, 1]
    ax.pcolormesh(
        weight_mat, 
        cmap=cmap, vmin=w_min, vmax=w_max, 
        shading='auto'
    ) # 全区域 - 全区域
    ax.pcolor(
        target_masked_mat, 
        facecolor='none', edgecolor='C0', linewidth=1.0, linestyle='dotted'
    ) # 対象区域 - 全区域
    ax.pcolor(
        adj_masked_mat, 
        facecolor='none', edgecolor='C0', linewidth=1.0, linestyle='solid'
    ) # 対象区域 - 隣接区域
    for j in range(N):
        ax.text(
            x=j+0.5, y=area_idx+0.5, 
            s=f'{weight_mat[area_idx, j]:.2f}', ha='center', va='center', 
            size=6
        ) # 重み
    ax.set_xticks(ticks=np.arange(N)+0.5)
    ax.set_xticklabels(labels=gdf_target['city2'], size=10, rotation=90) # 区域名
    ax.set_yticks(ticks=np.arange(N)+0.5)
    ax.set_yticklabels(labels=gdf_target['city2'], size=10) # 区域名
    ax.set_xlabel('$j$')
    ax.set_ylabel('$i$')
    ax.grid()
    ax.invert_yaxis() # (行番号との対応用)
    ax.set_aspect(aspect='equal', adjustable='box')

    ### カーネル関数の作図：距離 -----

    # 距離を取得
    d_vals   = km_per_degree * gdf_target['distance'].to_numpy()
    f_d_vals = np.exp(-0.5 * (d_vals/h_km)**2) / np.sqrt(2.0*np.pi)

    # カーネル関数を計算
    d_vec   = np.linspace(start=d_min, stop=d_max, num=1001)
    f_d_vec = np.exp(-0.5 * (d_vec/h_km)**2) / np.sqrt(2.0*np.pi)

    # 関数曲線を描画
    ax = axes[1, 0]
    ax.vlines(
        x=d_vals, ymin=w_min, ymax=w_max, 
        colors='black', linewidths=1.0, linestyles=':', 
        zorder=10
    ) # 区域
    ax.scatter(
        x=d_vals, y=f_d_vals, 
        c=cmap(f_d_vals), s=50, 
        zorder=11, clip_on=False
    ) # 重み
    ax.scatter(
        x=d_vals, y=np.zeros(N), 
        c='black', s=50, 
        zorder=12, clip_on=False
    ) # 距離
    ax.plot(
        d_vec, f_d_vec, 
        color='black', linewidth=1.5, 
        label='gaussian function', 
        zorder=20
    ) # 元の関数
    ax.axvline(
        x=h_km, 
        color='black', linewidth=1.5, linestyle='-.', 
        label='bandwidth', 
        zorder=30
    ) # バンド幅
    ax2x.set_xticks(ticks=d_vals) # 距離
    ax2x.set_xticklabels(
        labels=gdf_target['city2'].to_list(), 
        size=10, rotation=60, ha='left'
    ) # 区域名
    ax.set_xlim(xmin=d_min, xmax=d_max)
    ax2x.set_xlim(xmin=d_min, xmax=d_max)
    ax.set_ylim(ymin=w_min, ymax=f_z_max)
    ax.set_xlabel('$d$')
    fnc_lbl = '$w = f(d) = \\frac{1}{\\sqrt{2 \\pi}} \\exp(-\\frac{1}{2} \\frac{d^2}{h^2})$'
    ax.set_ylabel(fnc_lbl)
    ax.legend(loc='upper right')
    ax.grid()

    ### カーネル関数の作図：標準化距離 -----

    # 軸の範囲を設定
    z_min = d_min / h_km
    z_max = d_max / h_km

    # 距離を取得
    z_vals   = km_per_degree * gdf_target['distance'].to_numpy() / h_km
    f_z_vals = np.exp(-0.5 * z_vals**2) / np.sqrt(2.0*np.pi)
    w_vals   = f_z_vals.copy()
    w_vals[z_vals > 1.0] = 0.0

    # カーネル関数を計算
    z_vec   = np.linspace(start=z_min, stop=z_max, num=1001)
    f_z_vec = np.exp(-0.5 * z_vec**2) / np.sqrt(2.0*np.pi)
    w_vec   = f_z_vec.copy()
    w_vec[z_vec > 1.0] = 0.0

    # 関数曲線を描画
    ax = axes[1, 1]
    ax.vlines(
        x=z_vals, ymin=w_min, ymax=w_vals, 
        colors='black', linewidths=1.0, linestyles=':', 
        zorder=10
    ) # 区域
    ax.hlines(
        y=w_vals, xmin=z_vals, xmax=z_max, 
        colors='black', linewidths=1.0, linestyles=':', 
        zorder=10
    ) # 区域
    ax.scatter(
        x=z_vals, y=w_vals, 
        c=cmap(f_z_vals), s=50, 
        zorder=11, clip_on=False
    ) # 重み
    ax.scatter(
        x=z_vals, y=np.zeros(N), 
        c='black', s=50, 
        zorder=12, clip_on=False
    ) # 標準化距離
    ax.plot(
        z_vec, f_z_vec, 
        color='black', linewidth=1.5, linestyle=':', 
        zorder=20
    ) # 元の関数
    ax.plot(
        z_vec, w_vec, 
        color='black', linewidth=1.5, 
        label='kernel function', 
        zorder=20
    ) # カーネル関数
    ax.axvline(
        x=1.0, 
        color='black', linewidth=1.5, linestyle='-.', 
        label='bandwidth', 
        zorder=30
    ) # バンド幅
    ax2y.set_yticks(ticks=w_vals) # 重み
    ax2y.set_yticklabels(
        labels=gdf_target['city2'].to_list(), 
        size=10, rotation=30, va='bottom'
    ) # 区域名
    ax.set_xlim(xmin=z_min, xmax=z_max)
    ax.set_ylim(ymin=w_min, ymax=f_z_max)
    ax2y.set_ylim(ymin=w_min, ymax=f_z_max)
    ax.set_xlabel('$z = \\frac{d}{h}$')
    fnc_lbl = '$w = f(z) = \\frac{1}{\\sqrt{2 \\pi}} \\exp(-\\frac{1}{2} z^2)\ (z < 1)$'
    ax.set_ylabel(fnc_lbl)
    ax.legend(loc='upper right')
    ax.grid()

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=frame_num, interval=100
)

# 動画を書出
anim.save(
    filename=dir_path+'weight_mat_h_one.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%

# バンド幅の影響：全区域 ----------------------------------------------------------

### パラメータの設定 -----

# フレーム数を指定
frame_num = 101

# バンド幅(km)を指定
h_km_vals = np.linspace(start=0.0, stop=10.0, num=frame_num)
print(h_km_vals[:5])

# バンド幅の変換係数を作成
km_per_degree = 111.32
degree_per_km = 1.0/km_per_degree


# 区域数を取得
N = len(gdf_target)


# %%

### 作図 -----

# カラーマップを作成
cmap = LinearSegmentedColormap.from_list(
    name='white_red',
    colors=['white', 'red']
)

# 軸の範囲を設定
u = 5.0
k_min, k_max = 0.0, N
k_max = np.ceil(k_max /u)*u  # u単位で切り上げ
w_min, w_max = 0.0, 1.0 # 最小値・最大値

# グラフオブジェクトを初期化
fig, axes = plt.subplots(
    nrows=1, ncols=2, 
    figsize=(16, 6), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('spatial weight matrix: kernel method: Gaussian function', fontsize=20)

# 装飾用のダミーを設定
ax = axes[0]
gdf_target['cardinality'] = 0.0
gdf_target.plot(
    ax=ax, column='cardinality', 
    cmap='viridis', alpha=0.5, vmin=k_min, vmax=k_max, 
    legend=True, legend_kwds={'label': '$k$', 'shrink': 1.0}
) # 隣接数軸
ax = axes[1]
dummy_pc = ax.pcolormesh(
    np.zeros(shape=(N, N)), 
    cmap=cmap, vmin=w_min, vmax=w_max, 
    shading='auto'
) # カラーバーの表示用
fig.colorbar(
    mappable=dummy_pc, ax=ax, 
    label='$w$'
) # 重み軸

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(frame_i):

    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes]

    ### パラメータの設定 -----

    # バンド幅を取得
    h_km  = h_km_vals[frame_i]   # 度単位の距離
    h_deg = degree_per_km * h_km # キロメートル単位の距離

    # 空間重み行列を作成
    weight_obj = weights.distance.Kernel.from_dataframe(
        df=gdf_target, geom_col='geometry', 
        bandwidth=h_deg, function='gaussian'
    )
    weight_mat, _ = weight_obj.full()

    ### ネットワーク図の作図 -----

    # 隣接数を取得
    k_vals = np.array(list(weight_obj.cardinalities.values()))
    gdf_target['cardinality'] = k_vals.copy()

    # ラベルを作成
    param_lbl = f'$N = {N}, h = {h_km:.1f}\ (km)$'

    # ネットワーク図を描画
    ax = axes[0]
    gdf_target.boundary.plot(
        ax=ax, 
        edgecolor='black', linewidth=0.5
    ) # 行政区界
    gdf_target.plot(
        ax=ax, column='cardinality', 
        cmap='viridis', alpha=0.5, vmin=k_min, vmax=k_max
    ) # 隣接数
    gdf_target['centroids'].plot(
        ax=ax, 
        color='black', markersize=50, 
        label='centroids\nrepresentative point'
    ) # 重心座標
    for i in range(N):
        adj_idx, = np.where(weight_mat[i] > 0.0) # 隣接区域のインデックス
        adj_idx  = adj_idx[adj_idx > i]          # 重複を除去
        for j in adj_idx:
            O_x, O_y = gdf_target.loc[i, 'centroids'].coords[0] # 対象区域の座標
            P_x, P_y = gdf_target.loc[j, 'centroids'].coords[0] # 隣接区域の座標
            ax.plot(
                [O_x, P_x], 
                [O_y, P_y], 
                color='C0', linewidth=1.0
            ) # 対象区域 - 隣接区域
    for x, y, area_lbl in zip(gdf_target['centroids'].x, gdf_target['centroids'].y, gdf_target['city2']):
        ax.text(
            x=x, y=y, 
            s=area_lbl, ha='right', va='bottom', 
            size=10
        ) # 区域名
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_title(param_lbl, loc='left')
    ax.legend(loc='upper left')
    ax.grid()
    ax.set_aspect(aspect='equal', adjustable='box')

    ### ヒートマップの作図 -----

    # ヒートマップを描画
    ax = axes[1]
    ax.pcolormesh(
        weight_mat, 
        cmap=cmap, vmin=w_min, vmax=w_max, 
        shading='auto'
    ) # 全区域 - 全区域
    ax.set_xticks(ticks=np.arange(N)+0.5)
    ax.set_xticklabels(labels=gdf_target['city2'], size=10, rotation=90) # 区域名
    ax.set_yticks(ticks=np.arange(N)+0.5)
    ax.set_yticklabels(
        labels=[f'{lbl} ($k = {k}$)' for lbl, k in zip(gdf_target['city2'], k_vals)], 
        size=10
    ) # 区域名, 隣接数
    ax.set_xlabel('$j$')
    ax.set_ylabel('$i$')
    ax.grid()
    ax.invert_yaxis() # (行番号との対応用)
    ax.set_aspect(aspect='equal', adjustable='box')

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=frame_num, interval=100
)

# 動画を書出
anim.save(
    filename=dir_path+'weight_mat_h_all.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%


