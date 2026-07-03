
# 地理空間データの重み -----------------------------------------------------------

# book 1
# ch 2.2

# 空間重み行列
# 境界の共有
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
dir_path += 'spatial_weight_matrix_by_contiguity/' # フォルダを指定
print(dir_path)


# %%

# ライブラリの読込 --------------------------------------------------------------

# ライブラリを読込
import geopandas as gpd
from pysal.lib import weights
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
gdf_target['centers'] = gdf_target['geometry'].centroid # 重心座標

# データフレームを整形
gdf_target = gdf_target.reindex(
    columns=['city1', 'city2', 'd_code', 'geometry', 'centers']
) # (確認用)
gdf_target['city2_label'] = gdf_target['city2'].str.replace('大阪市|堺市', '', regex=True) # 地名が重なる対策用


# %%

# 地域を指定
city_name = '大阪市'

# データを抽出 
gdf_target = gdf_target[gdf_target['city1'] == city_name]
print(gdf_district)


# %%

# 隣接関係を作成
wr = weights.Rook.from_dataframe(gdf_target)

# 空間隣接行列を作成
adj_mat = wr.full()[0].astype(int)

# 空間重み行列を作成
weight_mat = adj_mat / adj_mat.sum(axis=1, keepdims=True)
np.nan_to_num(weight_mat, nan=0.0) # 列要素が全て0の場合用


# %%

# 区域数を取得
N = len(gdf_target)

# 色の調整用:(固定)
w_max = 1.0

# 隣接区域以外を非表示化
weight_mat_masked = np.ma.masked_where(adj_mat==0, weight_mat)

# グラフオブジェクトを初期化
fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True, 
                         figsize=(20, 10), width_ratios=[1, 1], dpi=100, facecolor='white')
fig.suptitle('spatial weight matrix', fontsize=20)

# 作図処理を定義
def update(n):
    
    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes]

    # 重みを格納
    gdf_target['weight'] = weight_mat[:, n]
    
    # 隣接ネットワークを作図
    ax = axes[0]
    gdf_district.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black') # 行政区界
    gdf_target.plot(ax=ax, column='weight', cmap='seismic', vmin=-w_max, vmax=w_max) # 各区域の値
    for j in range(N):
        # 隣接区域のインデックスを抽出
        adj_idx, = np.where(adj_mat[:, j] == 1)
        adj_idx = adj_idx[adj_idx > j] # 重複を除去
        for i in adj_idx:
            ax.plot([gdf_target.centers.x[j], gdf_target.centers.x[i]], 
                    [gdf_target.centers.y[j], gdf_target.centers.y[i]], 
                    color='C0', linewidth=1.0) # 各区域-隣接区域
    # 隣接区域のインデックスを抽出
    adj_idx, = np.where(adj_mat[:, n] == 1)
    for i in adj_idx:
        ax.plot([gdf_target.centers.x[n], gdf_target.centers.x[i]], 
                [gdf_target.centers.y[n], gdf_target.centers.y[i]], 
                color='orange', linewidth=2.5) # 対象区域-隣接区域
    for x, y, label in zip(gdf_target.centers.x, gdf_target.centers.y, gdf_target.city2_label):
        ax.annotate(text=label, xy=(x-0.015, y-0.005), size=9) # 区域名
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_title(f'Osaka: $N = {N}$', loc='left')
    ax.grid()
    ax.set_aspect('equal', adjustable='box')
    
    # 対象区域以外・隣接区域以外を非表示化
    target_mat_bool = np.tile(True, reps=adj_mat.shape)
    target_mat_bool[:, n] = False
    target_mat_masked = np.ma.masked_array(weight_mat_masked, target_mat_bool)
    
    # 空間重み行列を作図
    ax = axes[1]
    ax.pcolor(weight_mat, cmap='seismic', vmin=-w_max, vmax=w_max) # 全区域
    ax.pcolor(target_mat_masked, cmap='seismic', vmin=-w_max, vmax=w_max, color='gray') # 対象区域の隣接区域
    ax.vlines(x=n+0.5, ymin=0, ymax=N, color='gray', linestyle='dashed') # 対象区域
    ax.set_xticks(ticks=np.arange(N)+0.5)
    ax.set_xticklabels(labels=gdf_target.city2, size=9, rotation=90)
    ax.set_yticks(ticks=np.arange(N)+0.5)
    ax.set_yticklabels(labels=gdf_target.city2, size=9)
    ax.invert_yaxis() # 軸の反転
    ax.set_xlabel('city ( $j$ )')
    ax.set_ylabel('city ( $i$ )')
    ax.set_title(f'{gdf_target.city2[n]}: $\\sum_{{i=1}}^N \\tilde{{w}}_{{ij}} = {np.sum(adj_mat[:, n])}$', loc='left') # 対象区域の隣接数
    plt.grid()
    ax.set_aspect('equal', adjustable='box')

# 動画を作成
ani = FuncAnimation(fig=fig, func=update, frames=N, interval=1000)

# 動画を書出
ani.save(
    filename=dir_path+'weight_mat_i.mp4', 
    progress_callback = lambda i, n: print(f'frame: {i} / {n}')
)


# %%

# 隣接区域数を格納
gdf_target['count'] = np.sum(adj_mat, axis=1)

# 区域数を取得
N = len(gdf_target)

# 色の調整用
cnt_min, cnt_max = 0.0, gdf_target['count'].max()

# 隣接区域以外を非表示化
weight_mat_masked = np.ma.masked_where(adj_mat==0, weight_mat)

# グラフオブジェクトを初期化
fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, 
                       figsize=(10, 10), dpi=100, facecolor='white')
fig.suptitle('spatial weight matrix', fontsize=20)
gdf_target.plot(ax=ax, column='count', cmap='jet', vmin=cnt_min, vmax=cnt_max, 
                legend=True, legend_kwds={'label': 'count'}) # カラーバー表示用のダミー

# 作図処理を定義
def update(n):
    
    # 前フレームのグラフを初期化
    ax.cla()
    
    # 隣接ネットワークを作図
    gdf_district.boundary.plot(ax=ax, linewidth=0.5, edgecolor='white') # 行政区界
    gdf_target.plot(ax=ax, column='count', cmap='jet', vmin=cnt_min, vmax=cnt_max) # 隣接数
    for i in np.where(adj_mat[:, n] == 1)[0]:
        ax.plot([gdf_target.centers.x[n], gdf_target.centers.x[i]], 
                [gdf_target.centers.y[n], gdf_target.centers.y[i]], 
                color='white', linewidth=6.0) # 白抜き
    for j in range(N):
        # 隣接区域のインデックスを抽出
        adj_idx, = np.where(adj_mat[:, j] == 1)
        adj_idx = adj_idx[adj_idx > j] # 重複を除去
        for i in adj_idx:
            ax.plot([gdf_target.centers.x[j], gdf_target.centers.x[i]], 
                    [gdf_target.centers.y[j], gdf_target.centers.y[i]], 
                    color='C0', linewidth=1.0) # 各区域-隣接区域
    for i in np.where(adj_mat[:, n] == 1)[0]:
        ax.plot([gdf_target.centers.x[n], gdf_target.centers.x[i]], 
                [gdf_target.centers.y[n], gdf_target.centers.y[i]], 
                color=cm.seismic(weight_mat[i, n]*0.5+0.5), linewidth=2.5) # 対象区域-隣接区域
    for x, y, label in zip(gdf_target.centers.x, gdf_target.centers.y, gdf_target.city2_label):
        ax.annotate(text=label, xy=(x-0.015, y-0.005), size=9) # 区域名
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_title(f'Osaka: $N = {N}$', loc='left')
    ax.grid()
    ax.set_aspect('equal', adjustable='box')

# 動画を作成
ani = FuncAnimation(fig=fig, func=update, frames=N, interval=1000)

# 動画を書出
ani.save(
    filename=dir_path+'weight_mat_i_net.mp4', 
    progress_callback = lambda i, n: print(f'frame: {i} / {n}')
)


# %%
