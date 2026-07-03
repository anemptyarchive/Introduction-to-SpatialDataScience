
# 地理空間データの重み -----------------------------------------------------------

# book 1
# ch 2.1

# 空間隣接行列
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
dir_path += 'spatial_adjacency_matrix_by_contiguity/' # フォルダを指定
print(dir_path)


# %%

# ライブラリの読込 --------------------------------------------------------------

# ライブラリを読込
import geopandas as gpd
import pandas as pd
from pysal.lib import weights
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import japanize_matplotlib
from matplotlib.animation import FuncAnimation


# %%

# データの読込 ------------------------------------------------------------------

# ファイルパスを指定
DISTRICT_PATH = 'data/nlftp/N03-20260101_27_GML/N03-20260101_27.shp' # ポリゴンデータ:大阪府(2026年版)
#DISTRICT_PATH = 'data/nlftp/N03-20260101_18_GML/N03-20260101_18.shp' # ポリゴンデータ:福井県(2026年版)

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


# 空間隣接行列を作成
adj_obj = weights.Rook.from_dataframe(df=gdf_target)  # ルーク型
#adj_obj = weights.Queen.from_dataframe(df=gdf_target) # クイーン型
adj_mat, _ = adj_obj.full()
adj_mat = adj_mat.astype(dtype=np.int8)


# %%

### 作図 -----

# カラーマップを作成
cmap = ListedColormap(colors=['white', 'orange'])

# 軸の範囲を設定
w_min, w_max = 0.0, 1.0 # 最小値・最大値


# グラフオブジェクトを初期化
fig, axes = plt.subplots(
    nrows=1, ncols=2, 
    figsize=(15, 6), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('spatial adjacency matrix: contiguity', fontsize=20)

# 装飾用のダミーを設定
legend_handles = [
    mpatches.Patch(
        facecolor='white', #edgecolor='black', 
        label='not adjacent'
    ),
    mpatches.Patch(
        facecolor=cmap(1.0), 
        label='adjacent'
    )
]

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(frame_i):
    
    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes]

    ### パラメータの設定 -----

    # 区域を設定
    n = frame_i

    ### コロプレス図の作図 -----

    # 隣接関係を格納
    gdf_target['adjacency'] = adj_mat[n]

    # 隣接数を取得
    k = adj_obj.cardinalities[n]

    # ラベルを作成
    param_lbl = f'$N = {N}, i = {n+1}, k = {k}$'

    # コロプレス図を描画
    ax = axes[0]
    gdf_target.boundary.plot(
        ax=ax, 
        edgecolor='black', linewidth=0.5
    ) # 行政区界
    gdf_target.plot(
        ax=ax, column='adjacency', 
        cmap=cmap, vmin=w_min, vmax=w_max
    ) # 隣接関係
    for i in range(N):
        adj_idx, = np.where(adj_mat[i] == 1) # 隣接区域のインデックス
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
    # 隣接区域のインデックスを抽出
    for x, y, area_lbl in zip(gdf_target['centroids'].x, gdf_target['centroids'].y, gdf_target['city2']):
        ax.text(
            x=x, y=y, 
            s=area_lbl, ha='center', va='center', 
            size=10
        ) # 区域名
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_title(param_lbl, loc='left')
    ax.grid()
    ax.set_aspect(aspect='equal', adjustable='box')
    
    ### ヒートマップの作図 -----

    # 枠線の表示位置を設定
    target_bool_mat    = np.tile(True, reps=adj_mat.shape)
    target_bool_mat[n] = False
    target_masked_mat  = np.ma.masked_array(adj_mat, target_bool_mat) # 対象区域 - 全区域
    adj_idx, = np.where(adj_mat[i] == 1) # 隣接区域のインデックス
    adj_bool_mat             = np.tile(True, reps=adj_mat.shape)
    adj_bool_mat[n, adj_idx] = False
    adj_masked_mat           = np.ma.masked_array(adj_mat, adj_bool_mat) # 対象区域 - 隣接区域

    # ヒートマップを描画
    ax = axes[1]
    ax.pcolormesh(
        adj_mat, 
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
    ax.set_xticks(ticks=np.arange(N)+0.5)
    ax.set_xticklabels(labels=gdf_target['city2'], size=10, rotation=90) # 区域名
    ax.set_yticks(ticks=np.arange(N)+0.5)
    ax.set_yticklabels(labels=gdf_target['city2'], size=10) # 区域名
    ax.set_xlabel('$j$')
    ax.set_ylabel('$i$')
    ax.legend(
        handles=legend_handles, 
        bbox_to_anchor=(1.0, 1.0), loc='upper left'
    ) # 隣接関係
    ax.grid()
    ax.invert_yaxis() # (行番号との対応用)
    ax.set_aspect('equal', adjustable='box')

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=frame_num, interval=1000
)

# 動画を書出
anim.save(
    filename=dir_path+'adfacency_mat_i.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%

# 隣接型の影響 ----------------------------------------------------------

### パラメータの設定 -----

# 区域数を取得
N = len(gdf_target)

# フレーム数を設定
frame_num = N


# 空間隣接行列を作成
adj_obj_lt = [
    weights.Rook.from_dataframe(df=gdf_target), # ルーク型
    weights.Queen.from_dataframe(df=gdf_target) # クイーン型
]
adj_mat_lt = [
    adj_obj_lt[type_idx].full()[0].astype(dtype=np.int8) for type_idx in range(2)
]


# %%

### 作図 -----

# カラーマップを作成
cmap = ListedColormap(colors=['white', 'orange'])

# 軸の範囲を設定
w_min, w_max = 0.0, 1.0 # 最小値・最大値


# グラフオブジェクトを初期化
fig, axes = plt.subplots(
    nrows=1, ncols=2, 
    figsize=(15, 6), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('spatial adjacency matrix: contiguity', fontsize=20)

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(frame_i):
    
    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes]

    ### パラメータの設定 -----

    # 区域を設定
    n = frame_i

    ### コロプレス図の作図 -----

    for type_idx in range(2):

        # 隣接関係を格納
        gdf_target['adjacency'] = adj_mat_lt[type_idx][n]

        # 隣接数を取得
        k = adj_obj_lt[type_idx].cardinalities[n]

        # ラベルを作成
        type_str   = ['rook', 'queen'][type_idx]
        param_lbl  = f'$N = {N}, i = {n+1}$\n' if type_idx == 0 else ''
        param_lbl += type_str+': ' + f'$k = {k}$'

        # コロプレス図を描画
        ax = axes[type_idx]
        gdf_target.boundary.plot(
            ax=ax, 
            edgecolor='black', linewidth=0.5
        ) # 行政区界
        gdf_target.plot(
            ax=ax, column='adjacency', 
            cmap=cmap, vmin=w_min, vmax=w_max
        ) # 隣接関係
        gdf_target.iloc[[n]].plot(
            ax=ax, 
            color='red'
        ) # 対象区域
        for i in range(N):
            adj_idx, = np.where(adj_mat_lt[type_idx][i] == 1) # 隣接区域のインデックス
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
        # 隣接区域のインデックスを抽出
        for x, y, area_lbl in zip(gdf_target['centroids'].x, gdf_target['centroids'].y, gdf_target['city2']):
            ax.text(
                x=x, y=y, 
                s=area_lbl, ha='center', va='center', 
                size=10
            ) # 区域名
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        ax.set_title(param_lbl, loc='left')
        ax.grid()
        ax.set_aspect(aspect='equal', adjustable='box')

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=frame_num, interval=1000
)

# 動画を書出
anim.save(
    filename=dir_path+'adfacency_mat_type.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%


