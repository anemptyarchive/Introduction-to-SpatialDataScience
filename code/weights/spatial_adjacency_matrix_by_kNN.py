
# 地理空間データの重み -----------------------------------------------------------

# book 2
# ch 6.2.4

# 空間重み行列
# k近傍法
# 近傍関係の可視化


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
dir_path += 'spatial_adjacency_matrix_by_kNN/' # フォルダを指定
print(dir_path)


# %%

# ライブラリの読込 --------------------------------------------------------------

# ライブラリを読込
import geopandas as gpd
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

# 共通の設定 --------------------------------------------------------------------

# バンド幅の変換係数を作成
km_per_degree = 111.32
degree_per_km = 1.0/km_per_degree


# カラーマップを作成
cmap = ListedColormap(colors=['white', 'orange'])


# %%

# 区域の影響 --------------------------------------------------------------------

### パラメータの設定 -----

# 区域数を取得
N = len(gdf_target)

# フレーム数を設定
frame_num = N


# 近傍数を指定
k = 4

# 空間隣接行列を作成
adj_obj = weights.distance.KNN.from_dataframe(
    df=gdf_target, geom_col='geometry', k=k
)
adj_mat, _ = adj_obj.full()
adj_mat = adj_mat.astype(dtype=np.int8)


# %%

### 作図 -----

# 軸の範囲を設定
margin_ratio = 0.05
lon_min, lat_min, lon_max, lat_max = gdf_target.total_bounds
lon_min -= (lon_max - lon_min) * margin_ratio
lon_max += (lon_max - lon_min) * margin_ratio
lat_min -= (lat_max - lat_min) * margin_ratio
lat_max += (lat_max - lat_min) * margin_ratio
w_min, w_max = 0.0, 1.0 # 最小値・最大値


# グラフオブジェクトを初期化
fig, axes = plt.subplots(
    nrows=1, ncols=2, 
    figsize=(15, 6), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('spatial adjacency matrix: k-NN', fontsize=20)

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

    # 距離を計算
    gdf_target['distance'] = gdf_target['centroids'].distance(gdf_target.loc[n, 'centroids'])

    # バンド幅を取得
    adj_idx, = np.where(adj_mat[n] == 1) # 隣接区域のインデックス
    h_deg    = gdf_target.loc[adj_idx, 'distance'].to_numpy().max() # 度単位の距離
    h_km     = km_per_degree * h_deg # キロメートル単位の距離

    # 重心座標を取得
    O_x, O_y = gdf_target.loc[n, 'centroids'].coords[0] # 各区域の座標

    # バンド幅の座標を計算
    t_vec = np.linspace(start=0.0, stop=2.0*np.pi, num=361) # ラジアン
    x_vec = O_x + h_deg * np.cos(t_vec) # 経度
    y_vec = O_y + h_deg * np.sin(t_vec) # 緯度

    # ラベルを作成
    param_lbl = f'$N = {N}, i = {n+1}, k = {k}, h = {h_km:.1f}\ (km)$'

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
    gdf_target.centroids.plot(
        ax=ax, 
        color='black', markersize=50, 
        label='centroids\nrepresentative point'
    ) # 重心座標
    for i in range(N):
        adj_idx, = np.where(adj_mat[i] == 1) # 隣接区域のインデックス
        for j in adj_idx:
            O_x, O_y = gdf_target.loc[i, 'centroids'].coords[0] # 各区域の座標
            P_x, P_y = gdf_target.loc[j, 'centroids'].coords[0] # 隣接区域の座標
            mutual_flg = adj_mat[j, i] == 1 # 双方向フラグ:(ベクトルの書き分け用)
            ax.plot(
                [O_x, P_x], 
                [O_y, P_y], 
                color='C0', 
                linewidth=3.0 if i == n else 1.0, 
                linestyle='-' if mutual_flg else '--'
            ) # 各区域 - 隣接区域
            if not mutual_flg:
                ax.quiver(
                    O_x, O_y, 
                    P_x-O_x, P_y-O_y, 
                    angles='xy', scale_units='xy', scale=1.0, 
                    units='dots', width=0.2, 
                    headwidth=50.0, headlength=100.0, headaxislength=100.0, # (width引数に対する倍率)
                    color='C0'
                ) # 隣接関係の向き:(ベクトルの書き分け用)
    ax.plot(
        x_vec, y_vec, 
        color='black', linewidth=1.5, linestyle='-.', 
        label='bandwidth'
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
    target_bool_mat    = np.tile(True, reps=adj_mat.shape)
    target_bool_mat[n] = False
    target_masked_mat  = np.ma.masked_array(adj_mat, target_bool_mat) # 対象区域 - 全区域
    adj_idx, = np.where(adj_mat[n] == 1) # 隣接区域のインデックス
    adj_bool_mat       = np.tile(True, reps=adj_mat.shape)
    adj_bool_mat[n, adj_idx] = False
    adj_masked_mat     = np.ma.masked_array(adj_mat, adj_bool_mat) # 対象区域 - 隣接区域

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
    filename=dir_path+'adjacency_mat_i.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%

# 近傍数の影響：1区域 ------------------------------------------------------------

### パラメータの設定 -----

# 区域数を取得
N = len(gdf_target)

# 近傍数の最大値を指定
max_K = N - 1

# フレーム数を設定
frame_num = max_K


# 区域を指定
area_idx = 22

# 距離を計算
gdf_target['distance'] = gdf_target['centroids'].distance(gdf_target.loc[area_idx, 'centroids'])


# %%

### 作図 -----

# 軸の範囲を設定
margin_ratio = 0.05
lon_min, lat_min, lon_max, lat_max = gdf_target.total_bounds
lon_min -= (lon_max - lon_min) * margin_ratio
lon_max += (lon_max - lon_min) * margin_ratio
lat_min -= (lat_max - lat_min) * margin_ratio
lat_max += (lat_max - lat_min) * margin_ratio
w_min, w_max = 0.0, 1.0 # 最小値・最大値


# グラフオブジェクトを初期化
fig, axes = plt.subplots(
    nrows=1, ncols=2, 
    figsize=(15, 6), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('spatial adjacency matrix: k-NN', fontsize=20)

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

    # 隣接数を設定
    k = frame_i + 1

    # 空間隣接行列を作成
    adj_obj = weights.distance.KNN.from_dataframe(
        df=gdf_target, geom_col='geometry', k=k
    )
    adj_mat, _ = adj_obj.full()
    adj_mat = adj_mat.astype(dtype=np.int8)

    ### コロプレス図の作図 -----

    # 隣接関係を格納
    gdf_target['adjacency'] = adj_mat[area_idx]

    # 距離を計算
    gdf_target['distance'] = gdf_target['centroids'].distance(gdf_target.loc[area_idx, 'centroids'])

    # バンド幅を取得
    adj_idx, = np.where(adj_mat[area_idx] == 1) # 隣接区域のインデックス
    h_deg    = gdf_target.loc[adj_idx, 'distance'].to_numpy().max() # 度単位の距離
    h_km     = km_per_degree * h_deg # キロメートル単位の距離

    # 重心座標を取得
    O_x, O_y = gdf_target.loc[area_idx, 'centroids'].coords[0] # 各区域の座標

    # バンド幅の座標を計算
    t_vec = np.linspace(start=0.0, stop=2.0*np.pi, num=361) # ラジアン
    x_vec = O_x + h_deg * np.cos(t_vec) # 経度
    y_vec = O_y + h_deg * np.sin(t_vec) # 緯度

    # ラベルを作成
    param_lbl = f'$N = {N}, i = {area_idx+1}, k = {k}, h = {h_km:.1f}\ (km)$'

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
    gdf_target.centroids.plot(
        ax=ax, 
        color='black', markersize=50, 
        label='centroids\nrepresentative point'
    ) # 重心座標
    adj_idx, = np.where(adj_mat[area_idx] == 1) # 隣接区域のインデックス
    for j in adj_idx:
        P_x, P_y = gdf_target.loc[j, 'centroids'].coords[0] # 隣接区域の座標
        mutual_flg = adj_mat[j, area_idx] == 1 # 双方向フラグ:(ベクトルの書き分け用)
        ax.plot(
            [O_x, P_x], 
            [O_y, P_y], 
            color='C0', linewidth=1.5, linestyle='-' if mutual_flg else '--'
        ) # 各区域 - 隣接区域
        if not mutual_flg:
            ax.quiver(
                O_x, O_y, 
                P_x-O_x, P_y-O_y, 
                angles='xy', scale_units='xy', scale=1.0, 
                units='dots', width=0.2, 
                headwidth=50.0, headlength=100.0, headaxislength=100.0, # (width引数に対する倍率)
                color='C0'
            ) # 隣接関係の向き:(ベクトルの書き分け用)
    ax.plot(
        x_vec, y_vec, 
        color='black', linewidth=1.5, linestyle='-.', 
        label='bandwidth'
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
    ax.legend(loc='upper right')
    ax.grid()
    ax.set_xlim(xmin=lon_min, xmax=lon_max)
    ax.set_ylim(ymin=lat_min, ymax=lat_max)
    ax.set_aspect(aspect='equal', adjustable='box')
    
    ### ヒートマップの作図 -----

    # 枠線の表示位置を設定
    target_bool_mat   = np.tile(True, reps=adj_mat.shape)
    target_bool_mat[area_idx] = False
    target_masked_mat = np.ma.masked_array(adj_mat, target_bool_mat) # 対象区域 - 全区域
    adj_idx, = np.where(adj_mat[area_idx] == 1) # 隣接区域のインデックス
    adj_bool_mat      = np.tile(True, reps=adj_mat.shape)
    adj_bool_mat[area_idx, adj_idx] = False
    adj_masked_mat    = np.ma.masked_array(adj_mat, adj_bool_mat) # 対象区域 - 隣接区域

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
    filename=dir_path+'adjacency_mat_k_one.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%

# 近傍数の影響：全区域 ------------------------------------------------------------

### パラメータの設定 -----

# 区域数を取得
N = len(gdf_target)

# 近傍数の最大値を指定
max_K = N - 1

# フレーム数を設定
frame_num = max_K


# %%

### 作図 -----

# 軸の範囲を設定
u = 5.0
k_min, k_max = 0.0, max_K
k_max = np.ceil(k_max /u)*u  # u単位で切り上げ
w_min, w_max = 0.0, 1.0 # 最小値・最大値


# グラフオブジェクトを初期化
fig, axes = plt.subplots(
    nrows=1, ncols=2, 
    figsize=(15, 6), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('spatial adjacency matrix: k-NN', fontsize=20)

# 装飾用のダミーを設定
ax = axes[0]
gdf_target['cardinality'] = 0.0
gdf_target.plot(
    ax=ax, column='cardinality', 
    cmap='viridis', alpha=0.5, vmin=k_min, vmax=k_max, 
    legend=True, legend_kwds={'label': '$k$', 'shrink': 1.0}
) # 隣接数軸
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

    # 隣接数を設定
    k = frame_i + 1

    # 空間隣接行列を作成
    adj_obj = weights.distance.KNN.from_dataframe(
        df=gdf_target, geom_col='geometry', k=k
    )
    adj_mat, _ = adj_obj.full()
    adj_mat = adj_mat.astype(dtype=np.int8)

    ### コロプレス図の作図 -----

    # 隣接数を取得
    gdf_target['cardinality'] = np.array(list(adj_obj.cardinalities.values()))

    # バンド幅を取得
    adj_idx_lt = [np.where(adj_mat[i] == 1)[0] for i in range(N)] # 隣接区域のインデックス
    h_km_vals  = km_per_degree * np.array(
        [gdf_target.loc[adj_idx_lt[i], 'centroids'].distance(gdf_target.loc[i, 'centroids']).max() for i in range(N)]
    )

    # ラベルを作成
    param_lbl = f'$N = {N}, k = {k}$'

    # コロプレス図を描画
    ax = axes[0]
    gdf_target.boundary.plot(
        ax=ax, 
        edgecolor='black', linewidth=0.5
    ) # 行政区界
    gdf_target.plot(
        ax=ax, column='cardinality', 
        cmap='viridis', alpha=0.5, vmin=k_min, vmax=k_max
    ) # 隣接数
    gdf_target.centroids.plot(
        ax=ax, 
        color='black', markersize=50, 
        label='centroids\nrepresentative point'
    ) # 重心座標
    for i in range(N):
        adj_idx, = np.where(adj_mat[i] == 1) # 隣接区域のインデックス
        for j in adj_idx:
            O_x, O_y = gdf_target.loc[i, 'centroids'].coords[0] # 各区域の座標
            P_x, P_y = gdf_target.loc[j, 'centroids'].coords[0] # 隣接区域の座標
            mutual_flg = adj_mat[j, i] == 1 # 双方向フラグ:(ベクトルの書き分け用)
            ax.plot(
                [O_x, P_x], 
                [O_y, P_y], 
                color='C0', linewidth=1.0, linestyle='-' if mutual_flg else '--'
            ) # 各区域 - 隣接区域
            if not mutual_flg:
                ax.quiver(
                    O_x, O_y, 
                    P_x-O_x, P_y-O_y, 
                    angles='xy', scale_units='xy', scale=1.0, 
                    units='dots', width=0.2, 
                    headwidth=50.0, headlength=100.0, headaxislength=100.0, # (width引数に対する倍率)
                    color='C0'
                ) # 隣接関係の向き:(ベクトルの書き分け用)
    for x, y, area_lbl in zip(gdf_target['centroids'].x, gdf_target['centroids'].y, gdf_target['city2']):
        ax.text(
            x=x, y=y, 
            s=area_lbl, ha='right', va='bottom', 
            size=10
        ) # 区域名
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_title(param_lbl, loc='left')
    ax.grid()
    ax.set_aspect(aspect='equal', adjustable='box')
    
    ### ヒートマップの作図 -----

    # ヒートマップを描画
    ax = axes[1]
    ax.pcolormesh(
        adj_mat, 
        cmap=cmap, vmin=w_min, vmax=w_max, 
        shading='auto'
    ) # 全区域 - 全区域
    ax.set_xticks(ticks=np.arange(N)+0.5)
    ax.set_xticklabels(labels=gdf_target['city2'], size=10, rotation=90) # 区域名
    ax.set_yticks(ticks=np.arange(N)+0.5)
    ax.set_yticklabels(
        labels=[f'{lbl} ($h = {h:.1f}$)' for lbl, h in zip(gdf_target['city2'], h_km_vals)], 
        size=10
    ) # 区域名, バンド幅
    ax.set_xlabel('$j$')
    ax.set_ylabel('$i$')
    ax.set_title('$h:\ km$', loc='left')
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
    filename=dir_path+'adjacency_mat_k_all.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%


