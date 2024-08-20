# 『Pythonで学ぶ空間データサイエンス入門』のノート
# 第3章 線形回帰モデルと最小二乗法
# 3.2.1-3 線形回帰モデルの最小二乗法の可視化
# 2次元変数の場合

# %%

# ライブラリを読込
import numpy as np
from scipy.stats import norm # 確率密度の計算用
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation # アニメーションの作図用


# %%

### 最小二乗法の実装 -------------------------------------------------------------

# 線形回帰モデルに対するOLSを実装
class OLS:

    # データの設定
    def __init__(self, X, y):
        
        # 観測データを格納
        self.X = X # 説明変数
        self.y = y # 被説明変数
        self.N = X.shape[0] # 観測データ数
        self.K = X.shape[1] - 1 # 係数パラメータ数

    # 係数パラメータの推定
    def get_beta_hat(self):
        
        # 観測データを取得
        X = self.X
        y = self.y.flatten() # (横ベクトルで扱う場合)
        
        # 係数パラメータの推定値を計算
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        
        return beta_hat
    
    # 誤差項の分散パラメータの推定
    def get_sigma2_hat(self):
        
        # 観測データを取得
        X = self.X
        y = self.y.flatten() # (横ベクトルで扱う場合)

        # 係数パラメータの推定値を計算
        beta_hat = self.get_beta_hat()

        # 被説明変数の推定値を計算
        y_hat = X @ beta_hat
        
        # 残差を計算
        e = y - y_hat
        
        # 残差平方和を計算
        rss = e.T @ e
        
        # 分散パラメータの推定値を計算
        sigma2_hat = rss / (self.N - self.K - 1)
        
        return sigma2_hat
    
    # 係数パラメータの推定値の分散共分散行列の推定
    def get_sigma_mat_hat(self):

        # 観測データを取得
        X = self.X

        # 分散パラメータの推定値を計算
        sigma2_hat = self.get_sigma2_hat()
        
        # 係数パラメータの推定値の分散共分散行列を計算
        Sigma_beta = sigma2_hat * np.linalg.inv(X.T @ X)

        return Sigma_beta


# %%

### データの生成 ----------------------------------------------------------------

# データ数(フレーム数)を指定
N = 100

# 係数パラメータ数を指定:(固定)
K = 2

# 説明変数の値を生成
data_X = np.hstack(
    [np.ones(shape=(N, 1)), 
     np.random.uniform(low=-5.0, high=5.0, size=(N, K))]
)
print(data_X[:5].round(2))
print(data_X.shape)

# 真の係数パラメータを指定
beta_true = np.array([2.0, 1.0, -0.5])

# 誤差項の真の標準偏差パラメータを指定
sigma_true = 2.5

# 誤差項の真の分散パラメータを計算
sigma2_true = sigma_true**2

# 誤差項を生成
data_epsilon = np.random.normal(loc=0.0, scale=sigma_true, size=N)

# 被説明変数を計算
data_y = data_X @ beta_true + data_epsilon
print(data_y[:5].round(2))
print(data_y.shape)


# %%

### 係数パラメータの図：データ数の影響 --------------------------------------------

# グラフサイズを設定
x1_min = np.floor(data_X[:, 1].min())
x1_max = np.ceil(data_X[:, 1].max())
x2_min = np.floor(data_X[:, 2].min())
x2_max = np.ceil(data_X[:, 2].max())

# 座標計算用の格子点を作成
x1_vec = np.linspace(start=x1_min, stop=x1_max, num=11)
x2_vec = np.linspace(start=x2_min, stop=x2_max, num=11)
x1_grid, x2_grid = np.meshgrid(x1_vec, x2_vec)

# 真のモデルの座標を作成
y_true_grid   = beta_true[0] + beta_true[1] * x1_grid + beta_true[2] * x2_grid
y_true_x1_vec = beta_true[0] + beta_true[1] * x1_vec
y_true_x2_vec = beta_true[0] + beta_true[2] * x2_vec

# グラフサイズを設定
y_min = np.floor(min(data_y.min(), y_true_grid.min()))
y_max = np.ceil(max(data_y.max(), y_true_grid.max()))

# 矢サイズを指定
alr = 0.5

# 補助線の位置の調整用
offset_x = 0.025

# グラフオブジェクトを初期化
fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='white', constrained_layout=True, 
                       subplot_kw={'projection': '3d'})
fig.suptitle('Linear Regression Model', fontsize=20)

# 作図処理を定義
def update(i):
    
    # 前フレームのグラフを初期化
    ax.cla()
    
    # 計算できない範囲を除く
    i += 1

    # i個の観測データを取得
    X = data_X[:(i+1)]
    y = data_y[:(i+1)]
    
    # 係数パラメータの推定値を計算
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # 被説明変数を計算
    y_true = X @ beta_true
    y_hat  = X @ beta_hat

    # 回帰平面の座標を作成
    y_hat_grid   = beta_hat[0] + beta_hat[1] * x1_grid + beta_hat[2] * x2_grid
    y_hat_x1_vec = beta_hat[0] + beta_hat[1] * x1_vec
    y_hat_x2_vec = beta_hat[0] + beta_hat[2] * x2_vec
    
    # ラベル用の文字列を作成
    param_label  = f'$N = {i+1}, K = {K}, '
    param_label += f'\\beta = (' + ', '.join(f'{val:.2f}' for val in beta_true) +'), '
    param_label += f'\\hat{{\\beta}} = (' + ', '.join(f'{val:.2f}' for val in beta_hat) +')$'

    # 回帰平面を描画
    
    # x1・x2・y空間
    ax.plot_wireframe(x1_grid, x2_grid, y_true_grid, 
                      color='red', alpha=0.5, linewidth=1, 
                      label='$y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2$') # 真のモデル
    ax.plot(x1_vec, np.zeros(len(x1_vec)), y_true_x1_vec, 
            color='red', linewidth=2) # 真のモデル:x1軸方向
    ax.plot(np.zeros(len(x2_vec)), x2_vec, y_true_x2_vec, 
            color='red', linewidth=2) # 真のモデル:x2軸方向
    ax.plot_wireframe(x1_grid, x2_grid, y_hat_grid, 
                      color='orange', alpha=0.5, linewidth=1, 
                      label='$\\hat{y} = \\hat{\\beta}_0 + \\hat{\\beta}_1 x_1 + \\hat{\\beta}_2 x_2$') # 回帰平面
    ax.plot(x1_vec, np.zeros(len(x1_vec)), y_hat_x1_vec, 
            color='orange', linewidth=2) # 回帰直線:x1軸方向
    ax.plot(np.zeros(len(x2_vec)), x2_vec, y_hat_x2_vec, 
            color='orange', linewidth=2) # 回帰直線:x2軸方向
    ax.scatter(X[:, 1], X[:, 2], y, 
               color='C0', s=50, label='$(x_{i1}, y_i)$') # 実現値
    for j in range(i+1):
        ax.plot([X[j, 1]-offset_x, X[j, 1]-offset_x], 
                [X[j, 2]-offset_x, X[j, 2]-offset_x], 
                [y_true[j]-offset_x, y[j]-offset_x], 
                color='red', linestyle='dotted') # 誤差
        ax.plot([X[j, 1]+offset_x, X[j, 1]+offset_x], 
                [X[j, 2]+offset_x, X[j, 2]+offset_x], 
                [y_hat[j]+offset_x, y[j]+offset_x], 
                color='orange', linestyle='dotted') # 残差
    ax.plot(0.0, 0.0, color='red', linestyle='dotted', 
            label='$\\epsilon_i = y_i - x_i^{T} \\beta$') # 凡例用のダミー
    ax.plot(0.0, 0.0, color='orange', linestyle='dotted', 
            label='$e_i = y_i - \\hat{y}_i$') # 凡例用のダミー
    
    # x2・y平面
    ax.quiver(x1_min, x2_max, 0, x1_max-x1_min, 0, 0, 
              arrow_length_ratio=alr/(x1_max-x1_min), 
              color='black', linewidth=1) # x1軸線
    ax.quiver(0, x2_max, y_min, 0, 0, y_max-y_min, 
              arrow_length_ratio=alr/(y_max-y_min), 
              color='black', linewidth=1) # y軸線
    ax.plot(x1_vec, x2_max.repeat(len(x1_vec)), y_true_x1_vec, 
            color='red', linewidth=1, linestyle='dashed', 
            label='$y = \\beta_0 + \\beta_k x_k$') # 真のモデル:x1軸方向
    ax.plot(x1_vec, x2_max.repeat(len(x1_vec)), y_hat_x1_vec, 
            color='orange', linewidth=1, linestyle='dashed', 
            label='$\\hat{y} = \\hat{\\beta}_0 + \\hat{\\beta}_k x_k$') # 回帰直線
    
    # x1・y平面
    ax.quiver(x1_min, x2_min, 0, 0, x2_max-x2_min, 0, 
              arrow_length_ratio=alr/(x2_max-x2_min), 
              color='black', linewidth=1) # x2軸線
    ax.quiver(x1_min, 0, y_min, 0, 0, y_max-y_min, 
              arrow_length_ratio=alr/(y_max-y_min), 
              color='black', linewidth=1) # y軸線
    ax.plot(x1_min.repeat(len(x2_vec)), x2_vec, y_true_x2_vec, 
            color='red', linewidth=1, linestyle='dashed') # 真のモデル:x2軸方向
    ax.plot(x1_min.repeat(len(x2_vec)), x2_vec, y_hat_x2_vec, 
            color='orange', linewidth=1, linestyle='dashed') # 回帰直線
    
    ax.set_xlim(xmin=x1_min, xmax=x1_max)
    ax.set_ylim(ymin=x2_min, ymax=x2_max)
    ax.set_zlim(zmin=y_min, zmax=y_max)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$y$')
    ax.set_title(param_label, loc='left')
    ax.legend(loc='upper left')
    #ax.view_init(elev=30, azim=90) # 表示アングル

# 動画を作成
ani = FuncAnimation(fig=fig, func=update, frames=N-1, interval=100)

# 動画を書出
ani.save(
    filename='../../figure/ch3/2d_ols/2d_ols_beta_n.mp4', 
    progress_callback = lambda i, n: print(f'frame: {i} / {n}')
)


# %%

### 分散パラメータの図：データ数の影響 --------------------------------------------

# 正規分布の真の確率密度を計算
eps_size = max(data_epsilon.max(), abs(data_epsilon.min()), 2.0*sigma_true)
line_eps      = np.linspace(start=-eps_size, stop=eps_size, num=1000)
dens_eps_true = norm.pdf(line_eps, loc=0.0, scale=sigma_true)

# グラフサイズを設定
dens_max = 0.2

# グラフオブジェクトを初期化
fig, ax = plt.subplots(figsize=(8, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle('Normal Distribution', fontsize=20)

# 作図処理を定義
def update(i):
    
    # 前フレームのグラフを初期化
    ax.cla()
    
    # 計算できない範囲を除く
    i += 1
    
    # i個の観測データを取得
    X = data_X[:(i+1)]
    y = data_y[:(i+1)]
    
    # OLSのインスタンスを作成
    ols = OLS(X, y)
    
    # 分散パラメータの推定値を計算
    sigma2_hat = ols.get_sigma2_hat()
    sigma_hat  = np.sqrt(sigma2_hat)
    
    # 正規分布の確率密度を計算
    dens_eps_hat = norm.pdf(line_eps, loc=0.0, scale=sigma_hat)
    
    # ラベル用の文字列を作成
    dist_label  = f'$N = {i+1}, '
    dist_label += f'\\sigma = {sigma_true}, \\sigma^2 = {sigma2_true:.3f}, '
    dist_label += f'\\hat{{\\sigma}} = {sigma_hat:.3f}, \\hat{{\\sigma}}^2 = {sigma2_hat:.3f}$'
    
    # 正規分布を描画
    ax.plot(line_eps, dens_eps_true, 
            color='red', linestyle='dashed', label='$N(0, \\sigma^2)$') # 真の確率密度
    ax.plot(line_eps, dens_eps_hat, 
            color='black', label='$N(0, \\hat{\\sigma}^2)$') # 確率密度の推定値
    ax.scatter(data_epsilon[:(i+1)], np.zeros(i+1), 
               facecolor='none', edgecolor='deeppink', s=50, 
               label='$\\epsilon_i \\sim N(0, \\sigma^2)$') # 誤差項
    ax.set_ylim(ymin=-0.1*dens_max, ymax=dens_max)
    ax.set_xlabel('$\\epsilon$')
    ax.set_ylabel('density')
    ax.set_title(dist_label, loc='left')
    ax.grid()
    ax.legend(loc='upper left')

# 動画を作成
ani = FuncAnimation(fig=fig, func=update, frames=N-1, interval=100)

# 動画を書出
ani.save(
    filename='../../figure/ch3/2d_ols/2d_ols_sigma2_n.mp4', 
    progress_callback = lambda i, n: print(f'frame: {i} / {n}')
)


# %%

### 係数パラメータの分散共分散行列の図：データ数の影響 -------------------------------

# 座標の作成

# 変数の作成用のパラメータを作成
ols = OLS(data_X, data_y)
Sigma_beta_hat = ols.get_sigma_mat_hat()

# 各軸の範囲の調整値を指定
sgm_num = 5.0

# 各軸の範囲を設定
beta0_min = beta_true[0] - sgm_num * np.sqrt(Sigma_beta_hat[0, 0])
beta0_max = beta_true[0] + sgm_num * np.sqrt(Sigma_beta_hat[0, 0])
beta1_min = beta_true[1] - sgm_num * np.sqrt(Sigma_beta_hat[1, 1])
beta1_max = beta_true[1] + sgm_num * np.sqrt(Sigma_beta_hat[1, 1])
beta2_min = beta_true[2] - sgm_num * np.sqrt(Sigma_beta_hat[2, 2])
beta2_max = beta_true[2] + sgm_num * np.sqrt(Sigma_beta_hat[2, 2])

# 2軸方向の線の数を指定
line_num = 100

# 各軸の値を作成
beta0_vec = np.linspace(start=beta0_min, stop=beta0_max, num=50) # 0軸方向の点の数を指定
beta1_vec = np.linspace(start=beta1_min, stop=beta1_max, num=50) # 1軸方向の点の数を指定
beta2_vec = np.linspace(start=beta2_min, stop=beta2_max, num=line_num)
print(beta0_vec[:5].round(2))
print(beta1_vec[:5].round(2))
print(beta2_vec[:5].round(2))

# 0・1軸の格子点を作成
beta0_grid, beta1_grid = np.meshgrid(beta0_vec, beta1_vec)

# 0・1軸の形状を設定
grid_shape = beta0_grid.shape
grid_size  = beta0_grid.size
print(grid_shape)
print(grid_size)

# %%

## 距離の計算

# フレームごとに処理
trace_sigma_lt = [np.tile(np.nan, reps=(2, 2)) for _ in range(2)]
trace_delta_lt = [ np.tile(np.nan, reps=(line_num, grid_size)) for _ in range(2)]
for i in range(2, N):

    # i個の観測データを取得
    X = data_X[:(i+1)]
    y = data_y[:(i+1)]
    
    # OLSのインスタンスを作成
    ols = OLS(X, y)
    
    # 係数パラメータの推定値の分散共分散行列を計算
    Sigma_beta_hat = ols.get_sigma_mat_hat()
    
    # 係数パラメータの推定値の精度行列を計算
    inv_Sigma_beta_hat = np.linalg.inv(Sigma_beta_hat)
    
    # 高さごとに処理
    delta_arr = np.tile(np.nan, reps=(line_num, grid_size)) # 受け皿
    for l in range(line_num):
        
        # 座標を作成
        beta_arr = np.stack(
            [beta0_grid.flatten(), beta1_grid.flatten(), beta2_vec[l].repeat(grid_size)], 
            axis=1
        )
        
        # マハラノビス距離を計算
        delta_vec = np.array(
            [np.sqrt((beta - beta_true).T @ inv_Sigma_beta_hat @ (beta - beta_true)) for beta in beta_arr]
        )
        
        # 距離を格納
        delta_arr[l] = delta_vec.copy()
    
    # 結果を格納
    trace_sigma_lt.append(Sigma_beta_hat.copy())
    trace_delta_lt.append(delta_arr)
    
    # 途中経過を表示
    print(f'frame: {i+1} / {N}')

# %%

## アニメーションの作成

# グラデーションの範囲を設定
delta_min = 0.0
delta_max = np.ceil(max([arr.max() for arr in trace_delta_lt[3:]])) # (N > (K+1)の範囲を除く)

# 等高線の位置を指定
delta_levels = np.linspace(start=delta_min, stop=delta_max, num=11) # 線の数を指定
print(delta_levels)

# 軸サイズを設定
beta0_size = beta0_max - beta0_min
beta1_size = beta1_max - beta1_min
beta2_size = beta2_max - beta2_min

# %%

# グラフオブジェクトを初期化
fig, ax = plt.subplots(figsize=(9, 8), dpi=100, facecolor='white', constrained_layout=True, 
                       subplot_kw={'projection': '3d'})
fig.suptitle('Mahalanobis distance', fontsize=20)
cs = ax.contour(beta0_grid, beta1_grid, np.linspace(delta_min, delta_max, num=grid_size).reshape(grid_shape), offset=0.0, 
                cmap='viridis', vmin=delta_min, vmax=delta_max, levels=delta_levels) # カラーバー表示用のダミー
fig.colorbar(cs, ax=ax, shrink=0.6, label='$\\Delta$')

# 作図処理を定義
def update(i):
    
    # 前フレームのグラフを初期化
    ax.cla()

    # 計算できない範囲を除く
    i += 2
    
    # i個の観測データを取得
    X = data_X[:(i+1)]
    y = data_y[:(i+1)]
    
    # i個のデータにおけるパラメータを取得
    Sigma_beta_hat = trace_sigma_lt[i]

    # OLSのインスタンスを作成
    ols = OLS(X, y)
    
    # 係数パラメータの推定値を計算
    beta_hat = ols.get_beta_hat()
    
    # マハラノビス距離を計算
    delta_val  = np.sqrt((beta_hat - beta_true).T @ np.linalg.inv(Sigma_beta_hat) @ (beta_hat - beta_true))
    
    # ラベル用の文字列を作成
    arr_str = '(' + ', '.join('(' + ', '.join(str(val.round(3)) for val in vec) + ')' for vec in Sigma_beta_hat) + ')'
    delta_label  = f'$N = {i+1}, K = {K}, ' + '\\Sigma_{\\hat{\\beta}} = ' + arr_str
    delta_label += f', \\Delta_{{\\hat{{\\beta}}}} = {delta_val:.2f}$'
    
    # マハラノビス距離を描画
    for l in range(line_num):
        ax.contour(beta0_grid, beta1_grid, trace_delta_lt[i][l].reshape(grid_shape), offset=beta2_vec[l], 
                   cmap='viridis', vmin=delta_min, vmax=delta_max, levels=delta_levels, alpha=0.5, 
                   linewidths=1) # マハラノビス距離
    ax.scatter(*beta_true, 
               c='red', s=100, 
               label='$\\beta = ('+', '.join([f'{val:.2f}' for val in beta_true])+')$') # 真のパラメータ
    ax.scatter(*beta_hat, 
               c='orange', s=100, 
               label='$\\hat{\\beta} = ('+', '.join([f'{val:.2f}' for val in beta_hat])+')$') # 推定パラメータ
    ax.plot([beta_true[0], beta_hat[0]], 
            [beta_true[1], beta_hat[1]], 
            [beta_true[2], beta_hat[2]], 
            color='black', linestyle='dotted') # 距離
    ax.set_xlim(xmin=beta0_min, xmax=beta0_max)
    ax.set_ylim(ymin=beta1_min, ymax=beta1_max)
    ax.set_zlim(zmin=beta2_min, zmax=beta2_max)
    ax.set_xlabel('$\\hat{\\beta}_0$')
    ax.set_ylabel('$\\hat{\\beta}_1$')
    ax.set_zlabel('$\\hat{\\beta}_2$')
    ax.set_title(delta_label, loc='left')
    ax.legend(loc='upper left')
    ax.set_zlim(zmin=beta2_min, zmax=beta2_max)
    #ax.view_init(elev=90, azim=90) # 表示アングル

# 動画を作成
ani = FuncAnimation(fig=fig, func=update, frames=N-2, interval=100)

# 動画を書出
ani.save(
    filename='../../figure/ch3/2d_ols/2d_ols_sigma_beta_n.mp4', 
    progress_callback = lambda i, n: print(f'frame: {i} / {n}')
)


# %%


