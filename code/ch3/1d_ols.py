# 『Pythonで学ぶ空間データサイエンス入門』のノート
# 第3章 線形回帰モデルと最小二乗法
# 3.2.1-3 線形回帰モデルの最小二乗法の可視化
# 1次元変数の場合

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
K = 1

# 説明変数の値を生成
data_X = np.hstack(
    [np.ones(shape=(N, 1)), 
     np.random.uniform(low=-10.0, high=10.0, size=(N, K))] # 範囲を指定
)

# 真の係数パラメータを指定
beta_true = np.array([2.0, 0.5])

# 誤差項の真の標準偏差パラメータを指定
sigma_true = 0.5

# 誤差項の真の分散パラメータを計算
sigma2_true = sigma_true**2

# 誤差項を生成
data_epsilon = np.random.normal(loc=0.0, scale=sigma_true, size=N)

# 被説明変数を計算
data_y = data_X @ beta_true + data_epsilon


# %%

### 係数パラメータの図：データ数の影響 --------------------------------------------

# グラフサイズを設定
x_min = np.floor(data_X[:, 1].min())
x_max = np.ceil(data_X[:, 1].max())

# 真のモデルの座標を作成
line_x      = np.linspace(start=x_min, stop=x_max, num=1001)
line_y_true = beta_true[1] * line_x + beta_true[0]

# グラフサイズを設定
y_min = np.floor(min(data_y.min(), line_y_true.min()))
y_max = np.ceil(max(data_y.max(), line_y_true.max()))

# 補助線の位置の調整用
offset_x = 0.03

# グラフオブジェクトを初期化
fig, ax = plt.subplots(figsize=(8, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle('Linear Regression Model', fontsize=20)

# 作図処理を定義
def update(i):
    
    # 前フレームのグラフを初期化
    ax.cla()
    
    # i個の観測データを取得
    X = data_X[:(i+1)]
    y = data_y[:(i+1)]
    
    # 係数パラメータの推定値を計算
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # 被説明変数を計算
    y_true = X @ beta_true
    y_hat  = X @ beta_hat
    
    # 回帰直線の座標を計算
    line_y_hat = beta_hat[1] * line_x + beta_hat[0]
    
    # ラベル用の文字列を作成
    param_label  = f'$N = {i+1}, K = {K}, '
    param_label += f'\\beta = ({beta_true[0]}, {beta_true[1]}), '
    param_label += f'\\hat{{\\beta}} = ({beta_hat[0]:.3f}, {beta_hat[1]:.3f})$'
    
    # 回帰直線を描画
    ax.quiver(x_min, 0.0, x_max-x_min, 0.0, 
              angles='xy', scale_units='xy', scale=1.0, 
              units='dots', width=2.5, headwidth=5.0, headlength=7.5, headaxislength=7.5) # x軸線
    ax.quiver(0.0, y_min, 0.0, y_max-y_min, 
              angles='xy', scale_units='xy', scale=1.0, 
              units='dots', width=2.5, headwidth=5.0, headlength=7.5, headaxislength=7.5) # y軸線
    ax.plot(line_x, line_y_true, 
            color='red', linestyle='dashed', label='$y = \\beta_0 + \\beta_1 x_1$') # 真のモデル
    ax.plot(line_x, line_y_hat, 
            color='orange', linewidth=2.0, label='$\\hat{y} = \\hat{\\beta}_0 + \\hat{\\beta}_1 x$') # 回帰直線
    ax.scatter(X[:, 1], y, 
               facecolor='none', edgecolor='C0', s=50, label='$(x_{i1}, y_i)$') # 実現値
    ax.plot([X[:, 1]-offset_x, X[:, 1]-offset_x], [y_true, y], 
            color='red', linestyle='dotted') # 誤差
    ax.plot([X[:, 1]+offset_x, X[:, 1]+offset_x], [y_hat, y], 
            color='orange', linestyle='dotted') # 残差
    ax.plot(0.0, 0.0, color='red', linestyle='dotted', label='$\\epsilon_i = y_i - x_i^{T} \\beta$') # 凡例用のダミー
    ax.plot(0.0, 0.0, color='orange', linestyle='dotted', label='$e_i = y_i - \\hat{y}_i$') # 凡例用のダミー
    ax.set_xlim(xmin=x_min, xmax=x_max)
    ax.set_ylim(ymin=y_min, ymax=y_max)
    ax.set_xlabel('$x_{i1}$')
    ax.set_ylabel('$y_i$')
    ax.set_title(param_label, loc='left')
    ax.grid()
    ax.legend(loc='upper left')

# 動画を作成
ani = FuncAnimation(fig=fig, func=update, frames=N, interval=100)

# 動画を書出
ani.save(
    filename='../../figure/ch3/1d_ols/1d_ols_beta_n.mp4', 
    progress_callback = lambda i, n: print(f'frame: {i} / {n}')
)


# %%

### 分散パラメータの図：データ数の影響 --------------------------------------------

# 正規分布の真の確率密度を計算
eps_size = max(data_epsilon.max(), abs(data_epsilon.min()), 2.0*sigma_true)
line_eps      = np.linspace(start=-eps_size, stop=eps_size, num=1000)
dens_eps_true = norm.pdf(line_eps, loc=0.0, scale=sigma_true)

# グラフサイズを設定
dens_max = 1.0

# グラフオブジェクトを初期化
fig, ax = plt.subplots(figsize=(8, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle('Normal Distribution', fontsize=20)

# 作図処理を定義
def update(i):
    
    # 前フレームのグラフを初期化
    ax.cla()
    
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
               facecolor='none', edgecolor='deeppink', s=50, label='$\\epsilon_i \\sim N(0, \\sigma^2)$') # 誤差項
    ax.set_ylim(ymin=-0.1*dens_max, ymax=dens_max)
    ax.set_xlabel('$\\epsilon$')
    ax.set_ylabel('density')
    ax.set_title(dist_label, loc='left')
    ax.grid()
    ax.legend(loc='upper left')

# 動画を作成
ani = FuncAnimation(fig=fig, func=update, frames=N, interval=100)

# 動画を書出
ani.save(
    filename='../../figure/ch3/1d_ols/1d_ols_sigma2_n.mp4', 
    progress_callback = lambda i, n: print(f'frame: {i} / {n}')
)


# %%

### 係数パラメータの分散共分散行列の図：データ数の影響 -------------------------------

## 座標の作成

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

# 各軸の値を作成
beta0_vec = np.linspace(start=beta0_min, stop=beta0_max, num=50)
beta1_vec = np.linspace(start=beta1_min, stop=beta1_max, num=50)

# 格子点を作成
beta0_grid, beta1_grid = np.meshgrid(beta0_vec, beta1_vec)

# 格子点の形状を設定
grid_shape = beta0_grid.shape
grid_size  = beta0_grid.size

# 座標を作成
beta_arr = np.stack([beta0_grid.flatten(), beta1_grid.flatten()], axis=1)
print(beta_arr.shape)

# %%

## 距離の計算

# フレームごとに処理
trace_sigma_lt = [np.tile(np.nan, reps=(2, 2))]
trace_dist_lt  = [np.repeat(np.nan, repeats=grid_size)]
for i in range(1, N):

    # i個の観測データを取得
    X = data_X[:(i+1)]
    y = data_y[:(i+1)]
    
    # OLSのインスタンスを作成
    ols = OLS(X, y)
    
    # 係数パラメータの推定値の分散共分散行列を計算
    Sigma_beta_hat = ols.get_sigma_mat_hat()
    
    # 係数パラメータの推定値の精度行列を計算
    inv_Sigma_beta_hat = np.linalg.inv(Sigma_beta_hat)
    
    # マハラノビス距離を計算
    dist_vec = np.array(
        [np.sqrt((beta - beta_true).T @ inv_Sigma_beta_hat @ (beta - beta_true)) for beta in beta_arr]
    )
    
    # 結果を格納
    trace_sigma_lt.append(Sigma_beta_hat.copy())
    trace_dist_lt.append(dist_vec)
    
    # 途中経過を表示
    print(f'frame: {i+1} / {N}')

# %%

## アニメーションの作成

# グラデーションの範囲を設定
dist_min = 0.0
dist_max = np.ceil(max([arr.max() for arr in trace_dist_lt[2:]])) # (N > (K+1)の範囲を除く)

# 等高線の位置を設定
dist_levels = np.linspace(start=dist_min, stop=dist_max, num=9) # 線の数を指定
print(dist_levels)

# %%

## 等高線図の作成

# グラフオブジェクトを初期化
fig, ax = plt.subplots(figsize=(8, 6), dpi=100, facecolor='white', constrained_layout=True)
fig.suptitle('Mahalanobis Distance', fontsize=20)
cs = ax.contour(beta0_grid, beta1_grid, np.linspace(dist_min, dist_max, num=grid_size).reshape(grid_shape), 
                cmap='viridis', vmin=dist_min, vmax=dist_max, levels=dist_levels) # カラーバー表示用のダミー
fig.colorbar(cs, ax=ax, shrink=1.0, label='distance')

# 作図処理を定義
def update(i):
    
    # 前フレームのグラフを初期化
    ax.cla()

    # 計算できない範囲を除く
    i += 1
    
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
    dist_val  = np.sqrt((beta_hat - beta_true).T @ np.linalg.inv(Sigma_beta_hat) @ (beta_hat - beta_true))
    
    # ラベル用の文字列を作成
    arr_str = '(' + ', '.join('(' + ', '.join(str(val.round(3)) for val in vec) + ')' for vec in Sigma_beta_hat) + ')'
    dist_label  = f'$N = {i+1}, K = {K}, ' + '\\Sigma_{\\hat{\\beta}} = '+arr_str
    dist_label += f', \\Delta_{{\\hat{{\\beta}}}} = {dist_val:.2f}$'
    
    # マハラノビス距離を描画
    ax.contour(beta0_grid, beta1_grid, trace_dist_lt[i].reshape(beta0_grid.shape), 
               vmin=dist_min, vmax=dist_max, levels=dist_levels) # マハラノビス距離
    ax.scatter(*beta_true, 
               color='red', s=100, 
               label=f'$\\beta = ({beta_true[0]}, {beta_true[1]})$') # 真のパラメータ
    ax.scatter(*beta_hat, 
               color='orange', s=100, 
               label=f'$\\hat{{\\beta}} = ({beta_hat[0]:.2f}, {beta_hat[1]:.2f})$') # 推定パラメータ
    ax.plot([beta_true[0], beta_hat[0]], [beta_true[1], beta_hat[1]], 
            color='black', linestyle='dotted') # 距離
    ax.set_xlim(xmin=beta0_min, xmax=beta0_max)
    ax.set_ylim(ymin=beta1_min, ymax=beta1_max)
    ax.set_xlabel('$\\hat{\\beta}_0$')
    ax.set_ylabel('$\\hat{\\beta}_1$')
    ax.set_title(dist_label, loc='left')
    ax.grid()
    ax.legend(loc='upper left')
    #ax.set_aspect('equal', adjustable='box')

# 動画を作成
ani = FuncAnimation(fig=fig, func=update, frames=N-1, interval=100)

# 動画を書出
ani.save(
    filename='../../figure/ch3/1d_ols/1d_ols_sigma_beta_n_contour.mp4', 
    progress_callback = lambda i, n: print(f'frame: {i} / {n}')
)

# %%

## 曲面図の作成

# 軸サイズを設定
beta0_size = beta0_max - beta0_min
beta1_size = beta1_max - beta1_min

# グラフオブジェクトを初期化
fig, ax = plt.subplots(figsize=(9, 8), dpi=100, facecolor='white', constrained_layout=True, 
                       subplot_kw={'projection': '3d'})
fig.suptitle('Mahalanobis Distance', fontsize=20)

# 作図処理を定義
def update(i):
    
    # 前フレームのグラフを初期化
    ax.cla()

    # 計算できない範囲を除く
    i += 1
    
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
    dist_val  = np.sqrt((beta_hat - beta_true).T @ np.linalg.inv(Sigma_beta_hat) @ (beta_hat - beta_true))
    
    # ラベル用の文字列を作成
    arr_str = '(' + ', '.join('(' + ', '.join(str(val.round(3)) for val in vec) + ')' for vec in Sigma_beta_hat) + ')'
    dist_label  = f'$N = {i+1}, K = {K}, ' + '\\Sigma_{\\hat{\\beta}} = '+arr_str
    dist_label += f', \\Delta_{{\\hat{{\\beta}}}} = {dist_val:.2f}$'
    
    # マハラノビス距離を描画
    ax.contour(beta0_grid, beta1_grid, trace_dist_lt[i].reshape(beta0_grid.shape), offset=dist_min, 
               vmin=dist_min, vmax=dist_max, levels=dist_levels) # 等高線
    ax.scatter(*beta_true, dist_min, 
               color='red', s=100, 
               label=f'$\\beta = ({beta_true[0]}, {beta_true[1]})$') # 真のパラメータ
    ax.scatter(*beta_hat, dist_min, 
               facecolor='none', edgecolor='orange', s=100) # 推定パラメータ(座標平面上)
    ax.scatter(*beta_hat, dist_val, 
               color='orange', s=100, 
               label=f'$\\hat{{\\beta}} = ({beta_hat[0]:.2f}, {beta_hat[1]:.2f})$') # 推定パラメータ(曲面上)
    ax.plot([beta_true[0], beta_hat[0]], [beta_true[1], beta_hat[1]], [dist_min, dist_min], 
            color='black', linestyle='dotted', label='Euclid') # ユークリッド距離
    ax.plot([beta_hat[0], beta_hat[0]], [beta_hat[1], beta_hat[1]], [dist_min, dist_val], 
            color='black', linestyle='dashed', label='Mahalanobis') # マハラノビス距離
    ax.plot_surface(beta0_grid, beta1_grid, trace_dist_lt[i].reshape(grid_shape), 
                    cmap='viridis', vmin=dist_min, vmax=dist_max, alpha=0.8) # 曲面
    ax.set_xlim(xmin=beta0_min, xmax=beta0_max)
    ax.set_ylim(ymin=beta1_min, ymax=beta1_max)
    ax.set_zlim(zmin=dist_min, zmax=dist_max)
    ax.set_xlabel('$\\hat{\\beta}_0$')
    ax.set_ylabel('$\\hat{\\beta}_1$')
    ax.set_zlabel('$\\Delta$')
    ax.set_title(dist_label, loc='left')
    ax.legend(loc='upper left')
    #ax.set_box_aspect([1.0, beta1_size/beta0_size, 0.6]) # 高さ(横サイズに対する比)を指定

# 動画を作成
ani = FuncAnimation(fig=fig, func=update, frames=N-1, interval=100)

# 動画を書出
ani.save(
    filename='../../figure/ch3/1d_ols/1d_ols_sigma_beta_n_surface.mp4', 
    progress_callback = lambda i, n: print(f'frame: {i} / {n}')
)


# %%


