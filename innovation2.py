import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.losses import Huber
import joblib
from scipy.interpolate import interp1d

# ====================== 1. 读取数据 ======================
file_path = r'C:\\Users\\20980\\Desktop\\新建文件夹\\1.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1', usecols=[0, 1])
df.dropna(inplace=True)

measured = df.iloc[:, 0].values
true = df.iloc[:, 1].values

# 去除异常值（基于残差）
def remove_outliers_pairwise(x, y, threshold=2.5):
    residual = x - y
    z = np.abs((residual - residual.mean()) / residual.std())
    mask = z < threshold
    return x[mask], y[mask]

measured, true = remove_outliers_pairwise(measured, true)
delta = true - measured  # 偏差

# ====================== 2. 数据归一化 ======================
scaler_input = StandardScaler()
scaler_output = StandardScaler()

measured_scaled = scaler_input.fit_transform(measured.reshape(-1, 1))
delta_scaled = scaler_output.fit_transform(delta.reshape(-1, 1))

# ====================== 3. 构建残差神经网络 ======================
def build_residual_model():
    inputs = keras.Input(shape=(1,))
    x = layers.BatchNormalization()(inputs)

    # 第一层
    x1 = layers.Dense(128, activation='swish', kernel_regularizer=regularizers.l2(0.01))(x)  # 增加正则化
    x1 = layers.Dropout(0.5)(x1)  # 增加 Dropout

    # 第二层（保持输出维度一致）
    x2 = layers.Dense(128, activation='swish', kernel_regularizer=regularizers.l2(0.01))(x1)  # 增加正则化
    x2 = layers.Dropout(0.4)(x2)  # 增加 Dropout

    # 残差连接
    x = layers.Add()([x1, x2])
    x = layers.Dense(64, activation='swish')(x)
    outputs = layers.Dense(1)(x)

    return keras.Model(inputs, outputs)

model = build_residual_model()
optimizer = keras.optimizers.Adam(learning_rate=1e-4)  # 调整学习率
model.compile(optimizer=optimizer, loss=Huber(), metrics=['mae'])

# ====================== 4. 模型训练 ======================
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=1e-6)
checkpoint = keras.callbacks.ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(
    measured_scaled, delta_scaled,
    epochs=1000, batch_size=64, validation_split=0.2,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ====================== 5. 电流修正函数 ======================
def correct_current(measured_val):
    measured_scaled = scaler_input.transform([[measured_val]])
    delta_scaled = model.predict(measured_scaled, verbose=0)
    delta = scaler_output.inverse_transform(delta_scaled)
    return measured_val + delta[0, 0]

# 修正全部数据
corrected_all = np.array([correct_current(m) for m in measured])
corrected_smoothed = savgol_filter(corrected_all, window_length=7, polyorder=2)

# ====================== 6. 可视化 ======================
plt.figure(figsize=(10, 6))

# 画出真实电流与测量电流的对比
plt.plot(true, true, label="True Current", color='black', linestyle="dashed")
plt.scatter(measured, true, label="Measured", color='red', alpha=0.5)

# 通过增加插值点，生成更多修正点
measured_fine = np.linspace(measured.min(), measured.max(), 500)  # 生成更多修正点
corrected_fine = np.array([correct_current(m) for m in measured_fine])
corrected_smoothed_fine = savgol_filter(corrected_fine, window_length=7, polyorder=2)

# 使用插值方法使得 true 数据与 measured_fine 对齐
interp_true = interp1d(measured, true, kind='linear', fill_value="extrapolate")
true_fine = interp_true(measured_fine)

# 绘制平滑后的修正点
plt.scatter(corrected_smoothed_fine, true_fine, label="Corrected (Smoothed, Fine)", color='blue', alpha=0.7, s=10)

# 设置图表的x轴和y轴范围，专注于电流高低区域
plt.xlim(9, 10)  # 假设电流在0到10A之间
plt.ylim(9, 10)

plt.xlabel("Input Current (A)")
plt.ylabel("Actual Current (A)")
plt.title("Ammeter Correction with Residual Learning")
plt.legend()
plt.grid(True)

# 显示图形
plt.show()

# ====================== 9. 训练过程可视化 ======================
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss (Huber)")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# ====================== 7. 测试与保存 ======================
test_values = [5.0, 6.0, 7.8, 8.4, 9.8]
for val in test_values:
    corrected = correct_current(val)
    print(f"Measured: {val:.2f} A -> Corrected: {corrected:.2f} A")

model.save("current_correction_model.h5")
joblib.dump(scaler_input, "scaler_input.pkl")
joblib.dump(scaler_output, "scaler_output.pkl")

# ====================== 8. 加载验证 ======================
model = keras.models.load_model("best_model.h5", compile=False)
scaler_input = joblib.load("scaler_input.pkl")
scaler_output = joblib.load("scaler_output.pkl")

measured_value = 9.8
corrected_scaled = model.predict(scaler_input.transform([[measured_value]]))
delta = scaler_output.inverse_transform(corrected_scaled)
corrected_value = measured_value + delta[0, 0]
print(f"[Reloaded] Measured: {measured_value:.2f} A -> Corrected: {corrected_value:.2f} A")
