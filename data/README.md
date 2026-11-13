# data — 数据集

> **本地这个目录是空的。** 数据集共约 6.9 GB，全部在服务器上，不下到本地。

## 数据在哪

| 位置 | 内容 | 大小 |
|---|---|---|
| `zjut148:/public2/czz/ZhangYu/project/mffgcn/data/cache/` | 训练直接读的 npy 缓存 | 3.5 GB |
| `zjut148:/public2/czz/ZhangYu/project/mffgcn/data/raw/RadioMapSeer/` | 原始 PNG（已解压） | 3.4 GB |
| `data/RadioMapSeer.zip` | 原始压缩包 | 3.0 GB |

⚠ **不要重新下载**，服务器上已经有了。

## cache 目录的 5 个文件

| 文件 | 形状 | dtype | 大小 | 内容 |
|---|---|---|---:|---|
| `gain_dpm.npy` | (56080, 256, 256) | uint8 | 3.5 GB | 增益图，按 (city, tx) 排序 |
| `buildings.npy` | (701, 256, 256) | uint8 | 46 MB | 建筑物二值图 |
| `buildings_city.npy` | (701,) | int16 | — | 每行对应的 city id |
| `tx_coords.npy` | (56080, 2) | int16 | — | 发射机位置 (行, 列) |
| `index.csv` | 56080 行 | — | — | idx, city_id, tx_id, gain_file |

**索引语义**：`idx` 是全局行号（0..56079）。因为按 (city, tx) 字典序排序，
所以 `city 0` 占 idx 0..79，`city 1` 占 80..159……

⚠ `tx_coords.npy` **必须用全局 idx 索引**。用 split 内的局部下标会让 val/test 的
发射机位置全部错乱 —— 这是本项目最严重的一个 bug，见 `代码导读.md` 第七节。

## 数据划分

按**城市**切分，不是按样本：

```
train  city 0..499      40000 样本
val    city 500..599     8000 样本
test   city 600..700     8080 样本
```

同一城市的 80 个发射机位置会被一起分到同一侧，保证测试城市完全没在训练时出现过。

## 原始数据的目录结构

```
RadioMapSeer/
    gain/gain/carsDPM/<city>_<tx>.png     增益图 56080 张
    png/buildings_complete/<city>.png     建筑物图 701 张
    png/antennas/<city>_<tx>.png          单发射机指示图（本实现不读）
    gain/antenna/<city>.json              每个城市的 80 个发射机坐标
```

## 重新生成缓存（一般不需要）

```bash
python src/01_data/prepare_data.py \
    --root data/raw/RadioMapSeer \
    --out data/cache
```

56080 张 PNG 全读一遍约 20-30 分钟。

## 数值预处理

原始 PNG 是 `[-127, -47] dB` 线性映射到 `0..255`。`dataset.py::preprocess_gain`
把它转成 `[0,1]`：

```
x = raw / 255
x[x < 0.2] = 0          ← 低于 -111 dB 视为无信号
x = (x - 0.2) / 0.8     ← 拉伸回 [0,1]
```
