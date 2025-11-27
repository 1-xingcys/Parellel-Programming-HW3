# Parallel Programming HW3 - CUDA Ray Marching

## 專案簡介

本專案使用 CUDA 實作平行化的 Ray Marching 演算法，將原本循序執行的 CPU 程式轉換為 GPU 平行運算，大幅提升渲染效能。

## 實作特色

### 1. CUDA 平行化策略

- 採用 2D 資料平行化，每個 GPU thread 負責計算一個像素
- 使用 16×16 的 block 配置 (256 threads/block)
- 自動處理影像邊界，確保不同解析度都能正確執行

### 2. 效能優化技術

- **Constant Memory**：將唯讀參數 (相機位置、FOV 等) 儲存於 constant memory
- **單精度運算**：使用 `float` 取代 `double`，減少記憶體使用並提升 2-3× 效能
- **數學運算優化**：消除 `pow()` 呼叫，改用乘法展開
- **Fast Math**：使用 `-use_fast_math` 編譯旗標加速浮點運算
- **CUDA Intrinsic**：使用 `sincos()` 同時計算 sine 和 cosine

### 3. 核心實作

- 重新實作所有向量運算 (取代 GLM library)
- 使用 CUDA 內建型別 `float3`、`float2`
- 所有 device 函式加上 `__forceinline__` 確保效能

## 編譯與執行

### 編譯

```bash
make          # 編譯 CUDA 版本
make cpu      # 編譯 CPU 版本
```

### 執行

```bash
./hw3 <camera_x> <camera_y> <camera_z> <lookat_x> <lookat_y> <lookat_z> <width> <height> <output.png>
```

### 範例

```bash
# 測試案例 01
./hw3 4.152 2.398 -2.601 0 0 0 512 512 output_gpu/01.png

# 測試案例 08 (高負載)
./hw3 -1.2 -0.51 -0.8 -0.271 -0.299 -0.379 4096 4096 output_gpu/08.png
```

## 專案結構

```
hw3/
├── hw3.cu              # CUDA 實作
├── hw3_cpu.cpp         # CPU 參考實作
├── Makefile            # 編譯設定
├── report.pdf          # 詳細報告
├── testcases/          # 測試案例
├── output_gpu/         # GPU 輸出結果
└── lodepng/            # PNG 編碼/解碼函式庫
```
