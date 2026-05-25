# Corrected Paper-Style Rerun 实验需求单

## 背景

本项目是 Christensen, Siggaard & Veliyev (JFEC 2023) "A Machine Learning Approach to
Volatility Forecasting" 的论文复现。当前主线已完成 25 ticker × 全模型 × h=1/h=5 × 50
seeds 的训练（输出在 `outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/`）。

在分析复现结果时，识别出两个潜在的方法学偏离：

1. **Dropout 约定歧义**
   - 论文 p.1721 Appendix A.5 写 "dropout rate = 0.8 following Goodfellow et al. (2016)"
   - Goodfellow 教科书使用 keep probability 约定：0.8 = 保留 80%（丢弃 20%）
   - 当前实现 [src/rv1rep/nn.py:30](src/rv1rep/nn.py#L30) 使用 `tf.keras.layers.Dropout(rate=0.8)`，
     在 TF 2.x 约定下 rate = drop probability，即丢弃 80%
   - 论文意图很可能是 drop=0.2，当前实现是 drop=0.8，差 4 倍 over-regularization

2. **Y 标准化**
   - 论文 p.1693 正文只说 "we standardize the input data"（X 标准化）
   - 但论文 Table A.5 (p.1720) 注解写 "All variables are standardized"，且 Apple HAR-X 系数
     量级（RVD=0.185, RVW=0.415）暗示 y 也标准化
   - 论文 Figure 6 y 轴范围 [-0.5, 0.5] 进一步暗示 y 标准化
   - 当前实现 [src/rv1rep/nn.py:60](src/rv1rep/nn.py#L60) 直接传 raw y 给 `model.fit()`

### 已有诊断证据

- **Y 标准化诊断**（3 次，在 dropout=0.8 下）：
  - AAPL/PARTIAL_MALL/h=1/NN2：std-y test MSE 是 raw-y 的 2.26 倍
  - INTC/MHAR/h=1/NN3：1.88 倍
  - WMT/PARTIAL_MALL/h=1/NN3：1.02 倍
  - 结论：在 dropout=0.8（drop 80%）下，单独标准化 y 不改善 NN

- **Dropout 诊断**（已有部分测试）：dropout=0.2 显著改善 NN test MSE

**当前结论**：最需要先回答的问题不是单因素贡献分解，而是：如果按最接近原论文的 combined corrected specification 重跑，
模型相对 HAR 的 out-of-sample forecast accuracy 是否明显改善。单因素对照可以作为后续诊断，但不应先于主实验。

---

## 实验目的

主实验是一次 **combined corrected paper-style rerun**，优先覆盖 **h=1 和 h=5**，目标是生成与原论文 Table 2 风格一致的 test-period forecast evaluation 表：

- 在同一套修正后设定下，比较所有模型相对 HAR 的 h=1 与 h=5 relative MSE。
- 对应生成 pairwise relative MSE matrix 和 Diebold-Mariano tests。
- 重点判断 NN、RF、GB、regularized linear 等模型在修正后是否相对 HAR 明显改善。
- 不把旧主线结果作为主要 baseline；主比较基准是本次 corrected rerun 内部的 HAR。

`std_y_only` 和 `drop_fix_only` 单因素对照只作为后续可选诊断。如果 combined corrected rerun 已能解释主要差异，则不强制运行完整单因素分解。

---

## 项目环境

| 项 | 值 |
|---|---|
| 项目根 | `/Users/celiawong/Desktop/Course004 machine learning/004_individual/rv1_professional_reproduction_code/` |
| Python | `~/.pyenv/versions/3.11.8/bin/python` |
| 默认配置 | `config/default.yaml`（当前已修正为论文式 validation-tuned no-refit 默认） |
| 最终主线结果配置 | `config/paper_core_rolling_tuned_no_refit.yaml`；GB 使用 `config/paper_core_rolling_gb_tuned_no_refit_40grid.yaml` |
| 数据 | `data/processed/forecasting_panel.csv` |
| 旧 NN50 参考 checkpoints | `outputs_nn50_checkpointed_20260521/nn_seed_predictions/<dataset>/h<horizon>/<NN>/<ticker>/seed_*.csv`（主实验不复用） |
| 旧主线参考结果 | `outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/predictions/model_predictions.csv`（主实验不作为 baseline） |
| 硬件 | M4, 10 核（4 性能 + 6 效率）, 16GB RAM |

**已实现的关键代码**：
- [src/rv1rep/forecasting.py](src/rv1rep/forecasting.py)：rolling 和 fixed scheme 实现，连续 X 标准化模板
- [src/rv1rep/nn.py](src/rv1rep/nn.py)：NN ensemble 训练，TensorFlow backend
- [src/rv1rep/models.py](src/rv1rep/models.py)：包含 `_refit_tuned_models_on_train_validation` 开关，当前默认 `False`，即 validation-tuned 模型按论文式 train-only after validation selection 处理
- [scripts/04_run_nn_checkpoints.py](scripts/04_run_nn_checkpoints.py)：NN per-seed checkpoint runner
- [scripts/11_nn_target_scaling_diagnostic.py](scripts/11_nn_target_scaling_diagnostic.py)：单组合 y 标准化诊断（可参考结构）

---

## 实验设计

### 主实验：combined corrected paper-style specification

**核心设定**：

| 项 | 设定 |
|---|---|
| Horizon | 优先 h=1 和 h=5；h=1 对齐原论文 Table 2 / Figure 3 / Figure 4 / Figure 5 的核心设定，h=5 对应 one-week-ahead 扩展表 |
| Tickers | 优先 AAPL, JPM, MSFT（3 个 ticker）；smoke test 可临时用 1 个 ticker；全量 25 ticker 不是本阶段目标 |
| Datasets | MHAR, PARTIAL_MALL |
| Forecast scheme | 非 NN rolling；NN fixed-window |
| X 标准化 | 只用当前 training window / training split 估计；连续变量标准化，`ea` 保持 0/1 |
| Y target | 非 LogHAR 模型使用 train-only y 标准化训练，预测后反标准化回 raw RV；LogHAR 保持 log-target baseline |
| NN dropout | `tf.keras.layers.Dropout(rate=0.2)`，对应论文/Goodfellow 语境下 keep probability 0.8 |
| NN seeds | 50 seeds；输出 single-best seed 和 top-10 ensemble |
| Validation-tuned models | train fit + validation tuning；selection 后不做 train+validation final refit |
| Tree refit cadence | Bagging/RF/GB rolling refit cadence 按当前论文式配置；GB 使用 40-grid no-refit |

**模型范围**：

- HAR family：HAR, HAR-X, LogHAR, LevHAR, SHAR, HARQ
- Regularized linear：Ridge, Lasso, ElasticNet, AdaptiveLasso, PostLasso
- Trees：Bagging, RandomForest, GradientBoosting
- Neural networks：NN1, NN2, NN3, NN4；每个架构输出 `^1` 和 `^10` 两种表格列

**估计样本规则**：

- HAR family、Bagging、RandomForest 这类不需要 validation tuning 的模型，把 rolling train+validation window 作为完整 in-sample fit block，因此表现为 `n_val=0`；这不包含 test，不构成 look-ahead。
- Ridge/Lasso/ElasticNet/AdaptiveLasso/PostLasso/GradientBoosting 使用 train fit + validation selection，且保持 no-refit。
- NN 使用 fixed train/validation/test split；validation 只用于 seed/model selection 和 top-10 ensemble selection。

### 可选诊断：单因素 sensitivity checks

主实验完成后，如果需要进一步解释差异来源，再考虑在小样本上补跑：

| Optional diagnostic | Dropout | Y target | 用途 |
|---|---|---|---|
| `std_y_only` | 0.8 | standardized | 只看 y 标准化在旧 dropout 下的影响 |
| `drop_fix_only` | 0.2 | raw | 只看 dropout 修正在 raw y 下的影响 |
| `combined` | 0.2 | standardized | 主实验设定；若已全量完成，不需要重复小样本 |

这些诊断不是主表的必要条件；主表只依赖 combined corrected paper-style rerun。

### 主比较基准

主实验不复用旧主线 predictions 作为 baseline。所有 relative MSE、DM tests 和 paper-style table 都基于本次 corrected rerun 的 test-period predictions：

- 表格中的 benchmark row/column 是本次 corrected rerun 内部、同一 horizon 和 dataset 下的 HAR。
- 旧主线结果只作为背景参考，不参与主表计算。

### X 标准化实现规范

- 只对连续解释变量做训练窗口标准化；标准化参数只能由当前 training window 估计，再应用到 validation/test。
- 类别变量、dummy 变量、indicator 变量不做标准化，保留原始编码。
- 当前项目中已知的类别/指示变量是 `ea`（earnings-announcement indicator, 0/1）。如果后续加入新的类别变量，也必须加入显式排除列表。
- 不要用“唯一值数量少”自动判断类别变量，避免误排除离散但连续含义的变量；使用显式列表，例如 `CATEGORICAL_FEATURES = {'ea'}`。
- 输出 metadata 中记录 `x_standardization = 'continuous_only_train_window'`，并记录 `categorical_features_not_standardized = ['ea']`（若该模型/数据集实际使用了 `ea`）。

示意实现：

```python
CATEGORICAL_FEATURES = {'ea'}

cat_cols = [c for c in feature_cols if c in CATEGORICAL_FEATURES]
cont_cols = [c for c in feature_cols if c not in CATEGORICAL_FEATURES]

scaler = Standardizer().fit(train[cont_cols])
X_train_cont = scaler.transform(train[cont_cols])
X_val_cont = scaler.transform(val[cont_cols])
X_test_cont = scaler.transform(test[cont_cols])

X_train = pd.concat([X_train_cont.reset_index(drop=True), train[cat_cols].reset_index(drop=True)], axis=1)[feature_cols]
X_val = pd.concat([X_val_cont.reset_index(drop=True), val[cat_cols].reset_index(drop=True)], axis=1)[feature_cols]
X_test = pd.concat([X_test_cont.reset_index(drop=True), test[cat_cols].reset_index(drop=True)], axis=1)[feature_cols]
```

### Y 标准化实现规范

```python
# 训练时
y_mean = float(y_train.mean())
y_std = float(y_train.std(ddof=1))
if not np.isfinite(y_std) or y_std <= 0:
    y_std = 1.0

y_train_std = (y_train - y_mean) / y_std
y_val_std = (y_val - y_mean) / y_std

# 训练模型（用标准化的 y）
model.fit(X_train, y_train_std, ...)

# 预测时反标准化
pred_std = model.predict(X_test)
pred_raw = pred_std * y_std + y_mean

# 后续 forecast_rv、MSE、DM、relative MSE、summary table 全部使用 pred_raw。
# 禁止直接用 pred_std 与 raw actual_rv 做评估。

# 应用现有 positivity + insanity filter（用 raw y 的 in_sample_min/mean）
in_sample_rv = pd.concat([train['rv'], val['rv']]).dropna()
in_min = float(in_sample_rv.min())
in_mean = float(in_sample_rv.mean())
pred = enforce_positive_forecasts(pred_raw, in_min, cfg['estimation']['negative_forecast_policy'])
if cfg['estimation']['insanity_filter']['enabled']:
    pred = insanity_filter(pred, in_mean, in_min, cfg['estimation']['insanity_filter']['max_multiple_of_in_sample_mean'])
```

**Forecast evaluation 与 Figure 6 / ALE 的尺度区别**：
- Forecast evaluation 输出必须回到 raw RV 尺度。也就是说，`forecast_rv`、MSE、DM、relative MSE、summary table 都必须使用反标准化后的 `pred_raw`。
- Figure 6 / ALE 是解释图，不是 forecast evaluation 表格。如果后续使用 standardized-y 模型重画 ALE，ALE 的纵轴可以保留 standardized-y model-output scale，不强制反标准化。
- 因此，预测 CSV 与 corrected evaluation 表使用 raw-scale `forecast_rv`；ALE 数据表或图片如果使用 `pred_std` 尺度，必须在文件名、metadata、图题或 notes 中明确标注，避免和 raw RV 尺度结果混用。

**对 LogHAR 的特殊处理**：
- LogHAR 的 y 已经是对应 horizon 的 `target_log_rv_h<h>`（log 变换）
- 如果再标准化，需要在 log space 标准化
- 反标准化时先反 log（exp + Jensen 修正），再反标准化（× y_std + y_mean）
- 或者只在 NN 上做 y 标准化，LogHAR 保持现状（推荐，因为 LogHAR 本身就是 log 变换处理 y）

**推荐**：LogHAR 在非 NN 标准化对照中**保持原状**（不在 log space 再次标准化），其他 13 个非 NN 模型做标准化。

**非 NN `target_scaling` 字段规范**：
- 非 LogHAR 模型：`target_scaling = 'standardized_y_train_inverse_transformed'`
- LogHAR：`target_scaling = 'log_target_baseline_no_extra_standardization'`

---

## 输出结构

所有输出写入新目录，不覆盖主线结果：

```
outputs_corrected_paper_style_h1h5_<YYYYMMDD>/
├── nn_seed_predictions/
│   └── combined/
│       └── <dataset>/h<horizon>/<NN>/<ticker>/seed_*.csv
├── nn_aggregated/
│   └── combined_nn_ensembles.csv       # NN^1 和 NN^10
├── predictions/
│   ├── nonnn_model_predictions.csv
│   ├── model_predictions.csv           # 非 NN + NN combined corrected predictions
│   └── by_model/
├── evaluation/
│   ├── model_mse_summary.csv
│   ├── h1_pairwise_relative_mse_matrix.csv
│   ├── h1_diebold_mariano_tests.csv
│   ├── h1_table2_style_relative_mse_dm.docx
│   ├── h5_pairwise_relative_mse_matrix.csv
│   ├── h5_diebold_mariano_tests.csv
│   └── h5_table2_style_relative_mse_dm.docx
├── optional_diagnostics/
│   └── README.md                       # 若后续补跑单因素诊断，再写到这里
├── logs/
│   ├── nn_combined_<worker_id>.log
│   ├── nonnn_combined.log
│   └── build_tables.log
├── run_provenance.json                 # 运行参数、git commit、环境信息
└── README.md                            # 实验描述 + 结果摘要
```

---

## 实现要求

### 1. 新建脚本

#### `scripts/13_run_corrected_nn_combined.py`（NN 部分）

参考现有 [scripts/11_nn_target_scaling_diagnostic.py](scripts/11_nn_target_scaling_diagnostic.py) 的实现模式，但扩展为：

- 接受 CLI 参数：`--tickers`, `--datasets`, `--horizons`, `--archs`, `--output-dir`, `--workers`, `--seed-count`
- 接受 `--config`，默认 `config/default.yaml`。该默认配置已经修正为论文式 no-refit；脚本只能在内存中覆盖 dropout / target scaling，不得修改配置文件本身。
- 内部用 `multiprocessing.Pool` 并行 N 个 worker
- 每个任务是一个 (ticker, dataset, horizon, arch) 组合的 50 seeds 训练
- 每 seed 写独立 checkpoint CSV（per-seed reuse 逻辑）
- 每个架构输出 single-best seed (`NNk^1`) 和 top-10 ensemble (`NNk^10`)
- 跑完后聚合到 `nn_aggregated/combined_nn_ensembles.csv`
- X 标准化必须遵守“连续变量标准化、类别变量不标准化”的规则；`ea` 保持 0/1 原始编码
- 使用 `dropout = 0.2`
- 使用 y 标准化训练，并把 test predictions 反标准化回 raw RV 后写入 `forecast_rv`

**Combined 实现**：

```python
def train_combined(prepared, arch, seeds, output_dir):
    dropout = 0.2
    standardize_y = True
    for seed in seeds:
        train_one_seed(prepared, arch, dropout, standardize_y, seed, output_dir)
```

**Task priority（重任务优先）**：

```python
def task_priority(task):
    """重任务优先，避免 tail latency"""
    arch_weight = {'NN4': 0, 'NN3': 1, 'NN2': 2, 'NN1': 3}
    dataset_weight = {'PARTIAL_MALL': 0, 'MHAR': 1}
    return (
        arch_weight[task['arch']],
        dataset_weight[task['dataset']],
    )

tasks_sorted = sorted(tasks, key=task_priority)
```

#### `scripts/14_run_corrected_nonnn_combined.py`（非 NN 部分）

- 对 14 个非 NN 模型在 (ticker, dataset, horizon) 上跑 combined corrected specification，优先 horizons = 1, 5；其中 13 个非 LogHAR 模型使用 y 标准化训练并反变换预测，LogHAR 保持 log-target baseline 规格
- 接受 `--config`，默认 `config/default.yaml`，并保持 `refit_tuned_models_on_train_validation: false`
- 复用现有 [src/rv1rep/forecasting.py](src/rv1rep/forecasting.py) 的 rolling 实现
- 对非 LogHAR 模型，在调用 `fit_sklearn_model` 之前对 y 做标准化，预测后反标准化
- X 标准化必须遵守“连续变量标准化、类别变量不标准化”的规则；`ea` 保持 0/1 原始编码
- LogHAR 保持现状（不重复标准化），并在输出中标记 `target_scaling = 'log_target_baseline_no_extra_standardization'`
- 其他 13 个非 NN 模型输出 `target_scaling = 'standardized_y_train_inverse_transformed'`
- 输出 `outputs_corrected_paper_style_h1h5_<date>/predictions/nonnn_model_predictions.csv`

#### `scripts/15_build_corrected_paper_tables.py`（结果汇总）

- 读取 NN combined 输出 + 非 NN combined 输出
- 合并生成 `predictions/model_predictions.csv`
- 基于 test-period `actual_rv` 和 raw-scale `forecast_rv` 计算 MSE
- 生成 model MSE summary、pairwise relative MSE matrix、DM tests
- 分别为 h=1 和 h=5 生成 paper-style Word 表：`evaluation/h1_table2_style_relative_mse_dm.docx`、`evaluation/h5_table2_style_relative_mse_dm.docx`
- 表格主比较使用本次 corrected rerun 内部 HAR，不读取旧主线 baseline 作为主表基准

### 2. 关键设计约束

- **不覆盖现有数据**：所有输出写入新目录 `outputs_corrected_paper_style_h1h5_<date>/`
- **不修改主线配置**：`config/default.yaml`、`config/paper_core_rolling_tuned_no_refit.yaml`、`config/paper_core_rolling_gb_tuned_no_refit_40grid.yaml` 均不要改；corrected rerun 脚本只读取这些配置并在内存中覆盖 dropout / target scaling
- **保持论文式 tuned-model no-refit**：任何 validation-tuned 非 NN 模型必须保持 `refit_tuned_models_on_train_validation: false`；不要回到旧的 train+validation final refit 逻辑
- **X 标准化只作用于连续变量**：类别/指示变量不标准化；当前显式排除 `ea`
- **主表不使用旧 baseline**：主比较基准是本次 corrected rerun 内部的 HAR
- **per-seed checkpoint**：NN 每 seed 写独立 CSV，支持断点续跑
- **原子写盘**：先写 `.tmp` 文件，再 `rename`，避免半成品被误判为完成
- **不打扰其他进程**：如果当前有后台任务在跑，不冲突（独立 output dir）

---

## 并行执行方案

### 总任务规模

| 部分 | 任务数 | Seed 数 | 完成判据 |
|---|---|---|---|
| NN combined corrected | 3 ticker × 2 dataset × 2 horizon × 4 arch | 上一列 × 50 seeds = 2,400 seeds | Phase 1 验证检查点全部通过 |
| 非 NN combined corrected | 3 ticker × 2 dataset × 2 horizon × 14 model | n/a | Phase 2 验证检查点全部通过 |
| Paper-style evaluation | 3 ticker × 2 dataset × 2 horizon × 22 model labels | n/a | Phase 3 验证检查点全部通过 |

本阶段优先使用 3 个代表 ticker：AAPL, JPM, MSFT。smoke test 可以临时用 1 个 ticker，但最终本阶段 paper-style table 至少应覆盖这 3 个 ticker、h=1 和 h=5。

### Worker 分配

**M4 + 16GB 配置（3 workers）**：

| 优化项 | 设定 |
|---|---|
| Workers | **3** |
| 每 worker OMP threads | 2 |
| 总线程数 | 6（4 性能 + 2 效率，避免颠簸）|
| 内存预算 | 每 worker ~2-2.5 GB，3 workers ~7.5 GB |

### 启动脚本模板

```bash
#!/bin/bash
# launch_corrected_paper_style_h1h5.sh

set -e

OUTPUT_DIR="outputs_corrected_paper_style_h1h5_$(date +%Y%m%d)"
mkdir -p "${OUTPUT_DIR}/logs"

# === 1. 关闭后台干扰 ===
sudo mdutil -a -i off    # 关 Spotlight
sudo tmutil disable      # 关 Time Machine

# === 2. 环境变量限制底层线程 ===
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TF_INTRA_OP_PARALLELISM_THREADS=2
export TF_INTER_OP_PARALLELISM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3

# === 3. 防止 Mac 睡眠 ===
caffeinate -di &
CAFFEINATE_PID=$!
echo "${CAFFEINATE_PID}" > "${OUTPUT_DIR}/.caffeinate_pid"

# === 4. 启动 NN combined corrected run（screen 后台，taskpolicy 绑性能核）===
screen -dmS nn_corrected_h1h5 bash -lc "
  taskpolicy -c utility ~/.pyenv/versions/3.11.8/bin/python scripts/13_run_corrected_nn_combined.py \
    --tickers AAPL JPM MSFT \
    --datasets MHAR PARTIAL_MALL \
    --horizons 1 5 \
    --archs NN1 NN2 NN3 NN4 \
    --seed-count 50 \
    --workers 3 \
    --output-dir ${OUTPUT_DIR} \
    > ${OUTPUT_DIR}/logs/nn_combined.log 2>&1
"

echo "NN combined corrected run started in screen 'nn_corrected_h1h5'."
echo "Monitor: tail -f ${OUTPUT_DIR}/logs/nn_combined.log"
echo "Status:  screen -r nn_corrected_h1h5"
echo ""
echo "Non-NN combined corrected run starts separately AFTER the Phase 1 verification checkpoint passes:"
echo "  ~/.pyenv/versions/3.11.8/bin/python scripts/14_run_corrected_nonnn_combined.py \\"
echo "    --tickers AAPL JPM MSFT --datasets MHAR PARTIAL_MALL --horizons 1 5 \\"
echo "    --output-dir ${OUTPUT_DIR} --workers 3"
echo ""
echo "After all done, build tables:"
echo "  ~/.pyenv/versions/3.11.8/bin/python scripts/15_build_corrected_paper_tables.py \\"
echo "    --output-dir ${OUTPUT_DIR}"
echo ""
echo "Cleanup when finished:"
echo "  kill ${CAFFEINATE_PID}"
echo "  sudo mdutil -a -i on"
echo "  sudo tmutil enable"
```

### 执行顺序

**Phase 1 - NN combined corrected run**：

```bash
bash launch_corrected_paper_style_h1h5.sh
# Do not stop early. Continue until the Phase 1 verification checkpoint passes.
```

**Phase 2 - 非 NN combined corrected run**：

```bash
~/.pyenv/versions/3.11.8/bin/python scripts/14_run_corrected_nonnn_combined.py \
  --tickers AAPL JPM MSFT \
  --datasets MHAR PARTIAL_MALL \
  --horizons 1 5 \
  --output-dir outputs_corrected_paper_style_h1h5_<date> \
  --workers 3
```

**Phase 3 - paper-style 表格**：

```bash
~/.pyenv/versions/3.11.8/bin/python scripts/15_build_corrected_paper_tables.py \
  --output-dir outputs_corrected_paper_style_h1h5_<date>
```

### 推进标准

不要按主观感觉或日志沉默判断任务是否完成。每个 phase 只能在对应验证检查点通过后进入下一步：

- Phase 1：所有 NN seed checkpoint 和 top-10 aggregation 检查通过。
- Phase 2：非 NN corrected predictions 覆盖 3 ticker × 2 dataset × 2 horizon × 14 model 组合。
- Phase 3：`evaluation/` 下 MSE summary、pairwise relative MSE、DM tests 和 Word 表全部生成并通过基本完整性检查。

---

## 验证检查点

### Phase 1 完成后

```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import pandas as pd
from pathlib import Path

out_dir = Path('outputs_corrected_paper_style_h1h5_<date>')
expected_tickers = 3
expected_horizons = 2

# 1. 检查 seed checkpoint 完整性
seed_dir = out_dir / 'nn_seed_predictions' / 'combined'
expected = expected_tickers * 2 * 4 * 50
expected *= expected_horizons
actual = sum(1 for _ in seed_dir.rglob('seed_*.csv'))
print(f'combined NN seeds: {actual}/{expected}')
assert actual == expected, 'Missing combined NN seed checkpoints'

# 2. 检查 NN^1 和 NN^10 aggregation
agg = pd.read_csv(out_dir / 'nn_aggregated' / 'combined_nn_ensembles.csv')
expected_combos = expected_tickers * 2 * expected_horizons * 8  # 4 arch × {single best, top10}
actual_combos = agg.groupby(['ticker', 'dataset', 'horizon', 'model']).ngroups
print(f'combined NN aggregated: {actual_combos}/{expected_combos}')
assert actual_combos == expected_combos

print('Phase 1 OK.')
"
```

### Phase 2 完成后

```bash
~/.pyenv/versions/3.11.8/bin/python -c "
import pandas as pd

out_dir = 'outputs_corrected_paper_style_h1h5_<date>'
expected_tickers = 3
expected_horizons = 2
nonnn = pd.read_csv(f'{out_dir}/predictions/nonnn_model_predictions.csv')

expected_models = {
    'HAR', 'HARX', 'LogHAR', 'LevHAR', 'SHAR', 'HARQ',
    'Ridge', 'Lasso', 'ElasticNet', 'AdaptiveLasso', 'PostLasso',
    'Bagging', 'RandomForest', 'GradientBoosting',
}
seen = set(nonnn['model'])
missing = expected_models - seen
assert not missing, f'Missing non-NN models: {missing}'

expected_combos = expected_tickers * 2 * expected_horizons * 14
actual_combos = nonnn.groupby(['ticker', 'dataset', 'horizon', 'model']).ngroups
assert actual_combos == expected_combos, f'Expected {expected_combos}, got {actual_combos}'

print('Phase 2 OK.')
"
```

### Phase 3 完成后

`evaluation/` 应包含：

- `model_mse_summary.csv`
- `h1_pairwise_relative_mse_matrix.csv`
- `h1_diebold_mariano_tests.csv`
- `h1_table2_style_relative_mse_dm.docx`
- `h5_pairwise_relative_mse_matrix.csv`
- `h5_diebold_mariano_tests.csv`
- `h5_table2_style_relative_mse_dm.docx`
- 可选：`summary.md`，说明每个 dataset 下哪些模型相对 HAR 改善/恶化

---

## 风险与预案

| 风险 | 概率 | 预案 |
|---|---|---|
| Worker OOM | 低（3 workers + 16GB）| 重启时改 2 workers，已完成 seed checkpoint 自动复用 |
| 单 seed 训练无响应 | 低 | early stopping patience=100 保护；必要时中断后按 checkpoint 续跑 |
| 主机意外断电 | 低-中 | per-seed checkpoint 重启可恢复 |
| 运行进度慢于预期 | 中 | 不提前停止；以验证检查点和 checkpoint 完整性为准 |
| LogHAR y 标准化逻辑出错 | 中 | LogHAR 保持现状（不再标准化），有明确规范 |
| 散热降频 | 中 | 笔记本放硬质平面；接受 10-20% 性能损失 |

### 如果 worker crash

```bash
# 检查哪些 seed 已完成
~/.pyenv/versions/3.11.8/bin/python -c "
import pandas as pd
from pathlib import Path
out_dir = Path('outputs_corrected_paper_style_h1h5_<date>')
n = sum(1 for _ in (out_dir / 'nn_seed_predictions' / 'combined').rglob('seed_*.csv'))
print(f'combined NN seeds completed: {n}')
"

# 直接重启同样命令，已完成的会自动 skip
bash launch_corrected_paper_style_h1h5.sh
```

---

## 验收标准

实验完成后，应满足以下条件才能算"实验成功"：

1. **完整性**：
   - NN seed checkpoint 总数 = 3 ticker × 2 dataset × 2 horizon × 4 arch × 50 seeds = 2,400
   - NN aggregation 覆盖 3 ticker × 2 dataset × 2 horizon × 8 NN model labels（NN1^1, NN1^10, ..., NN4^1, NN4^10）
   - 非 NN 预测覆盖 3 ticker × 2 dataset × 2 horizon × 14 model
   - 合并后的 `predictions/model_predictions.csv` 覆盖 3 ticker × 2 dataset × 2 horizon × 22 model labels

2. **正确性**：
   - 每个 NN ensemble CSV 的 `dropout` 字段为 `0.2`
   - 每个 NN ensemble CSV 的 `target_scaling` 字段为 standardized-y train scaling with inverse-transformed forecasts
   - X 标准化只应用于连续变量；`ea` 等类别/指示变量在训练、验证和测试矩阵中保持原始 0/1 编码
   - 输出 metadata 中应包含 `x_standardization = 'continuous_only_train_window'`；使用 `ea` 的组合还应记录 `categorical_features_not_standardized = ['ea']`
   - 所有 standardized-y 模型的 `forecast_rv` 必须是反标准化后的 raw RV 尺度预测；禁止把标准化单位下的 `pred_std` 写入 `forecast_rv`
   - 所有 MSE、DM、relative MSE、summary table 必须使用 raw `actual_rv` 与 raw-scale `forecast_rv` 计算
   - 如果生成 Figure 6 / ALE，ALE 可以保留 standardized-y model-output scale，但必须在输出文件或 notes 中明确标注；不得与 forecast evaluation 的 raw-scale `forecast_rv` 混用
   - 非 NN 预测 CSV 中，除 LogHAR 外的模型均为 `target_scaling = 'standardized_y_train_inverse_transformed'`
   - 非 NN 预测 CSV 中，LogHAR 为 `target_scaling = 'log_target_baseline_no_extra_standardization'`
   - 主表中的 relative MSE benchmark 必须来自本次 corrected rerun 的 HAR，而不是旧主线 HAR

3. **无主线数据污染**：
   - 主线目录 `outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/` 未被修改
   - 已有 NN50 checkpoints `outputs_nn50_checkpointed_20260521/` 未被修改
   - `data/processed/forecasting_panel.csv` 未被修改
   - `config/default.yaml`、`config/paper_core_rolling_tuned_no_refit.yaml`、`config/paper_core_rolling_gb_tuned_no_refit_40grid.yaml` 未被修改

4. **可追溯**：
   - `run_provenance.json` 记录：实际启动时间、完成时间、git commit hash、Python/TF 版本、CLI 参数
   - 每个 seed CSV 的 `params` 字段记录：specification, dropout, target_scaling, y_mean_train, y_std_train, x_standardization, categorical_features_not_standardized

5. **可分析**：
   - `evaluation/h1_pairwise_relative_mse_matrix.csv` 和 `evaluation/h5_pairwise_relative_mse_matrix.csv` 可直接生成 Table 2-style Word 表
   - `evaluation/h1_diebold_mariano_tests.csv` 和 `evaluation/h5_diebold_mariano_tests.csv` 支持 10% / 5% / 1% 显著性格式
   - `evaluation/model_mse_summary.csv` 能直接查看每个模型相对 HAR 的改善或恶化

---

## 报告产出（实验完成后由人工写）

实验完成后，用 `evaluation/summary.md`、`model_mse_summary.csv`、`h1_pairwise_relative_mse_matrix.csv`、`h5_pairwise_relative_mse_matrix.csv`、`h1_diebold_mariano_tests.csv` 和 `h5_diebold_mariano_tests.csv` 的数据写入项目的最终报告 Section 12（或类似 methodology 章节）：

```markdown
## Section 12.X: Corrected Paper-Style Sensitivity Rerun

We identify two potential ambiguities in the paper's NN training description:
1. Dropout convention (keep vs drop probability)
2. Y target standardization

We therefore run a corrected paper-style specification for h=1 and h=5. Continuous
predictors are standardized using only the current training window, the EA indicator is
kept in its original 0/1 form, non-LogHAR targets are standardized on the training
sample and inverse-transformed before forecast evaluation, and neural networks use
drop=0.2 in TensorFlow/Keras so that the retained-unit probability is 0.8. The NN
experiments use 50 random seeds and report both single-best and top-10 ensemble
forecasts. Validation-tuned models retain the paper-style no-refit treatment after
hyperparameter selection.

[填入数字 + 解读]

The resulting Table 2-style relative MSE and Diebold-Mariano tests compare all models
against the HAR benchmark within the same corrected rerun and within the same horizon,
rather than against the previous mainline output. This makes the comparison internally
consistent and isolates whether the paper-style corrected specification materially
changes the ranking of NN, tree, and regularized linear forecasts relative to HAR.

Optional single-factor diagnostics (`std_y_only` and `drop_fix_only`) can be reported
separately if needed, but they are not required for the main corrected paper-style table.
```

---

## 关键文件清单（实施前必读）

| 文件 | 用途 |
|---|---|
| `src/rv1rep/nn.py` | NN ensemble 训练，参考 `_build_keras_model()` 和 `fit_nn_ensemble()` |
| `src/rv1rep/forecasting.py` | rolling 和 fixed 实现，X 标准化模板 |
| `src/rv1rep/models.py` | `fit_sklearn_model()` 非 NN 训练入口 |
| `src/rv1rep/preprocessing.py` | `Standardizer` 类 + `enforce_positive_forecasts` + `insanity_filter` |
| `src/rv1rep/features.py` | `feature_columns_for_model()` 决定每个模型用哪些特征 |
| `scripts/04_run_nn_checkpoints.py` | NN per-seed checkpoint runner，参考 reuse 逻辑 |
| `scripts/11_nn_target_scaling_diagnostic.py` | 单组合 y 标准化诊断，参考实现结构 |
| `config/default.yaml` | 当前默认配置，已修正为论文式 validation-tuned no-refit（**不要修改**）|
| `config/paper_core_rolling_tuned_no_refit.yaml` | 最终主线 regularized no-refit 配置（**不要修改**）|
| `config/paper_core_rolling_gb_tuned_no_refit_40grid.yaml` | 最终主线 GB 40-grid no-refit 配置（**不要修改**）|

---

## 实施 Checklist

实施 AI 在开工前确认：

- [ ] 已阅读 `src/rv1rep/nn.py` 理解 `_build_keras_model()` 的 dropout 参数位置
- [ ] 已阅读 `scripts/11_nn_target_scaling_diagnostic.py` 理解 y 标准化的实现模式
- [ ] 已阅读 `src/rv1rep/forecasting.py` 理解 X 标准化和 train/val/test 切分
- [ ] 已确认 X 标准化 helper 使用显式类别变量排除表，至少包含 `ea`
- [ ] 已确认 `data/processed/forecasting_panel.csv` 包含 `target_rv_h1`、`target_log_rv_h1`、`target_rv_h5`、`target_log_rv_h5` 列
- [ ] 已确认最终主实验使用 AAPL, JPM, MSFT 三个 ticker；smoke test 可以用 1 个 ticker，但不能替代最终 paper-style table
- [ ] 已写 `scripts/13_run_corrected_nn_combined.py`（NN combined corrected 部分）
- [ ] 已写 `scripts/14_run_corrected_nonnn_combined.py`（非 NN combined corrected 部分）
- [ ] 已写 `scripts/15_build_corrected_paper_tables.py`（paper-style evaluation 表格）
- [ ] 已写 `launch_corrected_paper_style_h1h5.sh` 启动脚本
- [ ] 已跑 smoke test（1 ticker × 1 arch × 5 seeds × 1 worker）验证 pipeline 正确
- [ ] 已确认 smoke test 输出的 `forecast_rv` 是 raw RV 尺度，且 `ea` 没有被标准化

---

## 联系信息

如实施过程中发现需求歧义或技术阻塞，**不要擅自修改**主线代码或配置。先标记问题，写到
`ABLATION_OPEN_QUESTIONS.md`，由人工评审后决定。

特别注意：
- LogHAR 在非 NN 部分的处理（推荐：不重复标准化）
- `_refit_tuned_models_on_train_validation` 开关在 corrected rerun 中保持 `False` / 默认 no-refit（与当前最终主线和论文式处理一致）；不要使用旧的 train+validation final refit 逻辑
- 主表不从旧主线结果提取 baseline；relative MSE benchmark 必须来自本次 corrected rerun 内部的 HAR
