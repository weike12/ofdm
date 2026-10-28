# 五档标称 SNR、无凹口、五种 Mode 的服务器实验

本轮使用同一个冻结 Stage C checkpoint，分别评估 DIV2K 和 Kodak。参数为：

```yaml
snr_db: [10, 5, 0, -5, -10]
csi_nmse_db: [-10]
spectral_notches: [null]
mapper_modes: [sequential, random, greedy, greedy_token_random, greedy_token_energy]
```

每档 SNR、每种 Mode 都计算 PSNR、MS-SSIM、LPIPS、Outage%。Outage 定义为 PSNR < 25 dB 的比例乘以 100；阈值固定，低 SNR 时可能接近 100%。无凹口场景仍使用原来的 8 taps、tau=2 多径衰落。接收端继续使用 temporal_mean CSI + MMSE，平均后的接收端 CSI NMSE 与原始标称 -10 dB 不同。

## 1. 上传和检查更新包

将 snr_five_levels_compact_update_20261022.zip 上传到 /data1/weike/ofdm。以下命令按截图中的项目和虚拟环境路径编写：

```bash
cd /data1/weike/ofdm
unzip -o snr_five_levels_compact_update_20261022.zip
sha256sum -c MANIFEST.sha256
bash -n scripts/run_snr_no_notch_baselines.sh
source /data1/weike/ofdm/.venv/bin/activate
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" -c 'import sys,torch,timm,lpips,pytorch_msssim,yaml; print(sys.executable); print(torch.__version__,torch.cuda.is_available())'
```

Python 需要 >=3.9，CUDA 检查应为 True。沿用上次成功运行的环境即可。仅在依赖缺失时执行：

```bash
"$PYTHON_BIN" -m pip install -e ".[metrics,dev,swin]"
```

检查数据和 checkpoint：

```bash
test -f runs/div2k_stageC_sds_ufd/stage_c_final.pt
test -d datasets/DIV2K/DIV2K_valid_HR
test -d datasets/Kodak
```

数据或权重路径不同，修改 configs/eval_snr_no_notch_div2k.yaml 和 configs/eval_snr_no_notch_kodak.yaml 的 data.test.root、evaluation.checkpoint。DIV2K 为 100 张验证图，Kodak 为 24 张原图；runner 会严格预检数量、唯一图像 ID、checkpoint、CUDA 和标准 LPIPS 权重。

## 2. 在 tmux 内运行

```bash
tmux new -s ofdm_snr_five_levels
```

进入新会话后执行：

```bash
cd /data1/weike/ofdm
source /data1/weike/ofdm/.venv/bin/activate
export PYTHON_BIN="$(command -v python)"
GPU=0 DATASET=div2k bash scripts/run_snr_no_notch_baselines.sh && \
GPU=0 DATASET=kodak bash scripts/run_snr_no_notch_baselines.sh
```

同一个 GPU 顺序运行两个数据集。只运行 Kodak 时，使用第二条 GPU 命令。更换 GPU 时修改 GPU=0。

按 Ctrl-b，松开后按 d 离开会话，任务会继续。重新进入：

```bash
tmux attach -t ofdm_snr_five_levels
```

runner 会重新执行完整五档实验，依次完成预检、CPU 测试、GPU 评估、配对统计和表格生成。终端实时显示每个阶段及图像进度。同一图像和 draw 在各 SNR、各 Mode 下使用相同信道、CSI 误差方向和标准化 AWGN 方向，噪声幅度按当前 SNR 缩放。

## 3. 查看进度和完成状态

新脚本把阶段状态、退出码和耗时统一写入 run.json，不再为它们分别生成文件。DIV2K 启动后，在另一终端查看：

```bash
cd /data1/weike/ofdm
DIV_DIR="$(cat runs/div2k_snr_no_notch_baselines_latest.txt)"
python -m json.tool "$DIV_DIR/run.json"
```

Kodak 启动后查看：

```bash
KODAK_DIR="$(cat runs/kodak_snr_no_notch_baselines_latest.txt)"
python -m json.tool "$KODAK_DIR/run.json"
```

status 为 running 时，current_stage 表示当前阶段；完成时 status=completed、exit_code=0，各 stages 的 exit_code 均为 0。阶段日志仅存 OS 临时目录，成功后自动清理。失败时 status=failed、failed_stage 标明阶段，目录中额外保留一个 failure.log，已生成的科学输出仍保留。

## 4. 读取五档结果表

两轮完成后执行：

```bash
DIV_DIR="$(cat runs/div2k_snr_no_notch_baselines_latest.txt)"
KODAK_DIR="$(cat runs/kodak_snr_no_notch_baselines_latest.txt)"
python -m json.tool "$DIV_DIR/run.json"
python -m json.tool "$KODAK_DIR/run.json"
cat "$DIV_DIR/table.txt" "$KODAK_DIR/table.txt"
```

输出按 10、5、0、-5、-10 dB 排列，每档五个 Mode，列为 Samples、PSNR、MS-SSIM、LPIPS、Outage%。

| 数据集 | 图像处理 | 每档 SNR、每种 Mode 样本数 | samples.csv 数据行数 | summary.csv 数据行数 |
| --- | --- | ---: | ---: | ---: |
| DIV2K | 100 张验证图，256×256 中心裁剪 | 2000 | 50000 | 25 |
| Kodak | 24 张原始分辨率图 | 480 | 12000 | 25 |

CSV 连同表头分别为 50001/12001 和 26 行。每个数据集生成 35 组预先指定的方法配对对比。-5/-10 dB 超出当前 Stage C 训练 SNR 范围 [0,20]，应作为低 SNR 压力测试解释。

## 5. 成功结果目录只保留 9 个文件

```text
<run_dir>/
  table.txt                 # 便于直接查看的五档结果表
  summary.csv               # 25 行指标汇总
  comparison.json           # 配对差值、图像聚类区间和审计
  environment.json          # 实验环境
  sha256.txt                # 源码和 checkpoint 哈希
  run.json                  # 状态、退出码、耗时、阶段和文件清单
  evaluation/
    config.yaml             # 实际执行配置
    samples.csv             # 全部逐样本数据
    summary.json            # 完整评估汇总与诊断
```

每个数据集在 runs/ 下另有一个 latest.txt 指针，方便定位最新目录。保留 samples.csv 便于核验配对、计算尾部指标和重新统计。无需保留的 base_config.yaml、重复日志、单独的 exitcode/seconds/status 文件不再生成到新结果目录。

## 6. 精简截图中的已有成功目录（可选）

旧三档目录可以直接精简，不需要重新评估。先在启动新五档任务前，用 latest 指针定位旧目录；也可以把 OLD_DIR 设置为你在 VS Code 中看到的具体旧目录路径。

```bash
OLD_DIR="$(cat runs/div2k_snr_no_notch_baselines_latest.txt)"
python scripts/compact_snr_run_outputs.py "$OLD_DIR"
```

上条命令仅显示计划，不改文件。确认目录后执行：

```bash
python scripts/compact_snr_run_outputs.py "$OLD_DIR" --apply
```

Kodak 对应将 latest 文件名改为 kodak_snr_no_notch_baselines_latest.txt。工具只处理完整成功的旧三档/新五档无凹口运行，先把状态/退出码/耗时保存到 run.json，再删除明确列出的旧日志和重复记录。逐样本数据、实际配置和结果报告保留；未识别的文件会列出并保留。失败或运行中的目录不会清理。

## 7. 重排表格或重做报告

只重排已有结果表：

```bash
python scripts/format_snr_no_notch_results.py "$DIV_DIR/summary.csv" "$KODAK_DIR/summary.csv"
```

GPU 评估完成后，仅重做 DIV2K 统计报告：

```bash
python -m si_fdjscc.evaluation.compare_token_energy \
  --run-dir "$DIV_DIR/evaluation" --dataset div2k --suite baselines \
  --protocol snr_sweep --expected-count 2000 --expected-draws 20 --resamples 10000 \
  --output-json "$DIV_DIR/comparison.json" --output-csv "$DIV_DIR/summary.csv"
python scripts/format_snr_no_notch_results.py "$DIV_DIR/summary.csv" > "$DIV_DIR/table.txt"
```

Kodak 改为 --dataset kodak、--expected-count 480 和 KODAK_DIR。新版五档统计器不接收旧三档数据；精简旧三档目录后，其原 summary.csv/table.txt 仍可直接读取。接收端配置、图像处理和 checkpoint 均沿用上一轮协议。
