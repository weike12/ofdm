# OFDM 图像语义通信：五种映射的配对比较

所有方法使用同一个冻结的 SwinJSCC/重要性预测器 checkpoint。实验比较映射规则，不需要重新训练；结果应表述为同一 checkpoint 下的映射消融，不能当作五个分别训练的模型。

| 表格名称 | mapper mode | 分配规则 |
| --- | --- | --- |
| SwinJSCC-OFDM + Sequential | `sequential` | 原 latent 复符号顺序填入数据槽位，padding 留在末尾 |
| SwinJSCC-OFDM + Random | `random` | 全局随机置换复符号，再填入与 sequential 相同的有效槽位 |
| SwinJSCC-OFDM + Greedy | `greedy` | token 重要性与估计槽位质量分别排序后匹配 |
| Greedy + token 内随机重排 | `greedy_token_random` | 先执行 greedy，固定每个 token 的精确槽位集合，只在该 token 内均匀随机置换 |
| Greedy + Token Energy | `greedy_token_energy` | 固定 greedy token 槽位，将高能量符号配给估计 MMSE 衰减较小的位置 |

两个随机方法各用独立的本地 CPU 随机数生成器，不改变全局或信道 RNG。固定 seed 可复现；每个信道 draw 使用一组新置换。token 内随机基线允许抽到原排列，不根据能量或结果拒绝任何置换。它保持 greedy 的 token 槽位和 padding，适合检验能量规则是否优于任意 token 内重排。

当前模拟器在接收端共享排列 metadata；其传输开销尚未计入 CBR。五种方法的数据符号数量和平均发送功率相同。

## 环境与数据

使用之前已成功跑通 DIV2K + LPIPS 的服务器环境。新环境需先安装适配 CUDA 的 PyTorch，再安装：

```bash
python -m pip install -e ".[metrics,dev,swin]"
```

`metrics` 包含 pytorch-msssim 和 lpips，`swin` 包含官方 backbone 需要的 timm。脚本预检会检查 Python >= 3.9、CUDA、数据数量、唯一 image_id、checkpoint，并预加载标准 AlexNet LPIPS 权重。首次加载权重可能需要联网。

checkpoint 路径：`runs/div2k_stageC_sds_ufd/stage_c_final.pt`。数据路径或 checkpoint 位置不同，修改相应 YAML 中的路径。

| 配置 | 数据 | 图像处理 | seed | 每种方法、每个场景的样本数 |
| --- | --- | --- | --- | --- |
| `configs/eval_token_energy.yaml` | DIV2K validation 100 张 | 256 × 256 中心裁剪 | 2000 | 100 × 20 = 2000 |
| `configs/eval_kodak.yaml` | Kodak 24 张 | 原始分辨率 | 3000 | 24 × 20 = 480 |

两者都固定 SNR=10 dB、原始估计 CSI NMSE=-10 dB、8 taps、tau=2、发送端 estimated CSI、接收端 temporal_mean + MMSE，包含无凹口和随机凹口两个场景。随机凹口宽度 8–12 个数据子载波、额外衰减 15–25 dB。每图 20 个配对信道实例。MS-SSIM、LPIPS、映射和复符号残差诊断全部启用。Kodak 的 draw_chunk_size=1，用于控制原图显存占用。

## 在 tmux 中运行

在服务器项目根目录，进入原来成功评估的 Python 环境，然后执行：

```bash
tmux new -s ofdm_baselines
```

在新会话中确认 Python 并运行 DIV2K；成功后顺序运行 Kodak，同一个 GPU 不同时启动两份评估：

```bash
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" -c "import sys,torch; print(sys.executable); print(torch.__version__,torch.cuda.is_available())"

GPU=0 DATASET=div2k SUITE=baselines bash scripts/run_token_energy_probe.sh && \
GPU=0 DATASET=kodak SUITE=baselines bash scripts/run_token_energy_probe.sh
```

只运行 Kodak 时，单独执行最后一条 Kodak 命令。用其他 GPU 时修改 GPU 编号。不用 CLIC。

按 `Ctrl-b`，松开后按 `d`，可以退出 tmux 界面并保持任务运行。重新连接：

```bash
tmux attach -t ofdm_baselines
```

五方法的数据前向次数是原来两方法的 2.5 倍。实际时间还取决于 GPU、图像分辨率和 mapper 开销，不能直接用 DIV2K 时间推算 Kodak；建议整轮放在 tmux 中运行。各阶段耗时写入 `*.seconds`。

## 进度和输出

脚本为每次运行新建目录，不复用旧结果。它依次完成预检、CPU 测试、GPU 评估和统计报告；某阶段失败就停止。

查看当前 DIV2K 任务：

```bash
RUN_DIR="$(cat runs/div2k_baselines_latest.txt)"
cat "$RUN_DIR/status.txt"
tail -n 30 "$RUN_DIR/evaluation.log"
```

Kodak 将第一行换为：

```bash
RUN_DIR="$(cat runs/kodak_baselines_latest.txt)"
```

完成后 `status.txt` 为 completed，`run.exitcode` 以及各阶段 `*.exitcode` 均为 0。检查结果：

```bash
cat "$RUN_DIR/status.txt" "$RUN_DIR/run.exitcode"
cat "$RUN_DIR/summary.csv"
```

主要文件：

- `evaluation/config.yaml`：实际执行的配置，包括五种 mapper。
- `evaluation/samples.csv`：逐图像、逐信道样本。
- `evaluation/summary.json`：每个 mapper、每个凹口场景的汇总。
- `summary.csv`：可直接用于结果表的十行方法/场景汇总。
- `comparison.json`、`comparison.log`：配对差值与图像聚类 bootstrap 区间。
- `environment.json`、`sha256.txt`：依赖版本、GPU、checkpoint 和代码哈希。

DIV2K 共 5 × 2 × 100 × 20 = 20000 个样本，CSV 连表头 20001 行；Kodak 共 4800 个样本，连表头 4801 行。两者各有 10 组 summary。

报告重点比较 energy 与四种基线，以及 greedy 与 sequential/random、token_random 与 greedy。所有差值都是左侧方法减右侧方法；PSNR 和 MS-SSIM 越高越好，LPIPS、失效率及残差越低越好。区间按图像重采样，保留每张图像的全部信道 draw；各项 95% 区间未做多重比较校正。`target_status` 只评价凹口场景 energy 相对 greedy 的 PSNR，不代表赢过所有基线或所有指标。

## 单独生成报告或保留两方法模式

如果评估已成功，只需重做五方法报告：

```bash
python -m si_fdjscc.evaluation.compare_token_energy \
  --run-dir "$RUN_DIR/evaluation" --dataset div2k --suite baselines \
  --expected-count 2000 --expected-draws 20 --resamples 10000 \
  --output-json "$RUN_DIR/comparison.json" --output-csv "$RUN_DIR/summary.csv"
```

Kodak 使用 `--dataset kodak --expected-count 480`。expected-count 是每种方法、每个凹口场景的数量，不是全文件行数。`configs/eval_spectral_notch.yaml` 是另一个 CSI/50-draw 扫参配置，不适用这个冻结协议的比较器。

需要重跑原来的两方法实验时：

```bash
GPU=0 DATASET=div2k SUITE=pair bash scripts/run_token_energy_probe.sh
```

runner 会显式覆盖 mapper 列表并使用 pair 报告。旧两方法结果仍可用默认 `--suite pair` 读取；五方法运行应使用 `--suite baselines`。不要把跨环境或旧轮次的结果直接拼成同一次配对比较。


## 双凹口与 CSI NMSE -5 dB 完整评估

新增的配置与入口是 `configs/eval_double_notch_div2k.yaml`、`configs/eval_double_notch_kodak.yaml` 和 `scripts/run_double_notch_csi_baselines.sh`。旧单凹口配置和结果不受影响。新评估使用相同的五种 mapper、同一个冻结 checkpoint、SNR=10 dB、每图 20 个配对信道 draw；CSI NMSE 分别为 -10 和 -5 dB。每个 CSI 下都评估无凹口及双凹口。双凹口在频率顺序的数据子载波中随机放置，每个宽 8–12 个数据子载波、衰减 15–25 dB，两者至少相隔 4 个数据子载波。额外衰减仍只施加于数据子载波，导频不额外衰减，与原单凹口模型一致。不同方法对同一图像和 draw 使用相同的双凹口、信道、CSI 误差和噪声。

这是**同一冻结模型在更困难信道上的映射对照实验**。Stage C 训练配置目前只包含 CSI NMSE 至 -10 dB，且随机场景是单凹口；因此 -5 dB 和双凹口结果也反映训练分布外的泛化，不能表述为该模型已针对这两个条件训练。新条件的真实指标必须在服务器重跑后从输出文件读取，不能沿用上面的单凹口数值。

先将本次修改的代码、两个新 YAML 和新脚本同步到原来能跑 DIV2K/LPIPS 的服务器环境。进入项目根目录，检查 Python/CUDA、数据和冻结 checkpoint；若该环境没有依赖，先安装与 CUDA 匹配的 PyTorch，再运行 `python -m pip install -e ".[metrics,dev,swin]"`。LPIPS 的 AlexNet 权重应已缓存在服务器，否则首次预检可能需要联网。下面的路径以服务器项目目录 `/data1/weike/ofdm` 为例，实际位置不同时相应替换：

```bash
cd /data1/weike/ofdm
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" -c 'import sys,torch,timm,lpips,pytorch_msssim; print(sys.executable, torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
test -f runs/div2k_stageC_sds_ufd/stage_c_final.pt
test -d datasets/DIV2K/DIV2K_valid_HR
test -d datasets/Kodak
find datasets/DIV2K/DIV2K_valid_HR -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l  # 应为 100
find datasets/Kodak -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l  # 应为 24
bash -n scripts/run_double_notch_csi_baselines.sh
```

若数据或 checkpoint 位于别处，先修改两个新 YAML 中的 `data.test.root` 或 `evaluation.checkpoint`。对 Kodak 保持原图分辨率，不添加裁剪。使用两块 GPU 并行运行两个完整数据集，分别保存控制台日志并检查两个退出码：

```bash
tmux new -s ofdm_double_notch
cd /data1/weike/ofdm
export PYTHON_BIN="$(command -v python)"
mkdir -p runs
GPU=0 DATASET=div2k bash scripts/run_double_notch_csi_baselines.sh > runs/div2k_double_notch_console.log 2>&1 &
DIV_PID=$!
GPU=1 DATASET=kodak bash scripts/run_double_notch_csi_baselines.sh > runs/kodak_double_notch_console.log 2>&1 &
KODAK_PID=$!
if wait "$DIV_PID"; then DIV_RC=0; else DIV_RC=$?; fi
if wait "$KODAK_PID"; then KODAK_RC=0; else KODAK_RC=$?; fi
printf 'DIV2K exit=%s, Kodak exit=%s\n' "$DIV_RC" "$KODAK_RC"
test "$DIV_RC" -eq 0 && test "$KODAK_RC" -eq 0
```

在 tmux 中按 `Ctrl-b` 后按 `d` 可退出并保持任务运行，用 `tmux attach -t ofdm_double_notch` 返回。脚本依次进行依赖/数据/checkpoint 预检、完整 CPU 测试、GPU 评估、配对比较审计和排版。每轮建立新结果目录，失败时停止，并在该目录留下对应的 `*.log`、`*.exitcode` 和耗时文件。两份评估共执行 5 方法 × 2 CSI × 2 凹口场景；DIV2K 的 `samples.csv` 为 40000 行数据，Kodak 为 9600 行数据；各自的 `summary.csv` 应有 20 行数据。

查看运行状态和与示例截图相同字段的实际结果表：

```bash
DIV_DIR="$(cat runs/div2k_double_notch_csi_baselines_latest.txt)"
KODAK_DIR="$(cat runs/kodak_double_notch_csi_baselines_latest.txt)"
cat "$DIV_DIR/status.txt" "$DIV_DIR/run.exitcode"
cat "$KODAK_DIR/status.txt" "$KODAK_DIR/run.exitcode"
tail -n 30 "$DIV_DIR/evaluation.log"
tail -n 30 "$KODAK_DIR/evaluation.log"
cat "$DIV_DIR/table.txt" "$KODAK_DIR/table.txt"
```

表格按数据集与 CSI 分块，每块列出 `no_notch` 和 `two_notches` 下五种方法的样本数、PSNR、MS-SSIM、LPIPS 和 Outage%。`summary.csv` 保留完整数值与 P5 等诊断项，`comparison.json` 保存逐图像聚类 bootstrap 的配对差值及审计结果，`evaluation/samples.csv` 保存逐样本结果；`evaluation/config.yaml`、`environment.json` 和 `sha256.txt` 记录实际配置、环境及代码/checkpoint 哈希。只需重新显示两份结果表时运行：

```bash
python scripts/format_double_notch_results.py "$DIV_DIR/summary.csv" "$KODAK_DIR/summary.csv"
```

双凹口和单凹口是不同实验条件，不应把旧 `runs/.../samples.csv` 与新样本合并为一次配对实验。比较时先检查 `status.txt` 为 `completed`，`run.exitcode`、`preflight.exitcode`、`tests.exitcode`、`evaluation.exitcode`、`comparison.exitcode` 和 `table.exitcode` 均为 0。

## 无凹口、固定 CSI NMSE -10 dB 的五档 SNR 评估

使用 `configs/eval_snr_no_notch_div2k.yaml`、`configs/eval_snr_no_notch_kodak.yaml` 和 `scripts/run_snr_no_notch_baselines.sh`。标称 SNR 为 **10、5、0、-5、-10 dB**；原始 CSI NMSE 仅 `[-10]`，凹口仅 `[null]`。每档 SNR 比较 sequential、random、greedy、greedy_token_random、greedy_token_energy，计算 PSNR、MS-SSIM、LPIPS、Outage%。Outage 固定为 PSNR < 25 dB 的比例乘以 100。

[完整服务器步骤](SNR_SWEEP_SERVER_STEPS.md) 包含上传更新包、环境检查、tmux、进度查看、结果读取和旧目录清理命令。截图中的虚拟环境可用 `source /data1/weike/ofdm/.venv/bin/activate` 激活。启动：

```bash
cd /data1/weike/ofdm
source /data1/weike/ofdm/.venv/bin/activate
export PYTHON_BIN="$(command -v python)"
GPU=0 DATASET=div2k bash scripts/run_snr_no_notch_baselines.sh && \
GPU=0 DATASET=kodak bash scripts/run_snr_no_notch_baselines.sh
```

本轮使用同一冻结 checkpoint、每图 20 个配对信道 draw；DIV2K 为 100 张图的 256×256 中心裁剪，Kodak 为 24 张原始分辨率图。不同方法和 SNR 共享图像、信道和 CSI 随机种子，AWGN 按当前 SNR 缩放。接收端仍使用 temporal_mean + MMSE。backbone 的固定 snr:10 参数不参与当前 SwinJSCC_w/o_SAandRA 分支的计算；实际物理信道、mapper、MMSE 和 SDS 上下文使用 evaluation.snr_db。

每个数据集有 **25 组汇总、35 组配对对比**。DIV2K 50000 条样本，Kodak 12000 条样本；每档 SNR、每种 Mode 仍分别有 2000/480 条样本。-5/-10 dB 超出当前 Stage C 训练范围 [0,20]，属于低 SNR 压力测试。

成功后的目录只保留九个文件：

```text
table.txt
summary.csv
comparison.json
environment.json
sha256.txt
run.json
evaluation/config.yaml
evaluation/samples.csv
evaluation/summary.json
```

状态、各阶段退出码和耗时统一在 run.json。中间日志放 OS 临时目录并在结束时清理；失败时额外保留一个 failure.log。查看最新结果：

```bash
DIV_DIR="$(cat runs/div2k_snr_no_notch_baselines_latest.txt)"
KODAK_DIR="$(cat runs/kodak_snr_no_notch_baselines_latest.txt)"
python -m json.tool "$DIV_DIR/run.json"
python -m json.tool "$KODAK_DIR/run.json"
cat "$DIV_DIR/table.txt" "$KODAK_DIR/table.txt"
```

run.json 应显示 status=completed、exit_code=0。旧三档成功目录可通过显式清理工具精简；默认仅预览，加 --apply 才删除已知旧日志和重复记录，科学输出及未知文件保留：

```bash
python scripts/compact_snr_run_outputs.py "$DIV_DIR"
python scripts/compact_snr_run_outputs.py "$DIV_DIR" --apply
```

## 三张 Kodak 图像的重建效果

固定选取 `kodim01.png`（建筑边缘和砖石）、`kodim07.png`（花叶细纹理）、`kodim04.png`（人像肤色和轮廓）。选择依据是原图内容，不读取各方法得分来挑图或挑信道实例。不同 Kodak 镜像可能存在朝向差异，程序以服务器文件的原始尺寸和朝向为准。

工具只运行 `greedy`、`greedy_token_random`、`greedy_token_energy`，在无凹口和随机凹口两场景下，固定第 0 个信道 draw，共生成 18 次重建。原图全分辨率输入模型，再裁剪同一个 ROI；每种方法不会单独调整亮度或对比度。指标在 PNG 量化前对整图计算，不能当作放大框内的指标。

建议先在 tmux 内导出原图预览，确认放大框后才运行重建：

```bash
tmux new -s kodak_visual
```

进入已跑通实验的 Python 环境，在项目根目录执行：

```bash
VIS_ROOT="$(mktemp -d runs/kodak_visual_XXXXXXXX)"
printf '%s\n' "$VIS_ROOT" | tee runs/kodak_visual_latest.txt
python -m si_fdjscc.tools.visualize_reconstruction \
  --config configs/eval_kodak.yaml \
  --images kodim01 kodim07 kodim04 --draw 0 --prepare-only \
  --output-dir "$VIS_ROOT/selection"
```

先查看 `$VIS_ROOT/selection/selection_preview.png`。默认 ROI 是中心候选框，尚未根据服务器原图确认具体窗框、花朵或面部的位置。如果需要移动框，在查看重建结果前，用 `--roi kodim01=x0,y0,x1,y1` 重新执行准备命令，并改用新的输出目录。四个坐标是相对于原图宽高的 0–1 数值，左上角为原点；三个图片可各指定一次 `--roi`。`selection.json` 固定原图哈希、ROI、完整数据集中的序号和信道 draw。

确认原图及 ROI 后运行：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u -m si_fdjscc.tools.visualize_reconstruction \
  --config configs/eval_kodak.yaml \
  --selection "$VIS_ROOT/selection/selection.json" \
  --output-dir "$VIS_ROOT/results" \
  2>&1 | tee "$VIS_ROOT/reconstruction.log"
```

三方法使用相同信道、CSI、发送符号和噪声种子，程序会检查实际信道/CSI及精确 token 槽位是否一致。信道 seed 使用图像在完整 24 张 Kodak 排序中的原索引，而非三张子集的序号，因此与同环境完整 Kodak 评估中该图的 draw 0 相对应。

完成后查看：

- `results/overview_no_notch.png`、`results/overview_notch.png`：三张图像的三方法总览。
- `results/comparisons/kodim01_no_notch.png` 等六个面板：列为原图、greedy、token random、token energy；上排原图大小比例，下排相同区域放大。
- `results/reconstructions/`：18 张保持原始分辨率的 PNG 重建图。
- `results/crops/`：对应原生像素 ROI，不含绘制在面板上的红框。
- `results/metrics.csv`：18 行整图 PSNR、MS-SSIM、LPIPS，以及 seed、draw、原数据集索引。
- `results/manifest.json`：完成状态、checkpoint/源码哈希和实际随机凹口位置。

局部面板使用统一的最近邻放大，原生 crop 和整图均单独保存。单次信道可视化用于观察细节，多 draw 的统计结论仍以完整评估报告为准。
