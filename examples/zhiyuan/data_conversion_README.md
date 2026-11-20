# π0.5 数据转换（智元官方 convert_to_lerobot.py → OpenPI）

> 原则：**优先使用智元官方转换框架**。官方转换器 `convert_to_lerobot.py` 来自
> AgiBotWorld 官方数据集仓库（`agibot-world/AgiBotWorld-Alpha` → `scripts/`，
> CC BY-NC-SA 4.0），**已随本包附带**（`pi05/convert_to_lerobot.py`），无需再申请 gate。

## 〇、为什么需要适配器

官方脚本读取的是 **AgiBot World Alpha 布局**，与课程导出格式（帧目录 jpg + 单个
aligned_joints.h5）不同，且键名有差异（官方要求合并后的 `effector`，课程是
`left/right_effector` 分开）。因此流程是：

```
课程导出 episode ──(pi05/prepare_agibot_layout.py 适配)──> AgiBot 官方布局
                 ──(官方 convert_to_lerobot.py 原样运行)──> LeRobot 数据集
                 ──> OpenPI（π0/π0.5）微调
```

适配器做三件事（均对照官方 `load_local_dataset` 的读取逻辑）：
1. 三路真实相机 jpg → mp4（ffmpeg, 30fps）；官方要求的 5 路鱼眼课程没有，生成
   **黑屏占位视频**（训练配置里不要选这些 key）；深度图同理生成黑 png；
2. `left/right_effector` 合并为官方的 `effector`，拼出 proprio_stats.h5
   （state 20 维 = 关节14+夹爪2+头2+腰2；action 22 维 = 再加底盘速度2）；
3. 生成 `task_info/task_<id>.json`（官方从中拼语言 prompt：`task_name.init_scene_text`）。

## 一、步骤 1：适配为官方布局（本目录脚本，已实测）

```bash
# 单条 / 多条 episode
python prepare_agibot_layout.py --input <episode目录> [--input <episode目录2> ...] \
    --out <AgiBot布局输出根，如 ./agibot_layout>
# 批量
python prepare_agibot_layout.py --input-list episodes.txt --out ./agibot_layout
```

依赖：`h5py`、`Pillow`、`ffmpeg`（PATH 中可用）。

实测输出（group2_ep1015，234 帧）：
- `observations/15/1015/videos/` 8 个 mp4（3 真实 + 5 黑占位）；
- `observations/15/1015/depth/` 234 张黑 png；
- `proprio_stats/15/1015/proprio_stats.h5`：拼接后 state (234,20)、action (234,22)；
- `task_info/task_15.json`：`init_scene_text="白色桌面中央区域有一个玩偶"`。

## 二、步骤 2：运行官方转换器（原样使用，不改代码）

```bash
# 依赖（注意！官方脚本基于 LeRobot 旧版 API：from lerobot.common.datasets...）
python3 -m venv venv_lerobot && source venv_lerobot/bin/activate
pip install torch einops h5py pillow tqdm "lerobot==0.1.0"   # 旧版 lerobot 才有 lerobot.common.*

python convert_to_lerobot.py \
    --src_path ./agibot_layout \
    --task_id 15 \
    --tgt_path ./lerobot_datasets \
    [--debug]        # debug=只转前 2 条 episode，先跑通再全量
```

输出：`./lerobot_datasets/agibotworld/task_15/`（LeRobot 格式：meta/ + parquet + videos/，
自动完成 norm stats 计算）。

> 视频编码说明：官方 FEATURES 元数据声明 av1，本适配器用 h264（libx264）——
> LeRobot 按容器实际编码解码，不受影响；转换后 `meta/info.json` 如提示 codec 不符可忽略。

## 三、步骤 3：OpenPI（π0 / π0.5）微调

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi && uv sync

# 1. 计算 norm stats（OpenPI 训练前必须）
uv run scripts/compute_norm_stats.py --config <你的配置名>

# 2. 微调（π0.5 低配微调用 LoRA；配置在 src/openpi/training/config.py 注册）
uv run scripts/train.py <你的配置名> --exp-name=group2_pio5
```

自定义配置最小写法（在 openpi 里新建配置模块并 import 注册）：

```python
import openpi_training.config as _config
_config.TrainConfig(
    model=openpi_models.Pi05Config(),                 # 或 Pi0Config()
    data=LeRobotDataConfig(
        repo_id="<本地 LeRobot 数据集路径>",
        prompt_from_task=True,                        # 语言任务列做 prompt
        # 相机只选真实三路，避开黑占位
        # video_keys / image_keys 视 openpi 版本的字段名而定
    ),
    batch_size=8, num_train_steps=20_000,
)
```

> openpi 主分支无 AgiBot 示例（历史版本的 agibot_genie1 示例已移除），
> 按 LeRobotDataset 自定义配置即可——LeRobot 格式是 OpenPI 的一等公民输入。

## 四、与 RDT 管线的关系

- 源数据相同（GenieStudio 导出 episode），两条管线各自转换，互不干扰；
- RDT 管线见 `../rdt/`（课程自有脚本，输出 hdf5 目录布局）；
- 本目录官方脚本（输出 LeRobot 布局）已验证与课程数据的适配。
