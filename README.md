# VoxCPM FastAPI TTS Service

这是一个基于 [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) 的 FastAPI 封装服务，目标是把 VoxCPM2 打造成桌面客户端可调用的 TTS provider。

当前项目文档聚焦 demo 级 HTTP 服务层：

- 支持异步提交语音合成任务、查询状态、下载结果和取消任务。
- 支持参考音频资产管理，便于桌面客户端复用同一段 reference/prompt audio。
- 支持运行时资源管理，包括模型懒加载、显式加载、卸载和队列状态。
- 支持 VoxCPM2 原生能力：普通 TTS、声音设计、音色克隆、可控克隆、音频续写和 hybrid/ultimate cloning。
- 支持 SSE 任务事件和流式合成事件。

详细 HTTP API 示例见 [docs/API_SERVER_ZH.md](docs/API_SERVER_ZH.md)。

## 启动命令

安装 API 依赖：

```bash
uv sync --extra api
```

启动服务：

```bash
uv run voxcpm-api --host 0.0.0.0 --port 7862 --model openbmb/VoxCPM2
```

Mac 本地调试时通常建议关闭 denoiser，并按机器情况指定设备：

```bash
uv run voxcpm-api --host 0.0.0.0 --port 7862 --device mps --no_denoiser
```

如果要开启鉴权：

```bash
uv run voxcpm-api \
  --host 0.0.0.0 \
  --port 7862 \
  --model openbmb/VoxCPM2 \
  --api_key change-me
```

客户端请求时使用下面任一方式传 key：

```http
Authorization: Bearer change-me
X-API-Key: change-me
```

## 常用启动参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `7862` | 监听端口 |
| `--model` | `openbmb/VoxCPM2` | Hugging Face 模型 ID 或本地模型目录 |
| `--output_dir` | `outputs/api` | 参考音频和合成结果保存目录 |
| `--api_key` | 空 | 设置后启用 API key 鉴权 |
| `--cors_origins` | 空 | 逗号分隔的 CORS origin 列表 |
| `--device` | `auto` | `auto`、`cpu`、`mps`、`cuda`、`cuda:0` 等 |
| `--cache_dir` | 空 | Hugging Face 缓存目录 |
| `--local_files_only` | `false` | 只使用本地模型文件 |
| `--eager_load` | 关闭 | 启动时立即加载模型 |
| `--no_denoiser` | 关闭 | 不加载 ZipEnhancer denoiser |
| `--allow_local_paths` | `false` | 允许请求直接引用服务端本地音频路径 |
| `--max_upload_mb` | `50` | 单个上传音频大小限制 |
| `--max_queue_size` | `100` | 最大排队任务数 |
| `--default_cfg_value` | `2.0` | 默认 CFG 参数 |
| `--default_inference_timesteps` | `10` | 默认推理步数 |
| `--lora_path` | 空 | 启动时加载 LoRA 权重 |
| `--reload` | 关闭 | 开发时启用 uvicorn reload |

这些参数也可以通过 `VOXCPM_*` 环境变量配置，例如 `VOXCPM_PORT`、`VOXCPM_MODEL`、`VOXCPM_API_KEY`。

## 服务定位

这个 API server 面向“桌面客户端调用本机或局域网 TTS 服务”的场景。它采用轻量进程内架构：

- 任务和队列状态由服务进程直接管理。
- 任务和引用资产使用进程内索引加本地文件保存。
- 后台单 worker 串行推理，控制 demo 环境下的显存和内存压力。
- 默认懒加载模型，首次合成或调用 `/v1/runtime/load` 时才真正加载。

生产部署时，如果使用带 NVIDIA GPU 的服务器，可以再评估官方推荐的 Nano-vLLM/vLLM Omni 路线。本仓库 HTTP 层按 VoxCPM2 的能力建模，围绕 `runtime/model`、`references` 和 `synthesis` 三类资源组织接口。

## API 资源

服务的核心资源分为三类。

### Runtime / Model

用于健康检查、模型加载卸载、队列状态和 LoRA 状态管理。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `GET` | `/v1/runtime` | 运行时状态、队列、设备和资源信息 |
| `POST` | `/v1/runtime/load` | 显式加载模型 |
| `POST` | `/v1/runtime/unload` | 卸载模型并释放资源 |
| `GET` | `/v1/model` | 模型能力、默认参数和采样率 |
| `GET` | `/v1/model/lora` | LoRA 状态 |
| `POST` | `/v1/model/lora/enable` | 启用已加载 LoRA |
| `POST` | `/v1/model/lora/disable` | 禁用已加载 LoRA |

### References

参考音频资产用于音色克隆、可控克隆、prompt continuation 和 hybrid cloning。同一段音频会按 sha256 去重并返回稳定的 `reference_id`。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v1/references` | multipart 上传参考音频 |
| `POST` | `/v1/references/json` | JSON/base64 上传参考音频 |
| `GET` | `/v1/references` | 查看参考音频列表 |
| `GET` | `/v1/references/{reference_id}` | 查看单个参考音频 |
| `DELETE` | `/v1/references/{reference_id}` | 删除参考音频 |

### Synthesis

语音合成任务异步执行。提交后立即返回 `synthesis_id`，客户端再通过状态接口、SSE 或音频下载接口获取结果。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v1/synthesis/jobs` | JSON 提交合成任务 |
| `POST` | `/v1/synthesis/jobs/multipart` | multipart 提交合成任务 |
| `GET` | `/v1/synthesis/jobs` | 查看任务列表 |
| `GET` | `/v1/synthesis/jobs/{synthesis_id}` | 查看任务状态 |
| `GET` | `/v1/synthesis/jobs/{synthesis_id}/events` | SSE 任务事件 |
| `POST` | `/v1/synthesis/jobs/{synthesis_id}/cancel` | 取消任务 |
| `GET` | `/v1/synthesis/jobs/{synthesis_id}/audio` | 下载 wav 结果 |
| `DELETE` | `/v1/synthesis/jobs/{synthesis_id}` | 删除任务和结果文件 |
| `POST` | `/v1/synthesis/stream/events` | SSE 流式合成，直接返回音频 chunk |

## VoxCPM2 合成模式

请求体中的 `voice.mode` 支持：

| mode | 说明 |
| --- | --- |
| `auto` | 根据字段自动判断，推荐客户端默认使用 |
| `plain` | 普通 TTS，只使用文本 |
| `design` | 声音设计，使用 `voice.instruction` 生成新声音 |
| `clone` | 音色克隆，使用 `voice.reference_id` 或上传 reference audio |
| `continuation` | 音频续写，使用 prompt audio 和 `voice.prompt_text` |
| `hybrid` | 同时使用 reference audio 和 prompt audio/text，对应 high similarity/ultimate cloning |

`voice.instruction` 会按 VoxCPM2 约定拼到文本前面，形如 `(<instruction>)<text>`，可用于声音设计，也可用于可控克隆。

## 快速调用示例

### 声音设计

```bash
curl -X POST http://127.0.0.1:7862/v1/synthesis/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "text": "欢迎使用 VoxCPM2 API 服务。",
    "voice": {
      "mode": "design",
      "instruction": "年轻女性，温暖自然，语速适中"
    }
  }'
```

### 上传参考音频

```bash
curl -X POST http://127.0.0.1:7862/v1/references \
  -F "audio=@examples/reference.wav"
```

返回的 `reference_id` 可以用于后续克隆任务。

### 音色克隆

```bash
curl -X POST http://127.0.0.1:7862/v1/synthesis/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是一段克隆音色的测试。",
    "voice": {
      "mode": "clone",
      "reference_id": "<reference_id>"
    }
  }'
```

### 可控克隆

```bash
curl -X POST http://127.0.0.1:7862/v1/synthesis/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是一段带风格控制的克隆音色测试。",
    "voice": {
      "mode": "clone",
      "reference_id": "<reference_id>",
      "instruction": "更开心一点，语速稍快"
    }
  }'
```

### Hybrid / Ultimate Cloning

```bash
curl -X POST http://127.0.0.1:7862/v1/synthesis/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "text": "现在开始生成新的语音内容。",
    "voice": {
      "mode": "auto",
      "prompt_id": "<reference_id>",
      "prompt_text": "参考音频里实际说出的文本。",
      "use_prompt_as_reference": true
    }
  }'
```

### 查询状态并下载音频

```bash
curl http://127.0.0.1:7862/v1/synthesis/jobs/<synthesis_id>
curl http://127.0.0.1:7862/v1/synthesis/jobs/<synthesis_id>/audio --output out.wav
```

### 流式合成

```bash
curl -N -X POST http://127.0.0.1:7862/v1/synthesis/stream/events \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是一段流式生成测试。",
    "voice": {
      "mode": "design",
      "instruction": "自然、清晰、语速适中"
    }
  }'
```

流式接口通过 SSE 返回：

- `metadata`：采样率、声道数、音频格式和模式。
- `chunk`：base64 编码的 float32 little-endian PCM。
- `done`：生成结束。
- `error`：生成失败。

## 开发和验证

常用检查命令：

```bash
uv run --extra dev black src/voxcpm/api_server.py tests/test_api_server.py api_server.py
uv run python -m py_compile src/voxcpm/api_server.py api_server.py tests/test_api_server.py
uv run --extra dev pytest tests/test_api_server.py -q
```

查看服务参数：

```bash
uv run voxcpm-api --help
```

## 上游项目和许可证

VoxCPM2 模型和原始实现来自 OpenBMB：

- GitHub: <https://github.com/OpenBMB/VoxCPM>
- Hugging Face: <https://huggingface.co/openbmb/VoxCPM2>
- Docs: <https://voxcpm.readthedocs.io/>

本项目继续遵循仓库内 [LICENSE](LICENSE)。
