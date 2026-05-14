# VoxCPM2 FastAPI 服务说明

这个服务按 VoxCPM2 的能力正向设计，而不是复用其他 TTS 模型的概念。核心资源只有三类：

- `references`：参考音频资产，用于音色克隆或 prompt continuation。
- `synthesis`：一次语音合成请求，可以是普通 TTS、声音设计、音色克隆、音频续写或混合克隆。
- `runtime/model`：运行时资源和模型能力管理。

服务默认懒加载模型；第一次合成或显式调用 `/v1/runtime/load` 时才加载。后台使用单 worker 队列串行推理，适合 demo 级桌面客户端 TTS provider。

## 启动

```bash
uv sync --extra api
uv run voxcpm-api --host 0.0.0.0 --port 7862 --model openbmb/VoxCPM2
```

Mac 本地调试：

```bash
uv run voxcpm-api --host 0.0.0.0 --port 7862 --device mps --no_denoiser
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--api_key` | 开启 API Key 鉴权，客户端使用 `Authorization: Bearer <key>` 或 `X-API-Key: <key>` |
| `--model` | Hugging Face 模型 ID 或本地模型目录 |
| `--output_dir` | 输出目录，默认 `outputs/api` |
| `--device` | `auto`、`cpu`、`mps`、`cuda`、`cuda:0` 等 |
| `--eager_load` | 启动时立即加载模型 |
| `--no_denoiser` | 不加载 ZipEnhancer 降噪模型 |
| `--allow_local_paths` | 允许请求传服务端本地音频路径，默认关闭 |
| `--max_queue_size` | 最大排队合成数，默认 `100` |
| `--lora_path` | 启动时加载 LoRA 权重 |

## 运行时和模型

```bash
curl http://127.0.0.1:7862/health
curl -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/runtime
curl -X POST -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/runtime/load
curl -X POST -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/runtime/unload
curl -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/model
```

LoRA 状态和开关：

```bash
curl -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/model/lora
curl -X POST -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/model/lora/enable
curl -X POST -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/model/lora/disable
```

## 参考音频

上传参考音频：

```bash
curl -X POST http://127.0.0.1:7862/v1/references \
  -H "Authorization: Bearer change-me" \
  -F "audio=@examples/reference_speaker.wav"
```

JSON/base64 上传：

```bash
VOICE_B64=$(base64 -i examples/reference_speaker.wav)

curl -X POST http://127.0.0.1:7862/v1/references/json \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\":\"$VOICE_B64\",\"filename\":\"reference.wav\",\"content_type\":\"audio/wav\"}"
```

同一段音频按 sha256 去重，返回稳定的 `reference_id`。

```bash
curl -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/references
curl -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/references/<reference_id>
curl -X DELETE -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/references/<reference_id>
```

## 合成模式

`voice.mode` 支持：

| mode | 说明 |
| --- | --- |
| `auto` | 根据字段自动判断，推荐客户端默认使用 |
| `plain` | 普通 TTS，仅 `text` |
| `design` | 声音设计，使用 `voice.instruction` |
| `clone` | 音色克隆，使用 `voice.reference_id` 或 `voice.reference_audio` |
| `continuation` | 音频续写，使用 `voice.prompt_*` + `voice.prompt_text` |
| `hybrid` | 混合克隆，同时使用 reference 音频和 prompt 音频/文本，对应 README 的 ultimate cloning |

`voice.instruction` 会按 VoxCPM2 约定拼成 `(<instruction>)<text>`，用于声音设计和可控克隆。

### 声音设计

```bash
curl -X POST http://127.0.0.1:7862/v1/synthesis/jobs \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "欢迎使用 VoxCPM2 API 服务。",
    "voice": {
      "mode": "design",
      "instruction": "年轻女性，温暖自然，语速适中"
    },
    "generation": {
      "cfg_value": 2.0,
      "inference_timesteps": 10,
      "normalize_text": true
    }
  }'
```

### 音色克隆 / 可控克隆

```bash
curl -X POST http://127.0.0.1:7862/v1/synthesis/jobs \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是一段克隆音色的测试。",
    "voice": {
      "mode": "clone",
      "reference_id": "<reference_id>",
      "instruction": "稍微开心一点"
    }
  }'
```

如果传了 `reference_id` 和 `instruction`，即使 `mode=auto`，服务也会识别为可控克隆。

### 音频续写 / Ultimate Cloning

只使用 prompt audio continuation：

```bash
curl -X POST http://127.0.0.1:7862/v1/synthesis/jobs \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "现在开始生成新的语音内容。",
    "voice": {
      "mode": "continuation",
      "prompt_id": "<reference_id>",
      "prompt_text": "参考音频中实际说出的文本。",
      "use_prompt_as_reference": false
    }
  }'
```

README 推荐的 ultimate cloning 是 prompt audio + prompt text，并把同一音频也作为 reference。服务默认 `use_prompt_as_reference=true`，因此下面会自动进入 `hybrid`：

```bash
curl -X POST http://127.0.0.1:7862/v1/synthesis/jobs \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是高相似度克隆生成的内容。",
    "voice": {
      "mode": "auto",
      "prompt_id": "<reference_id>",
      "prompt_text": "参考音频中实际说出的文本。"
    }
  }'
```

## 查询和下载

提交合成后返回 `synthesis_id`：

```json
{
  "synthesis_id": "9f6b5d9f7f8d4d3b9b7b0e4f3d2c1a00",
  "status": "queued",
  "progress": 0.0,
  "message": "queued",
  "synthesis_url": "http://127.0.0.1:7862/v1/synthesis/jobs/...",
  "events_url": "http://127.0.0.1:7862/v1/synthesis/jobs/.../events",
  "audio_url": "http://127.0.0.1:7862/v1/synthesis/jobs/.../audio"
}
```

```bash
curl -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/synthesis/jobs
curl -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/synthesis/jobs/<synthesis_id>
curl -N -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/synthesis/jobs/<synthesis_id>/events
curl -X POST -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/synthesis/jobs/<synthesis_id>/cancel
curl -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/synthesis/jobs/<synthesis_id>/audio --output out.wav
curl -X DELETE -H "Authorization: Bearer change-me" http://127.0.0.1:7862/v1/synthesis/jobs/<synthesis_id>
```

## 流式合成

流式接口不创建任务、不保存 wav，直接通过 SSE 返回 float32 little-endian PCM 的 base64 chunk：

```bash
curl -N -X POST http://127.0.0.1:7862/v1/synthesis/stream/events \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是一段流式生成测试。",
    "voice": {
      "mode": "design",
      "instruction": "自然、清晰、语速适中"
    }
  }'
```

事件：

- `metadata`：`sample_rate`、`channels`、`audio_format=f32le`、`chunk_encoding=base64_f32le`、`mode`
- `chunk`：`audio_base64`
- `done`
- `error`

## 接口列表

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `GET` | `/v1/runtime` | 运行时、队列、设备资源 |
| `POST` | `/v1/runtime/load` | 显式加载模型 |
| `POST` | `/v1/runtime/unload` | 卸载模型并释放缓存 |
| `GET` | `/v1/model` | 模型能力和默认生成参数 |
| `GET` | `/v1/model/lora` | LoRA 状态 |
| `POST` | `/v1/model/lora/enable` | 启用已加载 LoRA |
| `POST` | `/v1/model/lora/disable` | 禁用已加载 LoRA |
| `POST` | `/v1/references` | multipart 上传参考音频 |
| `POST` | `/v1/references/json` | JSON/base64 上传参考音频 |
| `GET` | `/v1/references` | 查看参考音频列表 |
| `GET` | `/v1/references/{reference_id}` | 查看参考音频 |
| `DELETE` | `/v1/references/{reference_id}` | 删除参考音频 |
| `POST` | `/v1/synthesis/jobs` | JSON 提交合成任务 |
| `POST` | `/v1/synthesis/jobs/multipart` | multipart 提交合成任务 |
| `GET` | `/v1/synthesis/jobs` | 查看合成任务 |
| `GET` | `/v1/synthesis/jobs/{synthesis_id}` | 查看任务状态 |
| `GET` | `/v1/synthesis/jobs/{synthesis_id}/events` | SSE 任务状态 |
| `POST` | `/v1/synthesis/jobs/{synthesis_id}/cancel` | 取消任务 |
| `GET` | `/v1/synthesis/jobs/{synthesis_id}/audio` | 下载 wav |
| `DELETE` | `/v1/synthesis/jobs/{synthesis_id}` | 删除任务和结果 |
| `POST` | `/v1/synthesis/stream/events` | SSE 流式合成 |
