# 02 · DeepSeek V4.1 Flash with Ray

[English](#english) · [Русский](#русский) · [All samples / Все примеры](../README.md)

## English

### What this sample provides

Serves the local DeepSeek V4.1 Flash (1017) checkpoint on two nodes with 16 H200 GPUs through Ray: **TP8 / DP2 / EP16 / PP1**, one DP engine per node and one OpenAI-compatible API on the head. The API distributes requests across both DP engines. A 16-GPU Ray/CUDA/NCCL smoke test runs before model startup. Readiness requires a generated answer from the real API.

The tested preset is text-only, uses eager execution and sets a **16,384-token total context**, eight concurrent sequences per DP engine and a `0.85` GPU memory budget. These settings differ from the multiprocessing examples. Ray and vLLM are installed inside jobs; the management Jupyter does not need them.

### Included files

- `submit_vllm.py`, `query_vllm.py`: submit/manage the service and send an authenticated request.
- `bootstrap_vllm.py`, `vllm_node.py`, `vllm_service.py`: install dependencies, start Ray and supervise the head API.
- `vllm_common.py`, `ep_preflight.py`: checkpoint/topology validation, HTTP helpers and MoE backend checks.
- `cluster_common.py`, `bootstrap_ray.py`, `ray_node.py`, `smoke_test.py`: Ray cluster and GPU/NCCL checks.
- `submit_ray.py`, `requirements-ray.txt`: optional standalone Ray check, with the parameters described below.
- `requirements-vllm.txt`: runtime pins; `tests/`: cloud-free cluster/deployment tests.

### Run

Use Cloud.ru SR009 management Jupyter with the **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`** image, which provides **`/usr/bin/python3.12`** and the platform-provided `client_lib`; reserve two distinct nodes with eight H200 GPUs each. The defaults target `hpcsrv-201.sr009` and `hpcsrv-202.sr009`. The repository and complete model checkpoint must be on NFS accessible at the same absolute paths from Jupyter and both job containers. Jobs need dependency-download access and active InfiniBand; Jupyter must be able to reach the job pod IPs. No custom image build is required.

```bash
cd /home/jovyan/samples/02-vllm-ray

# Inspect the plan without reading weights or allocating GPUs.
/usr/bin/python3.12 submit_vllm.py --dry-run

# Launch with the tested defaults; retain the service for 12 hours after readiness.
/usr/bin/python3.12 submit_vllm.py --serve-minutes 720
```

To use another checkpoint or allocation, add `--model-dir /absolute/path/to/checkpoint --nodes NODE0 NODE1`. `--prepare-only` validates weights and creates launch files without allocating GPUs. The submit command prints installation/startup progress and returns after the first verified response; the service continues in cloud jobs. Omitting `--serve-minutes` gives a two-hour lease after readiness.

Replace `ID` below with the exact printed run directory.

```bash
RUN_DIR=/home/jovyan/cloud-vllm-ep-ID
/usr/bin/python3.12 submit_vllm.py --status "$RUN_DIR"
/usr/bin/python3.12 query_vllm.py --run-dir "$RUN_DIR" --prompt "Explain expert parallelism in three sentences." --max-tokens 512

# Enable reasoning for a separate request.
/usr/bin/python3.12 query_vllm.py --run-dir "$RUN_DIR" --prompt "Solve 3*x + 7 = 22." --thinking --max-tokens 2048

# Inspect startup errors or stop the service on both nodes.
/usr/bin/python3.12 submit_vllm.py --diagnose "$RUN_DIR"
/usr/bin/python3.12 submit_vllm.py --stop "$RUN_DIR"
```

### Results and API access

`READY.json` records `base_url`, `openai_base_url`, model name and expiry. The generated `api-key` file supplies Bearer authentication. The run directory is a filesystem location for metadata, not a URL. The API runs at the job pod IP, for example `http://10.x.x.x:8000/v1`; `localhost` in Jupyter refers to Jupyter itself.

Each run keeps `run.json`, `node-*.json`, `status-*.json`, `READY.json`, `install-*.log`, `vllm.log`, `nccl.*.log` and `*-error*.json`. Ray checks also produce `PASSED.json` and `smoke.log`; the exact vLLM command is in `server-command.json`. `STOP` requests shutdown; `stopped-*.json` confirms supervisor exit. Keep `api-key` and generated run snapshots out of version control.

### Launch parameters

All options below belong to `submit_vllm.py`. Topology is fixed in this sample. `--status`, `--diagnose` and `--stop` are separate operations; use one per command. Paths used by default:

```text
BASE_IMAGE=cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42
MODEL_DIR=/home/jovyan/MODELS/deepseek-ai_DeepSeek-V4.1-Flash-1017/deepseek-ai_DeepSeek-V4.1-Flash-1017
```

| Parameter | Default | Description |
| --- | --- | --- |
| `--model-dir PATH` | `MODEL_DIR` | Local NFS checkpoint directory shown above; must contain config, tokenizer, index and all weight shards. No automatic model download. |
| `--image IMAGE` | `BASE_IMAGE` | GPU job image shown above: full `cr.ai.cloud.ru` reference with a tag or digest. |
| `--nodes NODE0 NODE1` | `hpcsrv-201.sr009 hpcsrv-202.sr009` | Two physical hosts, head first; each receives a separate job with eight H200 GPUs. |
| `--nfs-root PATH` | `/home/jovyan` | Existing shared directory for run snapshots; does not change the checkpoint path. |
| `--api-port PORT` | `8000` | HTTP port, 1024–65535; the same port is configured on each API node. |
| `--socket-ifname NAME` | `eth0` | Pod TCP control interface for Gloo/NCCL. NCCL GPU data transport uses InfiniBand. |
| `--served-model-name NAME` | `deepseek-v4.1-flash` | Model name exposed through the OpenAI-compatible API. |
| `--max-model-len TOKENS` | `16384` | Positive total context limit: input plus generated tokens. |
| `--max-num-seqs N` | `8` | Positive maximum concurrent sequences per DP engine; not the global HTTP concurrency. |
| `--gpu-memory-utilization FRACTION` | `0.85` | GPU memory budget fraction per engine; `0 < value ≤ 0.95`. |
| `--startup-timeout SECONDS` | `3600` | Positive time budget for queueing, environment installation and Ray startup. |
| `--model-timeout SECONDS` | `3600` | Positive time budget for weight loading, compilation and API startup. |
| `--serve-minutes MINUTES` | `120` | Positive maximum serving lifetime AFTER `READY.json` is created; `720` = 12 hours, `1440` = 24 hours. |
| `--dry-run` | `false` | Print configuration, resource plan and server command(s); do not read weights, create files or submit jobs. |
| `--prepare-only` | `false` | Validate the checkpoint and write launch files/API key to NFS without submitting jobs. |
| `--status RUN_DIR` | — | Print saved status/readiness/errors for an existing run; does not probe the live API. |
| `--diagnose RUN_DIR` | — | Print saved NFS logs/errors with API-key redaction; creates no jobs. |
| `--stop RUN_DIR` | — | Write `STOP`; supervisors terminate vLLM on both nodes. |
| `-h`, `--help` | — | Print command help and exit. |

### Request parameters

`query_vllm.py` reads the endpoint, model name and API key from the run directory and prints the full JSON response. It uses `/v1/chat/completions`, a fixed temperature of `0.2`, and Python standard-library HTTP without inherited proxies.

| Parameter | Default | Description |
| --- | --- | --- |
| `--run-dir PATH` | Required | Run directory printed by the submit command; must contain an unexpired `READY.json` and `api-key`. |
| `--prompt TEXT` | `Briefly explain tensor parallelism.` | User message sent to the model. |
| `--max-tokens N` | `512` | Maximum generated tokens, including reasoning tokens when enabled; must fit the remaining model context. |
| `--timeout SECONDS` | `300` | HTTP request timeout; does not extend the serving lifetime. |
| `--thinking` | `false` | Send `chat_template_kwargs.thinking=true`; increase the token budget for reasoning. No reasoning-effort option is exposed. |
| `-h`, `--help` | — | Print command help and exit. |

### Server configuration

The launcher builds the following vLLM configuration. Values tied to launch parameters are described above; inspect `--dry-run` for the complete command.

| Setting | Value | Purpose |
| --- | --- | --- |
| `--tokenizer-mode` | `deepseek_v41` | Model-specific tokenizer mode. |
| `--tool-call-parser`, `--enable-auto-tool-choice` | `deepseek_v41`, enabled | Parse tool calls and enable automatic tool selection. |
| `--reasoning-parser` | `deepseek_v41` | Parse model reasoning output. |
| `--enable-expert-parallel` | enabled | Shard MoE experts across all 16 GPUs. |
| `--host`, `--port`, `--served-model-name` | `0.0.0.0`, CLI values | Listen on pod interfaces; use the configured HTTP port and model alias. |
| `--distributed-executor-backend`, `--data-parallel-backend` | `ray`, `ray` | Use Ray for distributed workers and DP engines. |
| `--tensor-parallel-size`, `--pipeline-parallel-size` | `8`, `1` | An eight-GPU tensor-parallel group on each node; no pipeline split. |
| `--data-parallel-size`, `--data-parallel-size-local` | `2`, `1` | Two DP engines, one per node. |
| `--all2all-backend`, `--moe-backend` | `allgather_reducescatter`, `humming` | MoE communication and compute backends. |
| `--api-server-count` | `1` | One API frontend on the head node. |
| `--engram-config` | `{"cpu_offload":true,"embedding_across_dp":false}` | Offload Engram embeddings to CPU; do not shard embeddings across DP. |
| `--language-model-only` | enabled | Text-only inference; skip multimodal components. |
| `--enforce-eager`, `--no-swa-bounded-replay` | enabled | Eager execution with bounded SWA replay disabled. |
| `--max-num-batched-tokens` | `min(4096, max_model_len)` | Token budget per scheduler batch. |
| `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization` | `16384`, `8`, `0.85` | Default resource/context limits; overridable through the launcher. |
| Frontend / MoE environment | `VLLM_USE_RUST_FRONTEND=0`, `VLLM_HUMMING_MOE_GEMM_TYPE=grouped` | Python frontend and grouped Humming GEMM; strict Ray placement per node. |

### Worker entrypoints

The submit script starts these automatically inside jobs; they are not additional getting-started steps. All accept `-h` / `--help`.

| Script | Arguments | Description |
| --- | --- | --- |
| `bootstrap_vllm.py` | `--run-dir PATH --rank RANK` | Required: run snapshot; rank `0`, `1` or `auto` (MPI environment). |
| `vllm_node.py` | `--run-dir PATH --rank 0\|1` | Required: run snapshot and node rank. |
| `vllm_service.py` | `--run-dir PATH` | Required: run snapshot; supervises smoke tests and the head API. |

### Optional standalone Ray check

`submit_ray.py` is retained as an independent infrastructure check. It does not start the model. Its worker entrypoints and all CLI parameters follow. For its run lifecycle and commands, see [sample 01](../01-ray-cluster/README.md#english).

```bash
/usr/bin/python3.12 submit_ray.py --gpus-per-node 8
```

#### Ray worker entrypoints

The submit script starts these automatically inside jobs; they are not additional getting-started steps. All accept `-h` / `--help`.

| Script | Arguments | Description |
| --- | --- | --- |
| `bootstrap_ray.py` | `--run-dir PATH --rank RANK` | Required: run snapshot; rank `0`, `1` or `auto` (MPI environment). |
| `ray_node.py` | `--run-dir PATH --rank 0\|1` | Required: run snapshot and node rank. |
| `smoke_test.py` | `--run-dir PATH` | Required: run snapshot with head and node metadata. |

#### Ray launch parameters

All options belong to `submit_ray.py`. `BASE_IMAGE` is:

```text
cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42
```

| Parameter | Default | Description |
| --- | --- | --- |
| `--image IMAGE` | `BASE_IMAGE` | GPU job image shown above; full `cr.ai.cloud.ru` reference with a tag or digest. |
| `--nodes NODE0 NODE1` | `hpcsrv-201.sr009 hpcsrv-202.sr009` | Two physical node names, head first. |
| `--gpus-per-node N` | `1` | GPUs on each node, from 1 to 8; total GPU count is `2 × N`. |
| `--cpus-per-node N` | `12 × GPUs per node` | Logical Ray CPUs per node, at least the GPU count; does not change the Cloud.ru instance shape. |
| `--nfs-root PATH` | `/home/jovyan` | Existing shared directory for run snapshots and results. |
| `--placement MODE` | `pinned` | `pinned`: two jobs pinned to separate hosts. `single-job`: one job with two workers; the physical-node check still applies. |
| `--startup-timeout SECONDS` | `900` | Positive startup time budget for node registration and Ray. |
| `--test-timeout SECONDS` | `600` | Positive test time budget after the Ray head is ready. |
| `--keep-alive` | `false` | Keep the cluster after a successful test. |
| `--keep-minutes MINUTES` | `30` | Positive maximum retention after the test; effective with `--keep-alive`. |
| `--dry-run` | `false` | Print configuration and job requests; create no files or jobs. |
| `--prepare-only` | `false` | Write a launch snapshot to NFS without submitting jobs. |
| `--stop RUN_DIR` | — | Write a shutdown request for an existing run containing `run.json`. |
| `-h`, `--help` | — | Print command help and exit. |

### Local checks

Run from this directory without GPU jobs or model weights. Tests mock cloud/runtime operations; they do not replace a GPU inference run.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover -s tests -v
```

## Русский

### Что даёт пример

Поднимает локальный checkpoint DeepSeek V4.1 Flash (1017) на двух узлах с 16 H200 через Ray: **TP8 / DP2 / EP16 / PP1**, по одному DP-движку на узел и один OpenAI-совместимый API на головном узле. API распределяет запросы между двумя DP-движками. Перед запуском модели выполняется проверка Ray/CUDA/NCCL на 16 GPU. Готовность подтверждается сгенерированным ответом реального API.

Проверенный пресет работает только с текстом, использует eager-режим и задаёт **полный контекст 16 384 токена**, восемь одновременных последовательностей на DP-движок и долю памяти GPU `0.85`. Эти настройки отличаются от примеров multiprocessing. Ray и vLLM устанавливаются внутри задач; управляющему Jupyter они не нужны.

### Состав

- `submit_vllm.py`, `query_vllm.py`: запуск/управление сервисом и авторизованный запрос.
- `bootstrap_vllm.py`, `vllm_node.py`, `vllm_service.py`: установка зависимостей, запуск Ray и управление API на головном узле.
- `vllm_common.py`, `ep_preflight.py`: проверка весов/топологии, HTTP-функции и проверки бэкенда MoE.
- `cluster_common.py`, `bootstrap_ray.py`, `ray_node.py`, `smoke_test.py`: кластер Ray и проверки GPU/NCCL.
- `submit_ray.py`, `requirements-ray.txt`: дополнительная отдельная проверка Ray; параметры описаны ниже.
- `requirements-vllm.txt`: версии runtime; `tests/`: локальные тесты кластера и запуска без облака.

### Запуск

Используйте управляющий Jupyter Cloud.ru SR009 на образе **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`**. Именно в этом образе доступен **`/usr/bin/python3.12`** и установлен платформенный `client_lib`; нужны два разных узла по восемь H200. По умолчанию указаны `hpcsrv-201.sr009` и `hpcsrv-202.sr009`. Репозиторий и полный checkpoint должны лежать на NFS по одинаковым абсолютным путям в Jupyter и обоих контейнерах задач. Задачам нужны доступ к зависимостям и активный InfiniBand; Jupyter должен иметь сетевой доступ к IP подов задач. Собирать собственный образ не требуется.

```bash
cd /home/jovyan/samples/02-vllm-ray

# Посмотреть план без чтения весов и выделения GPU.
/usr/bin/python3.12 submit_vllm.py --dry-run

# Запустить с проверенными настройками; оставить сервис на 12 часов после готовности.
/usr/bin/python3.12 submit_vllm.py --serve-minutes 720
```

Для другого checkpoint или набора узлов добавьте `--model-dir /absolute/path/to/checkpoint --nodes NODE0 NODE1`. `--prepare-only` проверяет веса и создаёт файлы запуска без выделения GPU. Команда выводит прогресс установки/запуска и завершается после первого проверенного ответа; сервис продолжает работать в облачных задачах. Без `--serve-minutes` время работы после готовности составляет два часа.

Ниже замените `ID` точным каталогом, напечатанным командой запуска.

```bash
RUN_DIR=/home/jovyan/cloud-vllm-ep-ID
/usr/bin/python3.12 submit_vllm.py --status "$RUN_DIR"
/usr/bin/python3.12 query_vllm.py --run-dir "$RUN_DIR" --prompt "Explain expert parallelism in three sentences." --max-tokens 512

# Включить рассуждения для отдельного запроса.
/usr/bin/python3.12 query_vllm.py --run-dir "$RUN_DIR" --prompt "Solve 3*x + 7 = 22." --thinking --max-tokens 2048

# Посмотреть ошибки запуска или остановить сервис на обоих узлах.
/usr/bin/python3.12 submit_vllm.py --diagnose "$RUN_DIR"
/usr/bin/python3.12 submit_vllm.py --stop "$RUN_DIR"
```

### Результаты и доступ к API

В `READY.json` записываются `base_url`, `openai_base_url`, имя модели и время окончания работы. Созданный файл `api-key` используется для Bearer-авторизации. Каталог запуска — путь к метаданным на диске, а не URL. API работает по IP пода задачи, например `http://10.x.x.x:8000/v1`; `localhost` в Jupyter указывает на сам Jupyter.

Для каждого запуска сохраняются `run.json`, `node-*.json`, `status-*.json`, `READY.json`, `install-*.log`, `vllm.log`, `nccl.*.log` и `*-error*.json`. Проверки Ray также создают `PASSED.json` и `smoke.log`; точная команда vLLM записана в `server-command.json`. `STOP` запрашивает остановку; `stopped-*.json` подтверждают завершение управляющих процессов. Не добавляйте `api-key` и каталоги запусков в систему контроля версий.

### Параметры запуска

Все параметры ниже относятся к `submit_vllm.py`. Топология фиксирована для этого примера. `--status`, `--diagnose` и `--stop` — отдельные операции; указывайте одну на команду. Пути по умолчанию:

```text
BASE_IMAGE=cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42
MODEL_DIR=/home/jovyan/MODELS/deepseek-ai_DeepSeek-V4.1-Flash-1017/deepseek-ai_DeepSeek-V4.1-Flash-1017
```

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `--model-dir PATH` | `MODEL_DIR` | Каталог весов на NFS указан выше; нужны конфигурация, токенизатор, индекс и все шарды. Автоматической загрузки модели нет. |
| `--image IMAGE` | `BASE_IMAGE` | Образ GPU-задачи указан выше: полный адрес `cr.ai.cloud.ru` с тегом или digest. |
| `--nodes NODE0 NODE1` | `hpcsrv-201.sr009 hpcsrv-202.sr009` | Два физических узла, головной — первым; на каждый отправляется отдельная задача с восемью H200. |
| `--nfs-root PATH` | `/home/jovyan` | Существующий общий каталог для файлов запусков; путь к весам не меняет. |
| `--api-port PORT` | `8000` | Порт HTTP, 1024–65535; одинаковый на всех узлах с API. |
| `--socket-ifname NAME` | `eth0` | TCP-интерфейс пода для управления Gloo/NCCL. GPU-данные NCCL передаются через InfiniBand. |
| `--served-model-name NAME` | `deepseek-v4.1-flash` | Имя модели в OpenAI-совместимом API. |
| `--max-model-len TOKENS` | `16384` | Положительный лимит полного контекста: вход плюс генерируемые токены. |
| `--max-num-seqs N` | `8` | Положительный предел одновременных последовательностей на DP-движок; не глобальный лимит HTTP-запросов. |
| `--gpu-memory-utilization FRACTION` | `0.85` | Доля памяти GPU для движка; `0 < значение ≤ 0.95`. |
| `--startup-timeout SECONDS` | `3600` | Положительный лимит времени на очередь, установку окружения и запуск Ray. |
| `--model-timeout SECONDS` | `3600` | Положительный лимит времени на загрузку весов, компиляцию и запуск API. |
| `--serve-minutes MINUTES` | `120` | Положительный лимит работы ПОСЛЕ создания `READY.json`; `720` = 12 часов, `1440` = 24 часа. |
| `--dry-run` | `false` | Вывести конфигурацию, план ресурсов и команды сервера; не читать веса, не создавать файлы и задачи. |
| `--prepare-only` | `false` | Проверить веса и записать файлы запуска/API-ключ на NFS без отправки задач. |
| `--status RUN_DIR` | — | Вывести сохранённые статус, готовность и ошибки запуска; живой API не опрашивается. |
| `--diagnose RUN_DIR` | — | Вывести логи/ошибки с NFS, скрывая API-ключ; задачи не создаются. |
| `--stop RUN_DIR` | — | Записать `STOP`; управляющие процессы завершат vLLM на обоих узлах. |
| `-h`, `--help` | — | Вывести справку и завершиться. |

### Параметры запроса

`query_vllm.py` читает адрес, имя модели и API-ключ из каталога запуска и выводит полный JSON-ответ. Использует `/v1/chat/completions`, фиксированную температуру `0.2` и HTTP из стандартной библиотеки Python без унаследованных прокси.

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `--run-dir PATH` | Обязателен | Каталог, напечатанный командой запуска; нужны актуальный `READY.json` и `api-key`. |
| `--prompt TEXT` | `Briefly explain tensor parallelism.` | Пользовательское сообщение модели. |
| `--max-tokens N` | `512` | Максимум генерируемых токенов, включая reasoning при его включении; должен помещаться в оставшийся контекст. |
| `--timeout SECONDS` | `300` | Таймаут HTTP-запроса; время работы сервиса не продлевает. |
| `--thinking` | `false` | Передать `chat_template_kwargs.thinking=true`; для рассуждений увеличьте бюджет токенов. Отдельного параметра reasoning effort нет. |
| `-h`, `--help` | — | Вывести справку и завершиться. |

### Конфигурация сервера

Лаунчер формирует следующую конфигурацию vLLM. Значения, связанные с параметрами запуска, описаны выше; полная команда выводится через `--dry-run`.

| Настройка | Значение | Назначение |
| --- | --- | --- |
| `--tokenizer-mode` | `deepseek_v41` | Режим токенизатора модели. |
| `--tool-call-parser`, `--enable-auto-tool-choice` | `deepseek_v41`, включено | Разбор вызовов инструментов и автоматический выбор инструмента. |
| `--reasoning-parser` | `deepseek_v41` | Разбор рассуждений модели. |
| `--enable-expert-parallel` | включено | Распределить MoE-экспертов между всеми 16 GPU. |
| `--host`, `--port`, `--served-model-name` | `0.0.0.0`, значения CLI | Слушать интерфейсы пода; использовать указанный HTTP-порт и имя модели. |
| `--distributed-executor-backend`, `--data-parallel-backend` | `ray`, `ray` | Использовать Ray для распределённых воркеров и DP-движков. |
| `--tensor-parallel-size`, `--pipeline-parallel-size` | `8`, `1` | Группа тензорного параллелизма из восьми GPU на каждом узле; без разбиения pipeline. |
| `--data-parallel-size`, `--data-parallel-size-local` | `2`, `1` | Два DP-движка, по одному на узел. |
| `--all2all-backend`, `--moe-backend` | `allgather_reducescatter`, `humming` | Бэкенды обмена данными и вычисления MoE. |
| `--api-server-count` | `1` | Один API-фронтенд на головном узле. |
| `--engram-config` | `{"cpu_offload":true,"embedding_across_dp":false}` | Выгрузить Engram embeddings в CPU; не распределять embeddings между DP. |
| `--language-model-only` | включено | Только текстовый инференс; без мультимодальных компонентов. |
| `--enforce-eager`, `--no-swa-bounded-replay` | включено | Выполнение в eager-режиме с отключённым bounded SWA replay. |
| `--max-num-batched-tokens` | `min(4096, max_model_len)` | Бюджет токенов на батч планировщика. |
| `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization` | `16384`, `8`, `0.85` | Лимиты ресурсов и контекста по умолчанию; меняются параметрами лаунчера. |
| Frontend / MoE environment | `VLLM_USE_RUST_FRONTEND=0`, `VLLM_HUMMING_MOE_GEMM_TYPE=grouped` | Python-фронтенд и grouped Humming GEMM; строгое размещение Ray по узлам. |

### Служебные точки входа

Скрипт отправки запускает их автоматически внутри задач; вручную запускать их для начала работы не требуется. Все поддерживают `-h` / `--help`.

| Скрипт | Аргументы | Описание |
| --- | --- | --- |
| `bootstrap_vllm.py` | `--run-dir PATH --rank RANK` | Обязательны: каталог запуска; ранг `0`, `1` или `auto` (из окружения MPI). |
| `vllm_node.py` | `--run-dir PATH --rank 0\|1` | Обязательны: каталог запуска и ранг узла. |
| `vllm_service.py` | `--run-dir PATH` | Обязателен каталог запуска; управляет тестами и API на головном узле. |

### Дополнительная отдельная проверка Ray

`submit_ray.py` сохранён для независимой проверки инфраструктуры. Модель он не запускает. Ниже приведены служебные точки входа и все параметры CLI. Жизненный цикл и команды описаны в [примере 01](../01-ray-cluster/README.md#русский).

```bash
/usr/bin/python3.12 submit_ray.py --gpus-per-node 8
```

#### Служебные точки входа Ray

Скрипт отправки запускает их автоматически внутри задач; вручную запускать их для начала работы не требуется. Все поддерживают `-h` / `--help`.

| Скрипт | Аргументы | Описание |
| --- | --- | --- |
| `bootstrap_ray.py` | `--run-dir PATH --rank RANK` | Обязательны: каталог запуска; ранг `0`, `1` или `auto` (из окружения MPI). |
| `ray_node.py` | `--run-dir PATH --rank 0\|1` | Обязательны: каталог запуска и ранг узла. |
| `smoke_test.py` | `--run-dir PATH` | Обязателен каталог запуска с метаданными головного процесса и узлов. |

#### Параметры запуска Ray

Все параметры относятся к `submit_ray.py`. Значение `BASE_IMAGE`:

```text
cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42
```

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `--image IMAGE` | `BASE_IMAGE` | Образ GPU-задачи указан выше; полный адрес `cr.ai.cloud.ru` с тегом или digest. |
| `--nodes NODE0 NODE1` | `hpcsrv-201.sr009 hpcsrv-202.sr009` | Имена двух физических узлов, головной — первым. |
| `--gpus-per-node N` | `1` | GPU на каждом узле: от 1 до 8; всего `2 × N` GPU. |
| `--cpus-per-node N` | `12 × GPUs per node` | Логические CPU Ray на узел, не меньше числа GPU; не меняет тип инстанса Cloud.ru. |
| `--nfs-root PATH` | `/home/jovyan` | Существующий общий каталог для файлов запусков и результатов. |
| `--placement MODE` | `pinned` | `pinned`: две задачи, каждая закреплена за своим узлом. `single-job`: одна задача с двумя воркерами; проверка разных физических узлов сохраняется. |
| `--startup-timeout SECONDS` | `900` | Положительный лимит времени на регистрацию узлов и запуск Ray. |
| `--test-timeout SECONDS` | `600` | Положительный лимит времени на тест после готовности головного процесса Ray. |
| `--keep-alive` | `false` | Оставить кластер работающим после успешного теста. |
| `--keep-minutes MINUTES` | `30` | Положительный лимит работы после теста; действует с `--keep-alive`. |
| `--dry-run` | `false` | Вывести конфигурацию и запросы ресурсов без создания файлов и задач. |
| `--prepare-only` | `false` | Создать файлы запуска на NFS без отправки задач. |
| `--stop RUN_DIR` | — | Запросить остановку существующего запуска с файлом `run.json`. |
| `-h`, `--help` | — | Вывести справку и завершиться. |

### Локальные проверки

Запускаются из этого каталога без GPU-задач и весов модели. Тесты имитируют облако/runtime; они не заменяют запуск инференса на GPU.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover -s tests -v
```
