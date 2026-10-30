# 04 · DeepSeek V4.1 Flash with hybrid load balancing

[English](#english) · [Русский](#русский) · [All samples / Все примеры](../README.md)

## English

### What this sample provides

Serves DeepSeek V4.1 Flash (1017) on two nodes with 16 H200 GPUs using native vLLM multiprocessing: **TP1 / DP16 / EP16 / PP1**, eight DP engines per node. The MoE experts span all 16 GPUs. Uses stock vLLM 0.32.0 and the Rust frontend, without runtime source patches or Ray.

**Hybrid load balancing:** each node exposes an HTTP API that balances its eight local DP engines. This sample provides two API endpoints; it does not include an upstream HTTP load balancer. A client or an external load balancer chooses an endpoint. Readiness requires a real generated answer from both APIs and NCCL InfiniBand channel evidence.

The launcher leaves the context, sequence count, memory budget, CUDA graph and model-backend defaults to vLLM unless an exposed override is supplied. In the tested checkpoint the reported maximum context was **1,048,576 tokens**; this is a model limit, not a guarantee of available KV-cache capacity for every workload. Context includes input plus output.

### Included files

- `submit_vllm.py`, `query_vllm.py`: submit/manage the service and send an authenticated request.
- `bootstrap_vllm.py`, `vllm_node.py`: install the isolated runtime, verify nodes/GPUs and supervise vLLM.
- `vllm_common.py`, `cluster_common.py`: checkpoint validation, server command, environment, HTTP and Cloud.ru job helpers.
- `diagnose_vllm.py`: extract saved startup/worker errors, including interleaved tracebacks.
- `requirements-vllm.txt`: runtime pins; `tests/test_dep.py`: cloud-free deployment/lifecycle tests.

### Run

Use Cloud.ru SR009 management Jupyter with the **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`** image, which provides **`/usr/bin/python3.12`** and the platform-provided `client_lib`; reserve two distinct nodes with eight H200 GPUs each. The defaults target `hpcsrv-201.sr009` and `hpcsrv-202.sr009`. The repository and complete model checkpoint must be on NFS accessible at the same absolute paths from Jupyter and both job containers. Jobs need dependency-download access and active InfiniBand; Jupyter must be able to reach the job pod IPs. No custom image build is required.

```bash
cd /home/jovyan/samples/04-vllm-mp-hybrid-lb

# Inspect the plan without reading weights or allocating GPUs.
/usr/bin/python3.12 submit_vllm.py --dry-run

# Launch with the tested defaults; retain the service for 12 hours after readiness.
/usr/bin/python3.12 submit_vllm.py --serve-minutes 720
```

To use another checkpoint or allocation, add `--model-dir /absolute/path/to/checkpoint --nodes NODE0 NODE1`. `--prepare-only` validates weights and creates launch files without allocating GPUs. The submit command prints installation/startup progress and returns after the first verified response; the service continues in cloud jobs. Omitting `--serve-minutes` gives a two-hour lease after readiness.

Replace `ID` below with the exact printed run directory.

```bash
RUN_DIR=/home/jovyan/cloud-vllm-dep-ID
/usr/bin/python3.12 submit_vllm.py --status "$RUN_DIR"
/usr/bin/python3.12 query_vllm.py --run-dir "$RUN_DIR" --prompt "Explain expert parallelism in three sentences." --max-tokens 512
/usr/bin/python3.12 query_vllm.py --run-dir "$RUN_DIR" --node 1 --prompt "What is data parallel inference?" --max-tokens 512

# Enable reasoning for a separate request.
/usr/bin/python3.12 query_vllm.py --run-dir "$RUN_DIR" --prompt "Solve 3*x + 7 = 22." --thinking --max-tokens 2048

# Inspect startup errors or stop the service on both nodes.
/usr/bin/python3.12 submit_vllm.py --diagnose "$RUN_DIR"
/usr/bin/python3.12 submit_vllm.py --stop "$RUN_DIR"
```

### Results and API access

`READY.json` records `base_url`, `openai_base_url`, model name and expiry. Its `endpoints` array contains both node APIs; `--node 0` is the default. The generated `api-key` file supplies Bearer authentication. The run directory is a filesystem location for metadata, not a URL. The API runs at the job pod IP, for example `http://10.x.x.x:8000/v1`; `localhost` in Jupyter refers to Jupyter itself.

Each run keeps `run.json`, `node-*.json`, `status-*.json`, `READY.json`, `install-*.log`, `vllm.log`, `nccl.*.log` and `*-error*.json`. The remote-node log is `vllm-worker.log`; exact commands are saved in `server-command-0.json` and `server-command-1.json`. `STOP` requests shutdown; `stopped-*.json` confirms supervisor exit. Keep `api-key` and generated run snapshots out of version control.

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
| `--dp-rpc-port PORT` | `13345` | DP coordination port, 1024–65535; must differ from the HTTP port. |
| `--served-model-name NAME` | `deepseek-v4.1-flash` | Model name exposed through the OpenAI-compatible API. |
| `--max-model-len TOKENS` | vLLM/model default | Positive total context limit: input plus generated tokens. Omitted for MP unless explicitly set. |
| `--max-num-seqs N` | vLLM/model default | Positive maximum concurrent sequences per DP engine; not the global HTTP concurrency. |
| `--gpu-memory-utilization FRACTION` | vLLM/model default | GPU memory budget fraction per engine; `0 < value ≤ 1` when explicitly set. |
| `--startup-timeout SECONDS` | `3600` | Positive time budget for queueing, environment installation and node preflight. |
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
| `--node 0\|1` | `0` | API node to send the request to; this is not an individual DP engine. |
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
| `--data-parallel-size`, `--data-parallel-size-local` | `16`, `8` | Sixteen DP engines, eight per node; TP=1 and PP=1 use vLLM defaults. |
| `--data-parallel-address`, `--data-parallel-rpc-port` | head pod IP, `13345` | Shared DP coordination address; RPC port is configurable. |
| `--mm-encoder-tp-mode` | `data` | Use data mode for the multimodal encoder. |
| Frontend environment | `VLLM_USE_RUST_FRONTEND=1` | Stock Rust frontend; no runtime source patches. |
| Optional resource flags | `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization` | Passed only when explicitly supplied; otherwise retain vLLM/model defaults. |
| `--data-parallel-hybrid-lb` | enabled on both nodes | Each API balances requests among its eight local DP engines. |
| `--data-parallel-start-rank` | `0` / `8` | Node 0 owns DP ranks 0–7; node 1 owns ranks 8–15. Both expose HTTP APIs. |

### Worker entrypoints

The submit script starts these automatically inside jobs; they are not additional getting-started steps. All accept `-h` / `--help`.

| Script | Arguments | Description |
| --- | --- | --- |
| `bootstrap_vllm.py` | `--run-dir PATH --rank RANK` | Required: run snapshot; rank `0`, `1` or `auto` (MPI environment). |
| `vllm_node.py` | `--run-dir PATH --rank 0\|1` | Required: run snapshot and node rank. |
| `diagnose_vllm.py` | `--run-dir PATH` | Required: existing run; prints saved diagnostics without GPU jobs. |

### Local checks

Run from this directory without GPU jobs or model weights. Tests mock cloud/runtime operations; they do not replace a GPU inference run.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover -s tests -v
```

## Русский

### Что даёт пример

Поднимает DeepSeek V4.1 Flash (1017) на двух узлах с 16 H200 через штатный multiprocessing vLLM: **TP1 / DP16 / EP16 / PP1**, по восемь DP-движков на узел. MoE-эксперты распределены между всеми 16 GPU. Используются штатный vLLM 0.32.0 и Rust-фронтенд, без патчей исходного кода runtime и без Ray.

**Гибридная балансировка:** на каждом узле работает HTTP API, распределяющий запросы между восемью локальными DP-движками. Пример предоставляет два адреса API; внешний HTTP-балансировщик в него не входит. Адрес выбирает клиент или внешний балансировщик. Готовность подтверждается сгенерированным ответом от обоих API и наличием канала NCCL InfiniBand.

Без явно заданных параметров лаунчер сохраняет значения vLLM для контекста, числа последовательностей, памяти, CUDA graphs и бэкендов модели. Для проверенного checkpoint в логе указан максимальный контекст **1 048 576 токенов**; это лимит модели, а не гарантия достаточного KV-кэша для любой нагрузки. Контекст включает вход и выход.

### Состав

- `submit_vllm.py`, `query_vllm.py`: запуск/управление сервисом и авторизованный запрос.
- `bootstrap_vllm.py`, `vllm_node.py`: установка изолированного runtime, проверка узлов/GPU и управление vLLM.
- `vllm_common.py`, `cluster_common.py`: проверка весов, команда сервера, окружение, HTTP и задачи Cloud.ru.
- `diagnose_vllm.py`: извлечение сохранённых ошибок запуска/воркеров, в том числе перемешанных traceback.
- `requirements-vllm.txt`: версии runtime; `tests/test_dep.py`: локальные тесты запуска и жизненного цикла без облака.

### Запуск

Используйте управляющий Jupyter Cloud.ru SR009 на образе **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`**. Именно в этом образе доступен **`/usr/bin/python3.12`** и установлен платформенный `client_lib`; нужны два разных узла по восемь H200. По умолчанию указаны `hpcsrv-201.sr009` и `hpcsrv-202.sr009`. Репозиторий и полный checkpoint должны лежать на NFS по одинаковым абсолютным путям в Jupyter и обоих контейнерах задач. Задачам нужны доступ к зависимостям и активный InfiniBand; Jupyter должен иметь сетевой доступ к IP подов задач. Собирать собственный образ не требуется.

```bash
cd /home/jovyan/samples/04-vllm-mp-hybrid-lb

# Посмотреть план без чтения весов и выделения GPU.
/usr/bin/python3.12 submit_vllm.py --dry-run

# Запустить с проверенными настройками; оставить сервис на 12 часов после готовности.
/usr/bin/python3.12 submit_vllm.py --serve-minutes 720
```

Для другого checkpoint или набора узлов добавьте `--model-dir /absolute/path/to/checkpoint --nodes NODE0 NODE1`. `--prepare-only` проверяет веса и создаёт файлы запуска без выделения GPU. Команда выводит прогресс установки/запуска и завершается после первого проверенного ответа; сервис продолжает работать в облачных задачах. Без `--serve-minutes` время работы после готовности составляет два часа.

Ниже замените `ID` точным каталогом, напечатанным командой запуска.

```bash
RUN_DIR=/home/jovyan/cloud-vllm-dep-ID
/usr/bin/python3.12 submit_vllm.py --status "$RUN_DIR"
/usr/bin/python3.12 query_vllm.py --run-dir "$RUN_DIR" --prompt "Explain expert parallelism in three sentences." --max-tokens 512
/usr/bin/python3.12 query_vllm.py --run-dir "$RUN_DIR" --node 1 --prompt "What is data parallel inference?" --max-tokens 512

# Включить рассуждения для отдельного запроса.
/usr/bin/python3.12 query_vllm.py --run-dir "$RUN_DIR" --prompt "Solve 3*x + 7 = 22." --thinking --max-tokens 2048

# Посмотреть ошибки запуска или остановить сервис на обоих узлах.
/usr/bin/python3.12 submit_vllm.py --diagnose "$RUN_DIR"
/usr/bin/python3.12 submit_vllm.py --stop "$RUN_DIR"
```

### Результаты и доступ к API

В `READY.json` записываются `base_url`, `openai_base_url`, имя модели и время окончания работы. Массив `endpoints` содержит API обоих узлов; по умолчанию используется `--node 0`. Созданный файл `api-key` используется для Bearer-авторизации. Каталог запуска — путь к метаданным на диске, а не URL. API работает по IP пода задачи, например `http://10.x.x.x:8000/v1`; `localhost` в Jupyter указывает на сам Jupyter.

Для каждого запуска сохраняются `run.json`, `node-*.json`, `status-*.json`, `READY.json`, `install-*.log`, `vllm.log`, `nccl.*.log` и `*-error*.json`. Лог второго узла — `vllm-worker.log`; точные команды сохраняются в `server-command-0.json` и `server-command-1.json`. `STOP` запрашивает остановку; `stopped-*.json` подтверждают завершение управляющих процессов. Не добавляйте `api-key` и каталоги запусков в систему контроля версий.

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
| `--dp-rpc-port PORT` | `13345` | Порт координации DP, 1024–65535; должен отличаться от порта HTTP. |
| `--served-model-name NAME` | `deepseek-v4.1-flash` | Имя модели в OpenAI-совместимом API. |
| `--max-model-len TOKENS` | По умолчанию в vLLM/модели | Положительный лимит полного контекста: вход плюс генерируемые токены. В MP не передаётся без явного указания. |
| `--max-num-seqs N` | По умолчанию в vLLM/модели | Положительный предел одновременных последовательностей на DP-движок; не глобальный лимит HTTP-запросов. |
| `--gpu-memory-utilization FRACTION` | По умолчанию в vLLM/модели | Доля памяти GPU для движка; при явном указании `0 < значение ≤ 1`. |
| `--startup-timeout SECONDS` | `3600` | Положительный лимит времени на очередь, установку окружения и предварительные проверки узлов. |
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
| `--node 0\|1` | `0` | Узел с API для отправки запроса; это не номер отдельного DP-движка. |
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
| `--data-parallel-size`, `--data-parallel-size-local` | `16`, `8` | Шестнадцать DP-движков, по восемь на узел; TP=1 и PP=1 берутся из значений по умолчанию vLLM. |
| `--data-parallel-address`, `--data-parallel-rpc-port` | IP головного пода, `13345` | Общий адрес координации DP; порт RPC настраивается. |
| `--mm-encoder-tp-mode` | `data` | Режим data для мультимодального энкодера. |
| Frontend environment | `VLLM_USE_RUST_FRONTEND=1` | Штатный Rust-фронтенд; без патчей исходного кода runtime. |
| Optional resource flags | `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization` | Передаются только при явном указании; иначе сохраняются значения по умолчанию vLLM/модели. |
| `--data-parallel-hybrid-lb` | включён на обоих узлах | Каждый API распределяет запросы между восемью локальными DP-движками. |
| `--data-parallel-start-rank` | `0` / `8` | Узел 0 получает DP-ранги 0–7, узел 1 — 8–15. Оба предоставляют HTTP API. |

### Служебные точки входа

Скрипт отправки запускает их автоматически внутри задач; вручную запускать их для начала работы не требуется. Все поддерживают `-h` / `--help`.

| Скрипт | Аргументы | Описание |
| --- | --- | --- |
| `bootstrap_vllm.py` | `--run-dir PATH --rank RANK` | Обязательны: каталог запуска; ранг `0`, `1` или `auto` (из окружения MPI). |
| `vllm_node.py` | `--run-dir PATH --rank 0\|1` | Обязательны: каталог запуска и ранг узла. |
| `diagnose_vllm.py` | `--run-dir PATH` | Обязателен каталог существующего запуска; выводит сохранённую диагностику без GPU-задач. |

### Локальные проверки

Запускаются из этого каталога без GPU-задач и весов модели. Тесты имитируют облако/runtime; они не заменяют запуск инференса на GPU.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover -s tests -v
```
