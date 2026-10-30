# Cloud.ru distributed GPU samples

[English](#english) · [Русский](#русский)

## English

Four self-contained getting-started samples for distributed GPU workloads on **Cloud.ru SR009, two physical nodes, up to 16 NVIDIA H200 GPUs**. The inference samples serve the local **DeepSeek V4.1 Flash (1017)** checkpoint. Each directory includes its own entrypoints, pinned runtime requirements, tests and a bilingual README with all command parameters.

### Choose a sample

| Sample | Topology / resources | API entrypoints | What you get |
| --- | --- | --- | --- |
| [01 · Ray cluster](01-ray-cluster/README.md#english) | Two nodes; 1–8 H200 per node | No model API | Ray GPU scheduling, CUDA, bidirectional object transfer and NCCL AllReduce checks, with a saved report. |
| [02 · vLLM + Ray](02-vllm-ray/README.md#english) | TP8 / DP2 / EP16 / PP1; 16 H200 | One on the head | Ray-managed inference with two DP engines behind one API; tested text-only eager preset with 16K context. |
| [03 · vLLM multiprocessing + internal LB](03-vllm-mp-internal-lb/README.md#english) | TP1 / DP16 / EP16 / PP1; 16 H200 | One on the head | Native multiprocessing and built-in balancing across all 16 DP engines; the second node runs headless. |
| [04 · vLLM multiprocessing + hybrid LB](04-vllm-mp-hybrid-lb/README.md#english) | TP1 / DP16 / EP16 / PP1; 16 H200 | One per node, two total | Each API balances eight local DP engines. A client or external HTTP load balancer selects the API node. |

TP = tensor parallelism, DP = data parallelism, EP = expert parallelism, PP = pipeline parallelism. With EP enabled, experts span the TP × DP group. The two multiprocessing samples use stock vLLM without runtime source patches. Sample 04 does not include an external HTTP balancer; sample 03 provides the single-API setup.

```text
samples/
├── README.md
├── .gitignore
├── 01-ray-cluster/
├── 02-vllm-ray/
├── 03-vllm-mp-internal-lb/
└── 04-vllm-mp-hybrid-lb/
```

### Prerequisites

- Run launchers from Cloud.ru management Jupyter using the **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`** image. This Jupyter image provides **`/usr/bin/python3.12`**, the interpreter used in all commands below, and the platform-provided `client_lib`. There is no root-level dependency installation; Ray/vLLM are installed in isolated environments inside jobs.
- Copy this repository to shared NFS, for example `/home/jovyan/samples`. NFS paths must be identical and accessible in Jupyter and job containers. The run root defaults to `/home/jovyan` and must already exist.
- Supply two distinct available SR009 hosts through `--nodes NODE0 NODE1` if the tested defaults (`hpcsrv-201.sr009`, `hpcsrv-202.sr009`) do not match your allocation. Inference requests eight H200 GPUs on each host (`h200.8gpu.96C.1952G`), using two separately pinned jobs.
- Jobs need package-download access and working InfiniBand. The management Jupyter/client must have network access to the API pod IPs. No public ingress or localhost tunnel is created.
- For inference, place the complete local checkpoint on NFS. The tested default path is:

```text
/home/jovyan/MODELS/deepseek-ai_DeepSeek-V4.1-Flash-1017/deepseek-ai_DeepSeek-V4.1-Flash-1017
```

Use `--model-dir /absolute/path/to/checkpoint` to change it. Launchers validate the config, tokenizer, shard index and weight-file presence before requesting GPUs; they do not download model weights. The repeated directory name above reflects the tested checkpoint layout.

### Getting started

Use sample 01 for an infrastructure check, or sample 03 for inference behind one API. These are separate resource-allocating runs; stop an active sample before reusing the same full GPU allocation.

```bash
cd /home/jovyan/samples/03-vllm-mp-internal-lb

# Review the resource plan and exact vLLM commands without submitting jobs.
/usr/bin/python3.12 submit_vllm.py --dry-run

# Start the service with a 12-hour lifetime after readiness.
/usr/bin/python3.12 submit_vllm.py --serve-minutes 720

# Replace ID with the run directory printed by the launcher.
RUN_DIR=/home/jovyan/cloud-vllm-dep-single-api-ID
/usr/bin/python3.12 query_vllm.py --run-dir "$RUN_DIR" --prompt "Explain expert parallelism in three sentences." --max-tokens 512
/usr/bin/python3.12 submit_vllm.py --status "$RUN_DIR"
/usr/bin/python3.12 submit_vllm.py --diagnose "$RUN_DIR"
/usr/bin/python3.12 submit_vllm.py --stop "$RUN_DIR"
```

Inference launchers print progress, save logs/status and return after a verified generated answer. Cloud jobs keep serving for **120 minutes after readiness by default**; set `--serve-minutes` when launching to change this (`720` = 12 hours, `1440` = 24 hours). Sample 01 stops after its test unless `--keep-alive` is supplied. Platform job/project limits still apply independently of these script lifetimes.

### Run directories and network addresses

Each launch creates an independent directory under `--nfs-root`. Prefixes remain compatible with the tested launchers:

| Sample | Run directory prefix | Success marker |
| --- | --- | --- |
| 01 | `cloud-ray-ID` | `PASSED.json` |
| 02 | `cloud-vllm-ep-ID` | `READY.json` |
| 03 | `cloud-vllm-dep-single-api-ID` | `READY.json` |
| 04 | `cloud-vllm-dep-ID` | `READY.json` with both API endpoints |

These directories contain configuration, launch snapshots, progress, logs and, for inference, an API key. They are **filesystem paths**, not API addresses. Read `openai_base_url` from `READY.json` for an OpenAI-compatible client; it looks like `http://10.x.x.x:8000/v1`. `localhost` inside Jupyter points to Jupyter, not the GPU job. `query_vllm.py` reads the URL and Bearer key automatically. Generated run directories and caches are excluded by `.gitignore`.

### Runtime versions and validation

GPU jobs use the base image `cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42`. Sample 01 pins `ray[default]==2.59.0`; sample 02 pins Ray 2.59.0 and vLLM 0.32.0. Samples 03/04 pin vLLM 0.32.0 without Ray. Inference environments also pin `cuda-toolkit[nvcc,crt,cccl,nvvm,cudart]==13.0.3`. Exact dependencies are in each sample's `requirements-*.txt`.

All four samples were exercised on the target cluster; the inference variants passed sanity requests. Their context/execution presets differ, so this collection does not establish a performance comparison or a long-duration benchmark result. Source code, comments, messages and default prompts are in English; documentation is provided in both languages.

Run the existing local suites separately so each sample loads its own modules. These checks do not allocate cloud GPUs:

```bash
cd /home/jovyan/samples
for sample in 01-ray-cluster 02-vllm-ray 03-vllm-mp-internal-lb 04-vllm-mp-hybrid-lb; do
  (cd "$sample" && PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover -s tests -v) || exit 1
done
```

## Русский

Четыре самостоятельных примера для начала работы с распределёнными GPU-задачами в **Cloud.ru SR009: два физических узла, до 16 NVIDIA H200**. Примеры инференса используют локальный checkpoint **DeepSeek V4.1 Flash (1017)**. В каждом каталоге находятся свои точки входа, зафиксированные зависимости runtime, тесты и двуязычный README с описанием всех параметров команд.

### Выбор примера

| Пример | Топология / ресурсы | Точки входа API | Результат |
| --- | --- | --- | --- |
| [01 · Кластер Ray](01-ray-cluster/README.md#русский) | Два узла; по 1–8 H200 | Без API модели | Проверка GPU-размещения Ray, CUDA, двусторонней передачи объектов и NCCL AllReduce с сохранением отчёта. |
| [02 · vLLM + Ray](02-vllm-ray/README.md#русский) | TP8 / DP2 / EP16 / PP1; 16 H200 | Один на головном узле | Инференс под управлением Ray с двумя DP-движками за одним API; проверенный текстовый eager-пресет с контекстом 16K. |
| [03 · vLLM multiprocessing + internal LB](03-vllm-mp-internal-lb/README.md#русский) | TP1 / DP16 / EP16 / PP1; 16 H200 | Один на головном узле | Штатный multiprocessing и встроенная балансировка между всеми 16 DP-движками; второй узел работает в headless-режиме. |
| [04 · vLLM multiprocessing + hybrid LB](04-vllm-mp-hybrid-lb/README.md#русский) | TP1 / DP16 / EP16 / PP1; 16 H200 | По одному на узел, всего два | Каждый API распределяет запросы между восемью локальными DP-движками. Узел с API выбирает клиент или внешний HTTP-балансировщик. |

TP — тензорный параллелизм, DP — параллелизм данных, EP — параллелизм экспертов, PP — конвейерный параллелизм. При включённом EP эксперты распределены по группе TP × DP. Оба примера multiprocessing используют штатный vLLM без патчей исходного кода runtime. Внешний HTTP-балансировщик в пример 04 не входит; схема с одним API реализована в примере 03.

```text
samples/
├── README.md
├── .gitignore
├── 01-ray-cluster/
├── 02-vllm-ray/
├── 03-vllm-mp-internal-lb/
└── 04-vllm-mp-hybrid-lb/
```

### Требования

- Запускайте лаунчеры из управляющего Jupyter Cloud.ru на образе **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`**. Именно в этом образе Jupyter доступен **`/usr/bin/python3.12`**, используемый во всех командах ниже, и установлен платформенный `client_lib`. Общей установки зависимостей в корне нет: Ray/vLLM устанавливаются в изолированное окружение внутри задач.
- Скопируйте репозиторий на общий NFS, например в `/home/jovyan/samples`. Пути NFS должны совпадать и быть доступны в Jupyter и контейнерах задач. Корень запусков по умолчанию — `/home/jovyan`; каталог должен существовать.
- Если проверенные узлы (`hpcsrv-201.sr009`, `hpcsrv-202.sr009`) не соответствуют вашей квоте, укажите два разных доступных узла SR009 через `--nodes NODE0 NODE1`. Инференс запрашивает по восемь H200 на каждом (`h200.8gpu.96C.1952G`), отправляя две отдельно закреплённые задачи.
- Задачам нужны доступ к загрузке пакетов и рабочий InfiniBand. Управляющий Jupyter/клиент должен иметь сетевой доступ к IP подов с API. Публичный ingress и туннель на localhost не создаются.
- Для инференса разместите полный локальный checkpoint на NFS. Проверенный путь по умолчанию:

```text
/home/jovyan/MODELS/deepseek-ai_DeepSeek-V4.1-Flash-1017/deepseek-ai_DeepSeek-V4.1-Flash-1017
```

Другой путь задаётся через `--model-dir /absolute/path/to/checkpoint`. До запроса GPU лаунчеры проверяют конфигурацию, токенизатор, индекс шардов и наличие файлов весов; сами веса они не скачивают. Повтор имени каталога выше соответствует проверенной структуре checkpoint.

### Начало работы

Используйте пример 01 для проверки инфраструктуры или пример 03 для инференса с одним API. Каждый запуск отдельно выделяет ресурсы; перед повторным использованием всех GPU остановите активный пример.

```bash
cd /home/jovyan/samples/03-vllm-mp-internal-lb

# Проверить план ресурсов и точные команды vLLM без отправки задач.
/usr/bin/python3.12 submit_vllm.py --dry-run

# Запустить сервис на 12 часов после готовности.
/usr/bin/python3.12 submit_vllm.py --serve-minutes 720

# Заменить ID каталогом запуска, напечатанным лаунчером.
RUN_DIR=/home/jovyan/cloud-vllm-dep-single-api-ID
/usr/bin/python3.12 query_vllm.py --run-dir "$RUN_DIR" --prompt "Explain expert parallelism in three sentences." --max-tokens 512
/usr/bin/python3.12 submit_vllm.py --status "$RUN_DIR"
/usr/bin/python3.12 submit_vllm.py --diagnose "$RUN_DIR"
/usr/bin/python3.12 submit_vllm.py --stop "$RUN_DIR"
```

Лаунчеры инференса выводят прогресс, сохраняют логи/статус и завершаются после проверенного сгенерированного ответа. Облачные задачи продолжают обслуживать запросы **120 минут после готовности по умолчанию**; время меняется при запуске через `--serve-minutes` (`720` = 12 часов, `1440` = 24 часа). Пример 01 останавливается после теста, если не указан `--keep-alive`. Ограничения платформы на задачи/проект действуют независимо от этих лимитов скриптов.

### Каталоги запусков и сетевые адреса

Каждый запуск создаёт отдельный каталог внутри `--nfs-root`. Префиксы сохранены совместимыми с проверенными лаунчерами:

| Пример | Префикс каталога запуска | Признак успеха |
| --- | --- | --- |
| 01 | `cloud-ray-ID` | `PASSED.json` |
| 02 | `cloud-vllm-ep-ID` | `READY.json` |
| 03 | `cloud-vllm-dep-single-api-ID` | `READY.json` |
| 04 | `cloud-vllm-dep-ID` | `READY.json` с адресами обоих API |

В них находятся конфигурация, копии файлов запуска, прогресс, логи и API-ключ для инференса. Это **пути в файловой системе**, а не адреса API. Для OpenAI-совместимого клиента берите `openai_base_url` из `READY.json`: например, `http://10.x.x.x:8000/v1`. `localhost` в Jupyter указывает на сам Jupyter, а не на GPU-задачу. `query_vllm.py` автоматически читает URL и Bearer-ключ. Созданные каталоги запусков и кэши исключены через `.gitignore`.

### Версии runtime и проверки

GPU-задачи используют готовый базовый образ `cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42`. В примере 01 зафиксирован `ray[default]==2.59.0`; в примере 02 — Ray 2.59.0 и vLLM 0.32.0. В примерах 03/04 — vLLM 0.32.0 без Ray. Окружения инференса также фиксируют `cuda-toolkit[nvcc,crt,cccl,nvvm,cudart]==13.0.3`. Точные зависимости находятся в `requirements-*.txt` соответствующих каталогов.

Все четыре примера были проверены на целевом кластере; варианты инференса прошли sanity-запросы. Их настройки контекста/выполнения различаются, поэтому этот набор не является сравнением производительности или результатом длительного нагрузочного теста. Исходный код, комментарии, сообщения и промпты по умолчанию написаны на английском; документация доступна на двух языках.

Запускайте существующие локальные тесты отдельно для каждого примера, чтобы загружались его собственные модули. Эти проверки не выделяют GPU в облаке:

```bash
cd /home/jovyan/samples
for sample in 01-ray-cluster 02-vllm-ray 03-vllm-mp-internal-lb 04-vllm-mp-hybrid-lb; do
  (cd "$sample" && PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover -s tests -v) || exit 1
done
```
