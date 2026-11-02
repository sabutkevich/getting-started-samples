# DeepSeek V4.1 Flash with multiprocessing and hybrid LB

[English](#english) · [Русский](#русский)

## English

When TP/PP replicas fit locally, every node exposes an API balancing its local DP engines. When a replica spans nodes, only its leading node exposes an API. No external HTTP load balancer is included. With local DP=1, vLLM automatically selects external DP mode: each API owns one engine and clients select an endpoint.

Node/GPU counts, DP, TP, PP and EP enablement are configurable. Defaults preserve the earlier topology: TP1 / DP16 / EP16 / PP1.

### Requirements

Run from the management Jupyter image **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`**, which provides **`/usr/bin/python3.12`** and the platform `client_lib`. Copy the sample onto shared NFS, for example `/home/jovyan/samples`; absolute paths must match in Jupyter and worker containers. The run root must already exist.

Workers use **`cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42`**. Cloud.ru selects physical hosts: one binary job with `n_workers=N` and `processes_per_worker=1`, without `include_nodes`/`exclude_nodes`. Node ranks come from MPI and addresses are discovered at launch. Resource shape: `h200.Ggpu.(12*G)C.(244*G)G`. The platform enforces available shapes, allocation limits and queue permissions. Rejected submissions fail visibly; there are no automatic submission retries.

Default checkpoint:

```text
/home/jovyan/MODELS/deepseek-ai_DeepSeek-V4.1-Flash-1017/deepseek-ai_DeepSeek-V4.1-Flash-1017
```

Package-download access and InfiniBand are required for multiple nodes. Clients must reach API pod IPs; Jupyter localhost is not the job API address.

### Included files

- Launcher, bootstrap and per-node supervisors; `cluster_common.py` and `topology.py` handle placement, discovery and validation.
- `requirements-*.txt` pins runtime; `tests/` checks plans and lifecycle without cloud GPUs.
- `vllm_common.py` builds model commands; `query_vllm.py` sends requests. Logs, errors and readiness are saved on NFS.

### Launch

```bash
cd /home/jovyan/samples/04-vllm-mp-hybrid-lb

/usr/bin/python3.12 submit_vllm.py --num-nodes 3 --gpus-per-node 8 --tp 1 --dry-run
/usr/bin/python3.12 submit_vllm.py --num-nodes 3 --gpus-per-node 8 --tp 1 --serve-minutes 720
```

Example: ten nodes, TP2, DP40, EP80:

```bash
/usr/bin/python3.12 submit_vllm.py \
  --num-nodes 10 --gpus-per-node 8 \
  --tp 2 --dp 40 --ep-size 80 \
  --allocation-name alloc-huawei-rnd-sr009 --queue-name shared \
  --startup-timeout 7200 --serve-minutes 720 --dry-run
```

Remove `--dry-run` after reviewing the plan to submit it. EP-off example: `--num-nodes 2 --tp 8 --dp 2 --no-enable-expert-parallel --ep-size 1`. For spanning TP/PP, inspect `--num-nodes 4 --tp 8 --pp 2 --dp 2 --dry-run`; model compatibility still needs GPU validation.

### Topology rules

- Each of N workers requests the same 1–8 H200 GPUs. N is any positive count accepted by the platform.
- `DP * TP * PP = N * GPUs_per_node`. DP is calculated by default; an explicit DP must use all requested GPUs.
- With EP enabled: `EP = TP * DP`; with EP disabled its group size is 1. vLLM 0.32.0 does not expose an independent arbitrary EP size. `--ep-size` asserts the expected result.
- A TP*PP replica must fit uniformly within a worker or span a whole number of complete workers. Local DP is calculated automatically.
- Topology arithmetic and available checkpoint geometry, including attention/o_groups TP divisibility, are checked before submission. Uneven expert placement is permitted without requiring experts % EP == 0.
- Planning checks cannot guarantee GPU memory capacity or model-kernel support for every combination. Runtime remains stock, without patches. New cluster sizes and parallelism combinations need GPU validation.

[Official EP documentation](https://docs.vllm.ai/en/v0.32.0/serving/expert_parallel_deployment/) · [DP and balancing](https://docs.vllm.ai/en/v0.32.0/serving/data_parallel_deployment/).

### All launcher parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--num-nodes N` | `2` | Positive worker count; automatic platform placement. |
| `--gpus-per-node N` | `8` | Uniform GPU count per worker, 1..8. |
| `--allocation-name NAME` | `workspace default` | Target training allocation; no physical host names. |
| `--queue-name NAME` | `shared` | Target queue; shared is selected explicitly by default. |
| `--tensor-parallel-size N / --tp N` | `1` | Tensor parallel size for one DP replica. |
| `--data-parallel-size N / --dp N` | `automatic` | Total GPUs / (TP*PP); an explicit value must satisfy the same equality. |
| `--pipeline-parallel-size N / --pp N` | `1` | Pipeline parallel stages. |
| `--data-parallel-size-local N` | `automatic` | Optional assertion of local DP: GPUs/(TP*PP), or 1 for a spanning replica. |
| `--enable-expert-parallel / --no-enable-expert-parallel` | `enabled` | Enable or disable EP; with EP enabled its size is TP*DP. |
| `--expert-parallel-size N / --ep-size N` | `automatic` | Optional assertion of the derived EP size; 1 when EP is disabled. |
| `--model-dir PATH` | `see above` | Complete local DeepSeek V4.1 Flash checkpoint; never downloaded by the launcher. |
| `--image IMAGE` | `cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42` | Existing worker image; full cr.ai.cloud.ru tag/digest. |
| `--nfs-root PATH` | `/home/jovyan` | Existing shared directory for run snapshots, keys, logs and status. |
| `--api-port PORT` | `8000` | API HTTP port in job containers. |
| `--socket-ifname NAME` | `eth0` | Pod TCP/bootstrap interface; does not select an InfiniBand HCA. |
| `--served-model-name NAME` | `deepseek-v4.1-flash` | Model identifier exposed by the API. |
| `--dp-rpc-port PORT` | `13345` | DP coordinator/RPC port; distinct from API and master ports. |
| `--master-port PORT` | `29501` | Torch rendezvous port for a TP/PP replica spanning workers. |
| `--max-model-len N` | `vLLM/model default` | Maximum input+output context; runtime/model limits still apply. |
| `--max-num-seqs N` | `vLLM default` | Maximum concurrent sequences per DP rank. |
| `--gpu-memory-utilization FLOAT` | `vLLM default` | GPU memory fraction; does not guarantee that weights/KV cache fit. |
| `--startup-timeout SECONDS` | `3600` | Queue/install/node discovery budget; also used for preflight waits. |
| `--model-timeout SECONDS` | `3600` | Model loading, JIT and generated-answer readiness budget. |
| `--serve-minutes MINUTES` | `120` | Serving lease after READY; 720=12h, 1440=24h. |
| `--dry-run` | `off` | Print resource plan and exact commands; no model reads, files, SDK imports or jobs. |
| `--prepare-only` | `off` | Validate checkpoint and write a launch snapshot without requesting GPUs. |
| `--status RUN_DIR` | `—` | Read saved service, platform and all node status files. |
| `--diagnose RUN_DIR` | `—` | Read saved logs and errors; no platform/GPU work. |
| `--stop RUN_DIR` | `—` | Write STOP for every supervisor; verify platform jobs actually finish. |
| `-h / --help` | `—` | Show usage and exit. |

### Results and stopping

Each launch creates a separate directory and snapshots its source. `run.json` stores configuration, `platform-job.json` the job ID/submission reply, and `node-RANK.json`/`stopped-RANK.json` record discovered workers and shutdown. Expect records for all N node ranks. A rejected submission may create no platform job.

`READY.json` follows actual generation and contains topology, `openai_base_url`, `endpoints`, `endpoint_nodes` (MP), lease expiry and the key-file path. The key is in `api-key`, not the report. `--serve-minutes` starts after readiness; returning from the launcher leaves serving GPU processes running.

```bash
RUN_DIR=/home/jovyan/cloud-vllm-dep-ID
/usr/bin/python3.12 query_vllm.py --run-dir "$RUN_DIR" --prompt "Explain expert parallelism." --max-tokens 512
/usr/bin/python3.12 submit_vllm.py --status "$RUN_DIR"
/usr/bin/python3.12 submit_vllm.py --diagnose "$RUN_DIR"
/usr/bin/python3.12 submit_vllm.py --stop "$RUN_DIR"
```

| Parameter | Default | Description |
| --- | --- | --- |
| `--run-dir PATH` | `required` | Existing READY run directory. |
| `--node RANK` | `0` | Physical worker rank exposing an API; see READY.endpoint_nodes. |
| `--prompt TEXT` | `Briefly explain tensor parallelism.` | English user request text. |
| `--max-tokens N` | `512` | Maximum response tokens. |
| `--timeout SECONDS` | `300` | HTTP request timeout. |
| `--thinking` | `off` | Enable model reasoning. |
| `-h / --help` | `—` | Show help. |

**Stop the service when finished and verify that the Cloud.ru binary job actually ends.** STOP is an NFS request; if supervisors hang, stop the platform job using the ID from `platform-job.json`. Ctrl+C while the launcher is active invokes cleanup. Check every `stopped-RANK.json` and a terminal platform status; Terminating does not yet confirm resource release.

### Versions and validation

vLLM 0.32.0, CUDA toolkit 13.0.3. The original two-node preset was exercised on GPUs. The generalized version has local plan/lifecycle tests; this does not validate every topology on hardware. All source comments and messages are in English.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover -s tests -v
```

## Русский

При локальных TP/PP-репликах API есть на каждом узле и распределяет запросы между его локальными DP-движками. Если реплика занимает несколько узлов, API находится только на её ведущем узле. Внешний HTTP-балансировщик не включён. При локальном DP=1 vLLM автоматически переходит в режим внешнего DP: каждый API обслуживает один движок, а клиент выбирает endpoint.

Число узлов, GPU, DP, TP, PP и включение EP настраиваются. Значения по умолчанию сохраняют прежнюю схему: TP1 / DP16 / EP16 / PP1.

### Требования

Запускайте из управляющего Jupyter на образе **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`**: именно там доступен **`/usr/bin/python3.12`** и платформенный `client_lib`. Скопируйте каталог примера в общий NFS, например `/home/jovyan/samples`; абсолютные пути должны совпадать в Jupyter и контейнерах задач. Корень запусков должен существовать.

Воркеры используют **`cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42`**. Физические серверы выбирает Cloud.ru: одна binary-задача, `n_workers=N`, `processes_per_worker=1`, без `include_nodes`/`exclude_nodes`. Ранг берётся из MPI; адреса обнаруживаются при запуске. Тип ресурсов — `h200.Ggpu.(12*G)C.(244*G)G`. Доступность типов ресурсов, лимиты аллокации и права очереди проверяет платформа. Ошибка `submit()` выводится сразу; запросы автоматически не повторяются.

Checkpoint по умолчанию:

```text
/home/jovyan/MODELS/deepseek-ai_DeepSeek-V4.1-Flash-1017/deepseek-ai_DeepSeek-V4.1-Flash-1017
```

Нужны доступ к загрузке пакетов и InfiniBand для нескольких узлов. Клиент должен видеть IP подов с API; localhost в Jupyter не является адресом API задачи.

### Состав

- Лаунчер, bootstrap и управляющий процесс на каждый узел; `cluster_common.py` и `topology.py` для размещения, обнаружения и проверок.
- `requirements-*.txt` фиксирует runtime; `tests/` проверяет планы и жизненный цикл без облачных GPU.
- `vllm_common.py` формирует команды модели; `query_vllm.py` отправляет запросы. Логи, ошибки и готовность сохраняются на NFS.

### Запуск

```bash
cd /home/jovyan/samples/04-vllm-mp-hybrid-lb

/usr/bin/python3.12 submit_vllm.py --num-nodes 3 --gpus-per-node 8 --tp 1 --dry-run
/usr/bin/python3.12 submit_vllm.py --num-nodes 3 --gpus-per-node 8 --tp 1 --serve-minutes 720
```

Пример десяти узлов, TP2, DP40, EP80:

```bash
/usr/bin/python3.12 submit_vllm.py \
  --num-nodes 10 --gpus-per-node 8 \
  --tp 2 --dp 40 --ep-size 80 \
  --allocation-name alloc-huawei-rnd-sr009 --queue-name shared \
  --startup-timeout 7200 --serve-minutes 720 --dry-run
```

После проверки плана уберите `--dry-run` для отправки. Пример с выключенным EP: `--num-nodes 2 --tp 8 --dp 2 --no-enable-expert-parallel --ep-size 1`. Для TP/PP на нескольких узлах можно проверить план `--num-nodes 4 --tp 8 --pp 2 --dp 2 --dry-run`; совместимость модели должна быть проверена на GPU.

### Правила топологии

- Выделяются одинаковые 1–8 H200 на каждом из N узлов. Число узлов — любое положительное в пределах платформы.
- `DP * TP * PP = N * GPU_per_node`. DP по умолчанию вычисляется; явно заданный DP должен использовать все запрошенные GPU.
- При включённом EP: `EP = TP * DP`; при выключенном EP размер группы EP равен 1. Независимый произвольный EP vLLM 0.32.0 не предоставляет. `--ep-size` проверяет ожидаемый размер.
- Реплика TP*PP должна целиком помещаться в узле с равномерным размещением либо занимать целое число полных узлов. Локальный DP вычисляется автоматически.
- Перед отправкой проверяются арифметика топологии и доступная геометрия checkpoint, включая TP для attention/o_groups. Разрешённое размещение экспертов не требует делимости числа экспертов на EP.
- Проверки планирования не гарантируют достаточную память или поддержку каждой комбинации ядрами модели. Runtime остаётся штатным, без патчей. Новые размеры кластера и комбинации нужно проверить на GPU.

[Официальная документация EP](https://docs.vllm.ai/en/v0.32.0/serving/expert_parallel_deployment/) · [DP и балансировка](https://docs.vllm.ai/en/v0.32.0/serving/data_parallel_deployment/).

### Все параметры лаунчера

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `--num-nodes N` | `2` | Положительное число воркеров; размещение выбирает платформа. |
| `--gpus-per-node N` | `8` | Одинаковое число GPU на каждом воркере, 1..8. |
| `--allocation-name NAME` | `workspace default` | Целевая аллокация задач обучения; имена физических узлов не нужны. |
| `--queue-name NAME` | `shared` | Целевая очередь; по умолчанию явно выбирается shared. |
| `--tensor-parallel-size N / --tp N` | `1` | Размер тензорного параллелизма одной DP-реплики. |
| `--data-parallel-size N / --dp N` | `automatic` | Всего GPU / (TP*PP); явно заданное значение должно удовлетворять тому же равенству. |
| `--pipeline-parallel-size N / --pp N` | `1` | Число стадий конвейерного параллелизма. |
| `--data-parallel-size-local N` | `automatic` | Необязательная проверка локального DP: GPU/(TP*PP) либо 1 для реплики на нескольких узлах. |
| `--enable-expert-parallel / --no-enable-expert-parallel` | `enabled` | Включить или выключить EP; при включении размер EP равен TP*DP. |
| `--expert-parallel-size N / --ep-size N` | `automatic` | Необязательная проверка вычисленного размера EP; при выключенном EP равен 1. |
| `--model-dir PATH` | `see above` | Полный локальный checkpoint DeepSeek V4.1 Flash; лаунчер веса не скачивает. |
| `--image IMAGE` | `cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42` | Готовый образ воркеров; полный адрес cr.ai.cloud.ru с тегом/digest. |
| `--nfs-root PATH` | `/home/jovyan` | Существующий общий каталог для копий запуска, ключей, логов и статусов. |
| `--api-port PORT` | `8000` | Порт HTTP API в контейнерах задач. |
| `--socket-ifname NAME` | `eth0` | TCP-интерфейс пода для инициализации; не выбирает InfiniBand HCA. |
| `--served-model-name NAME` | `deepseek-v4.1-flash` | Идентификатор модели в API. |
| `--dp-rpc-port PORT` | `13345` | Порт координатора/RPC DP; отличается от API и master-порта. |
| `--master-port PORT` | `29501` | Порт Torch rendezvous для TP/PP-реплики на нескольких узлах. |
| `--max-model-len N` | `vLLM/model default` | Максимальный контекст вход+выход; ограничения runtime/модели сохраняются. |
| `--max-num-seqs N` | `vLLM default` | Максимальное число одновременных последовательностей на DP-ранг. |
| `--gpu-memory-utilization FLOAT` | `vLLM default` | Доля памяти GPU; не гарантирует, что веса и KV-кеш поместятся. |
| `--startup-timeout SECONDS` | `3600` | Время на очередь, установку и обнаружение узлов; также используется при проверках. |
| `--model-timeout SECONDS` | `3600` | Время загрузки модели, JIT и проверки готовности генерацией ответа. |
| `--serve-minutes MINUTES` | `120` | Срок обслуживания после READY; 720=12ч, 1440=24ч. |
| `--dry-run` | `off` | Вывести план ресурсов и команды; без чтения модели, файлов, импорта SDK и задач. |
| `--prepare-only` | `off` | Проверить checkpoint и записать копию запуска без запроса GPU. |
| `--status RUN_DIR` | `—` | Прочитать сохранённые статусы сервиса, платформы и всех узлов. |
| `--diagnose RUN_DIR` | `—` | Прочитать логи и ошибки; без запросов платформе/GPU. |
| `--stop RUN_DIR` | `—` | Записать STOP для всех управляющих процессов; проверить фактическое завершение задач. |
| `-h / --help` | `—` | Показать справку и выйти. |

### Результаты и остановка

Каждый запуск создаёт отдельный каталог и копирует туда исходники. `run.json` хранит конфигурацию, `platform-job.json` — ID/ответ отправки, `node-RANK.json` и `stopped-RANK.json` — фактические узлы и завершение. Нужны записи для всех N рангов. При ошибке отправки задача может не существовать на платформе.

`READY.json` создаётся после реальной генерации; содержит topology, `openai_base_url`, `endpoints`, `endpoint_nodes` (для MP), срок службы и путь к ключу. Сам ключ находится в `api-key`, не в отчёте. `--serve-minutes` отсчитывается после готовности; после возврата лаунчера модель продолжает занимать GPU.

```bash
RUN_DIR=/home/jovyan/cloud-vllm-dep-ID
/usr/bin/python3.12 query_vllm.py --run-dir "$RUN_DIR" --prompt "Explain expert parallelism." --max-tokens 512
/usr/bin/python3.12 submit_vllm.py --status "$RUN_DIR"
/usr/bin/python3.12 submit_vllm.py --diagnose "$RUN_DIR"
/usr/bin/python3.12 submit_vllm.py --stop "$RUN_DIR"
```

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `--run-dir PATH` | `required` | Каталог запуска с READY. |
| `--node RANK` | `0` | Ранг воркера с API; допустимые значения в READY.endpoint_nodes. |
| `--prompt TEXT` | `Briefly explain tensor parallelism.` | Текст пользовательского запроса. |
| `--max-tokens N` | `512` | Максимум токенов ответа. |
| `--timeout SECONDS` | `300` | Таймаут HTTP-запроса. |
| `--thinking` | `off` | Включить рассуждение модели. |
| `-h / --help` | `—` | Показать справку. |

**После работы обязательно остановите сервис и проверьте завершение binary-задачи в Cloud.ru.** STOP — запрос через NFS; если воркеры зависли, остановите задачу платформы по ID из `platform-job.json`. Ctrl+C во время активного лаунчера включает его очистку. Проверьте `stopped-RANK.json` для всех узлов и конечный статус платформы; Terminating ещё не означает освобождение ресурсов.

### Версии и проверка

vLLM 0.32.0, CUDA toolkit 13.0.3. Исходный пресет на двух узлах ранее проверен на GPU. Новая параметризованная версия проверяется локальными тестами; это не подтверждение работоспособности всех топологий на оборудовании. Все комментарии и сообщения исходного кода — на английском.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover -s tests -v
```

