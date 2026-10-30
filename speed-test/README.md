# NCCL Ring AllReduce benchmark for Cloud.ru SR009

[English](#english) · [Русский](#русский) · [Script / Скрипт](ring_allreduce.py)

## English

### What this tool provides

[`ring_allreduce.py`](ring_allreduce.py) measures NCCL Ring AllReduce performance on NVIDIA H200 GPUs in Cloud.ru SR009. It supports one or more physical nodes and **1–8 GPUs per node**, including different GPU counts on different nodes. No model checkpoint is needed.

The launcher submits one separately pinned binary job per physical node. Each job starts one process per allocated GPU; all processes join the same NCCL group using a fresh rendezvous file on shared NFS. The total process count, `world_size`, is the sum of the GPU counts. Global ranks are contiguous: for `8 + 4 + 2` GPUs, the nodes receive ranks `0–7`, `8–11` and `12–13`.

The tool checks visible GPU counts and physical-host identities, verifies AllReduce results, prints progress and saves intermediate measurements. The final reports contain timing, algorithm bandwidth and collective-equivalent bus bandwidth for every buffer size.

### Included files

- [`ring_allreduce.py`](ring_allreduce.py): a self-contained launcher and worker implementation. Copying this single file is sufficient for a cloud run.
- [`tests/test_ring_allreduce.py`](tests/test_ring_allreduce.py): local tests for topology, job plans, snapshots, bandwidth calculations, worker logic and cleanup; not required for a cloud run.

Every run copies the script to its own NFS directory as `worker.py`; generated workers and launch scripts do not need to be prepared manually.

### Requirements

- Use the Cloud.ru management Jupyter image **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`**. This image provides **`/usr/bin/python3.12`**, used in the commands below, and the platform-provided `client_lib`.
- Copy `ring_allreduce.py` to shared NFS, for example `/home/jovyan/ring_allreduce.py`. The script and run directory must be accessible at identical absolute paths from Jupyter and all job containers. The run root must already exist and be writable.
- Choose distinct physical hosts available in your SR009 allocation. The script accepts 1–8 GPUs per host; the corresponding resource shape must be available in your project. For `N` GPUs, it requests `h200.Ngpu.(12*N)C.(244*N)G`, for example `h200.8gpu.96C.1952G` for eight GPUs.
- GPU jobs use **`cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42`** by default. This is the worker image, separate from the management Jupyter image. It supplies PyTorch/CUDA; the script does not install packages or create virtual environments.
- The selected hosts must have working GPU drivers and InfiniBand. NCCL uses IPv4 for network bootstrap and `IB` as its network backend. Ray and vLLM are not used.

### Run

Run commands from the directory containing the script. Replace node names with hosts from your allocation.

```bash
cd /home/jovyan

# Inspect the job requests and global GPU ranks without submitting jobs.
/usr/bin/python3.12 ring_allreduce.py \
  --nodes hpcsrv-201.sr009 hpcsrv-202.sr009 \
  --gpus-per-node 8 \
  --dry-run

# Run on two nodes with eight GPUs each; allow up to 60 minutes including queueing.
/usr/bin/python3.12 ring_allreduce.py \
  --nodes hpcsrv-201.sr009 hpcsrv-202.sr009 \
  --gpus-per-node 8 \
  --timeout-minutes 60
```

One GPU count applies to every node. To assign different counts, provide one value per node in the same order:

```bash
# Replace NODE0, NODE1 and NODE2 with three distinct physical host names.
/usr/bin/python3.12 ring_allreduce.py \
  --nodes NODE0 NODE1 NODE2 \
  --gpus-per-node 8 4 2 \
  --timeout-minutes 60
```

A single node is also supported. This example permits local P2P/NVLink and shared-memory transports:

```bash
/usr/bin/python3.12 ring_allreduce.py \
  --nodes hpcsrv-201.sr009 \
  --gpus-per-node 8 \
  --allow-local-transports
```

For interactive configuration, omit `--nodes`. The script prints `client_lib.get_nodes()`, asks for the number and names of physical nodes, then asks for the GPU count on each node unless `--gpus-per-node` was supplied:

```bash
/usr/bin/python3.12 ring_allreduce.py
```

To change the workload, set buffer sizes, repetitions and time limits explicitly:

```bash
/usr/bin/python3.12 ring_allreduce.py \
  --nodes hpcsrv-201.sr009 hpcsrv-202.sr009 \
  --gpus-per-node 4 \
  --sizes-mib 64 256 1024 \
  --warmup 10 \
  --iterations 100 \
  --trials 5 \
  --timeout-minutes 60 \
  --collective-timeout 1800
```

### Launch parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--nodes NODE [NODE ...]` | Interactive input | One or more distinct, nonempty physical SR009 host names. Their order determines node indices and global rank ranges. With this option, the script does not prompt for node names. |
| `--gpus-per-node N [N ...]` | `1` per node with `--nodes`; otherwise interactive input | One count for all nodes, or exactly one count per node, in node order. Each count must be an integer from `1` to `8`. In interactive mode, an empty GPU-count answer means `1`. |
| `--nfs-root PATH` | `/home/jovyan` | Existing shared directory under which a fresh `ring_allreduce_ID` run directory is created. |
| `--image IMAGE` | `cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42` | GPU job image. Requires a full `cr.ai.cloud.ru` reference with a tag or SHA-256 digest. A replacement must provide compatible PyTorch/CUDA and the worker's `python` command. |
| `--sizes-mib N [N ...]` | `64 256 1024` | Positive integer buffer sizes in MiB **per GPU rank**. One MiB is `1024²` bytes. FP32 data buffers and additional checking/communication memory are allocated on each GPU. |
| `--iterations N` | `32` | Positive number of timed AllReduce calls in each trial. It does not include warmup, correctness checks or timing aggregation. |
| `--warmup N` | `5` | Warmup AllReduce calls per buffer size, before trials begin. A nonnegative integer; `0` disables warmup. |
| `--trials N` | `3` | Positive number of measurements for each buffer size. The summary uses the median time across these trials. |
| `--timeout-minutes MINUTES` | `15` | Positive finite overall deadline in minutes, including submission, queueing, startup and measurement. Fractional values are accepted. Increase it for larger allocations or longer workloads. |
| `--collective-timeout SECONDS` | `600` | Positive integer timeout passed to the distributed process group for initialization and collective operations. It does not extend the overall deadline. |
| `--allow-local-transports` | Disabled | Set `NCCL_P2P_DISABLE=0` and `NCCL_SHM_DISABLE=0`, allowing local GPU P2P/NVLink and SHM where supported. The network backend remains `IB`. Without this flag, both transports are disabled, preserving the original network-only test. |
| `--socket-ifname NAME` | Not set by the script | Optional TCP bootstrap interface selector passed to `NCCL_SOCKET_IFNAME`, for example `eth0`. It does not select an InfiniBand HCA. |
| `--dry-run` | Disabled | Print JSON configuration, rank layout and job requests. Requires `--nodes`; creates no files or jobs and does not import `client_lib`. |
| `-h`, `--help` | — | Print help, including usage examples, and exit. |

### Measurement method and interpretation

For each buffer size, each rank allocates an FP32 tensor, performs warmup and runs the requested number of trials. Before every trial, it reduces nonzero rank-dependent values and checks the expected sum. The timed loop reduces zeros so repeated SUM does not overflow FP32. Verification and buffer initialization happen outside the measured interval.

Each rank measures the complete batch of `--iterations` calls with CUDA synchronization. After measurement, the ranks reduce their elapsed times with MAX; the slowest rank's time is divided by the iteration count. The final summary uses the median of these per-call times across trials.

Let `S` be the buffer size in bytes, `t` the time per AllReduce in seconds, and `N` the total GPU rank count:

```text
algbw [GB/s] = S / t / 1e9
busbw [GB/s] = algbw * 2 * (N - 1) / N
busbw [Gbit/s] = busbw [GB/s] * 8
equivalent per-rank TX+RX [Gbit/s] = busbw [GB/s] * 16
```

The bus-bandwidth normalization follows the [NCCL tests performance definition](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md). It describes collective-equivalent bandwidth per rank. The displayed TX+RX value is a derived aggregate, not a measured NIC counter or a cluster-wide sum. For `N=2`, the bus factor is `1`; for `N=16`, it is `1.875`. With a single GPU, there is no inter-GPU communication and `busbw` is zero.

The default disables P2P and SHM and forces the NCCL network backend to IB. `--allow-local-transports` changes the transport mix; label results accordingly when comparing runs. The script prints selected NCCL log lines; inspect `nccl.*.log` to confirm which channels/transports were used. This Python benchmark includes its own launch/synchronization overhead and is not the `nccl-tests` binary.

### Progress and output files

The launcher prints the run directory as `Reports: /home/jovyan/ring_allreduce_ID`. After submitting all jobs, it reports completed rank trials, finished ranks and remaining time approximately every five seconds. A rank trial is one buffer-size/trial measurement on one GPU; total progress is `world_size × number of sizes × trials`.

Each worker saves `progress-RANK.json` after every trial and updates it with the median after completing a buffer size. JSON reports are replaced atomically, so they can be read while the run is active. Starting the script again creates a new run; existing progress is retained for inspection, not automatically resumed.

| File in the run directory | Contents |
| --- | --- |
| `run.json` | Node/GPU allocation, global rank layout and benchmark settings. |
| `worker.py`, `launch-NODE.sh` | Script snapshot and one launch script per node. |
| `store` | Temporary shared-file rendezvous used by the distributed process group; it may be removed after completion. |
| `peers.json` | GPU rank metadata and hashed physical-host identities gathered after group initialization. |
| `rank-RANK.log` | Standard output and errors from an individual GPU worker. |
| `nccl.HOST.PID.log` | NCCL initialization, network and graph diagnostics. |
| `progress-RANK.json` | Completed trial measurements and available per-size medians; present during the run. |
| `done-RANK.json` | Final results from a GPU rank after the distributed group has shut down successfully. |
| `error-RANK.json`, `error-node-NODE.json` | Worker or node-supervisor exception traceback. |
| `STOP` | Request for supervisors/workers to terminate the run. |
| `report.json` | Final configuration and results from every rank, written after all ranks finish. |
| `report.txt` | Human-readable summary, also printed by the launcher. |

Replace `ID` with the actual value printed by your run:

```bash
RUN_DIR=/home/jovyan/ring_allreduce_ID

# Inspect the saved allocation and intermediate results.
/usr/bin/python3.12 -m json.tool "$RUN_DIR/run.json"
/usr/bin/python3.12 -m json.tool "$RUN_DIR/progress-0.json"

# Inspect final results after successful completion.
cat "$RUN_DIR/report.txt"
/usr/bin/python3.12 -m json.tool "$RUN_DIR/report.json"
```

### Completion and stopping

The benchmark ends automatically after all measurements. GPU worker processes and their node supervisors exit; the corresponding binary jobs then finish. There is no keep-alive serving period.

To interrupt a foreground launch, press **Ctrl+C** in its Jupyter terminal. The launcher writes `STOP` and attempts to kill every submitted job. To request cancellation from another terminal, create `STOP` in the existing run directory:

```bash
RUN_DIR=/home/jovyan/ring_allreduce_ID
test -f "$RUN_DIR/run.json" && touch "$RUN_DIR/STOP"
```

Node supervisors observe this file and terminate local GPU processes. The launcher may report cancellation as an error; already saved progress remains on NFS. Confirm that all jobs associated with the run have finished in Cloud.ru. If jobs remain queued or supervisors cannot respond, stop those binary jobs in the Cloud.ru interface to release the allocation.

### Troubleshooting and local checks

- No completed trials yet: check whether all node jobs have started, then inspect their platform logs and `rank-*.log`. The distributed group needs all requested ranks.
- `requested N GPUs but sees M`: the job's visible GPU allocation differs from the requested count; inspect its instance shape and allocation.
- Initialization/collective timeout: check shared NFS paths, node availability, TCP bootstrap connectivity, InfiniBand and `nccl.*.log`. Increasing a timeout changes the waiting budget, not connectivity.
- Overall deadline exceeded: increase `--timeout-minutes` for queueing or a larger workload; `--collective-timeout` controls a different timeout.
- CUDA out of memory: reduce `--sizes-mib`; buffer sizes apply independently to every GPU, with additional memory needed for correctness checks and communication.

The repository includes ten local tests using mocked cloud/GPU operations. They allocate no cloud resources and do not establish hardware performance:

```bash
# Run from the repository directory containing ring_allreduce.py and tests/.
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover \
  -s tests -p test_ring_allreduce.py -v
```

The worker-only arguments `--run-dir PATH`, `--_node INDEX` and `--_rank RANK` are used automatically in generated jobs. Worker mode requires the run directory and exactly one selector: `--_node` starts a node supervisor, while `--_rank` starts one global GPU rank. They are not needed for normal launches.

## Русский

### Что даёт инструмент

[`ring_allreduce.py`](ring_allreduce.py) измеряет производительность NCCL Ring AllReduce на NVIDIA H200 в Cloud.ru SR009. Поддерживаются один или несколько физических узлов и **1–8 GPU на каждом**, в том числе разное число GPU на разных узлах. Веса модели не нужны.

Лаунчер отправляет по одной отдельно закреплённой binary-задаче на физический узел. Каждая задача запускает по одному процессу на выделенную GPU; все процессы входят в общую группу NCCL через новый файл rendezvous на общем NFS. Полное число процессов, `world_size`, равно сумме GPU. Глобальные ранги назначаются последовательно: для `8 + 4 + 2` GPU узлы получают ранги `0–7`, `8–11` и `12–13`.

Инструмент проверяет число видимых GPU и идентичность физических узлов, сверяет результат AllReduce, выводит прогресс и сохраняет промежуточные измерения. Итоговые отчёты содержат время операции, алгоритмическую и приведённую пропускную способность для каждого размера буфера.

### Состав

- [`ring_allreduce.py`](ring_allreduce.py): самостоятельный файл с лаунчером и воркерами. Для запуска в облаке достаточно скопировать его.
- [`tests/test_ring_allreduce.py`](tests/test_ring_allreduce.py): локальные тесты топологии, планов задач, файлов запуска, формулы пропускной способности, логики воркеров и очистки; для облачного запуска не нужны.

Каждый запуск копирует скрипт в собственный каталог на NFS под именем `worker.py`; вручную готовить воркеры и скрипты запуска не требуется.

### Требования

- Используйте образ управляющего Jupyter Cloud.ru **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`**. Именно в нём доступны **`/usr/bin/python3.12`**, используемый в командах ниже, и установленный платформой `client_lib`.
- Скопируйте `ring_allreduce.py` на общий NFS, например в `/home/jovyan/ring_allreduce.py`. Скрипт и каталог запуска должны быть доступны по одинаковым абсолютным путям из Jupyter и всех контейнеров задач. Корень запусков должен существовать и быть доступен для записи.
- Выберите разные физические узлы из своей квоты SR009. Скрипт принимает от 1 до 8 GPU на узел; соответствующий тип ресурсов должен быть доступен в вашем проекте. Для `N` GPU запрашивается `h200.Ngpu.(12*N)C.(244*N)G`, например `h200.8gpu.96C.1952G` для восьми GPU.
- GPU-задачи по умолчанию используют **`cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42`**. Это образ воркеров, отдельный от образа управляющего Jupyter. Он предоставляет PyTorch/CUDA; скрипт не устанавливает пакеты и не создаёт виртуальные окружения.
- На выбранных узлах должны работать GPU-драйверы и InfiniBand. Для сетевой инициализации NCCL используется IPv4, сетевой бэкенд — `IB`. Ray и vLLM не используются.

### Запуск

Выполняйте команды из каталога со скриптом. Замените имена узлов на доступные в вашей квоте.

```bash
cd /home/jovyan

# Посмотреть запросы ресурсов и глобальные ранги GPU без отправки задач.
/usr/bin/python3.12 ring_allreduce.py \
  --nodes hpcsrv-201.sr009 hpcsrv-202.sr009 \
  --gpus-per-node 8 \
  --dry-run

# Два узла по восемь GPU; до 60 минут на весь запуск вместе с очередью.
/usr/bin/python3.12 ring_allreduce.py \
  --nodes hpcsrv-201.sr009 hpcsrv-202.sr009 \
  --gpus-per-node 8 \
  --timeout-minutes 60
```

Одно число GPU применяется ко всем узлам. Для разных количеств укажите по одному значению на узел в том же порядке:

```bash
# Заменить NODE0, NODE1 и NODE2 именами трёх разных физических узлов.
/usr/bin/python3.12 ring_allreduce.py \
  --nodes NODE0 NODE1 NODE2 \
  --gpus-per-node 8 4 2 \
  --timeout-minutes 60
```

Поддерживается и один узел. Следующий пример разрешает локальные P2P/NVLink-передачи и транспорт через общую память:

```bash
/usr/bin/python3.12 ring_allreduce.py \
  --nodes hpcsrv-201.sr009 \
  --gpus-per-node 8 \
  --allow-local-transports
```

Для интерактивной настройки не указывайте `--nodes`. Скрипт выведет `client_lib.get_nodes()`, запросит количество и имена физических узлов, затем число GPU на каждом, если не был задан `--gpus-per-node`:

```bash
/usr/bin/python3.12 ring_allreduce.py
```

Чтобы изменить нагрузку, задайте размеры буферов, число повторений и лимиты времени:

```bash
/usr/bin/python3.12 ring_allreduce.py \
  --nodes hpcsrv-201.sr009 hpcsrv-202.sr009 \
  --gpus-per-node 4 \
  --sizes-mib 64 256 1024 \
  --warmup 10 \
  --iterations 100 \
  --trials 5 \
  --timeout-minutes 60 \
  --collective-timeout 1800
```

### Параметры запуска

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `--nodes NODE [NODE ...]` | Интерактивный ввод | Одно или несколько разных непустых имён физических узлов SR009. Порядок задаёт индексы узлов и диапазоны глобальных рангов. С этим параметром скрипт не запрашивает имена узлов. |
| `--gpus-per-node N [N ...]` | По `1` на узел с `--nodes`; иначе интерактивный ввод | Одно число для всех узлов либо ровно по одному числу на узел, в порядке их перечисления. Каждое — целое от `1` до `8`. Пустой ответ на запрос числа GPU в интерактивном режиме означает `1`. |
| `--nfs-root PATH` | `/home/jovyan` | Существующий общий каталог, внутри которого создаётся новый каталог запуска `ring_allreduce_ID`. |
| `--image IMAGE` | `cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42` | Образ GPU-задач. Нужен полный адрес `cr.ai.cloud.ru` с тегом или SHA-256 digest. При замене образ должен предоставлять совместимые PyTorch/CUDA и команду `python` для воркера. |
| `--sizes-mib N [N ...]` | `64 256 1024` | Положительные целые размеры буфера в MiB **на каждый GPU-ранг**. Один MiB — `1024²` байт. На каждой GPU выделяются FP32-буфер и дополнительная память для проверок/обмена данными. |
| `--iterations N` | `32` | Положительное число измеряемых вызовов AllReduce в одном замере. Прогрев, проверка данных и сбор времени сюда не входят. |
| `--warmup N` | `5` | Число прогревочных AllReduce для каждого размера буфера до начала замеров. Целое неотрицательное число; `0` отключает прогрев. |
| `--trials N` | `3` | Положительное число замеров для каждого размера буфера. В сводке используется медианное время этих замеров. |
| `--timeout-minutes MINUTES` | `15` | Положительный конечный общий лимит времени в минутах: отправка задач, очередь, запуск и измерения. Допустимы дробные значения. Увеличьте для большого набора ресурсов или длительной нагрузки. |
| `--collective-timeout SECONDS` | `600` | Положительный целый таймаут инициализации и коллективных операций, передаваемый группе распределённых процессов. Общий лимит времени не продлевает. |
| `--allow-local-transports` | Выключен | Установить `NCCL_P2P_DISABLE=0` и `NCCL_SHM_DISABLE=0`, разрешая локальные GPU P2P/NVLink и SHM там, где они поддерживаются. Сетевой бэкенд остаётся `IB`. Без флага оба транспорта отключены, как в исходном тесте только через сеть. |
| `--socket-ifname NAME` | Скрипт не задаёт | Необязательный селектор TCP-интерфейса для инициализации NCCL, передаваемый в `NCCL_SOCKET_IFNAME`, например `eth0`. Адаптер InfiniBand HCA он не выбирает. |
| `--dry-run` | Выключен | Вывести JSON с конфигурацией, распределением рангов и запросами задач. Требует `--nodes`; не создаёт файлы/задачи и не импортирует `client_lib`. |
| `-h`, `--help` | — | Вывести справку с примерами и завершиться. |

### Методика измерения и трактовка результатов

Для каждого размера буфера ранг выделяет FP32-тензор, выполняет прогрев и заданное число замеров. Перед каждым замером выполняется AllReduce ненулевых значений, зависящих от ранга, с проверкой ожидаемой суммы. Во время измерения суммируются нули, чтобы повторный SUM не переполнял FP32. Проверка данных и заполнение буфера находятся вне измеряемого интервала.

Каждый ранг измеряет весь блок из `--iterations` вызовов с синхронизацией CUDA. После измерения времена рангов объединяются через MAX; время самого медленного ранга делится на число итераций. Итоговая сводка использует медиану полученных времён одной операции по замерам.

Пусть `S` — размер буфера в байтах, `t` — время одного AllReduce в секундах, `N` — полное число GPU-рангов:

```text
algbw [GB/s] = S / t / 1e9
busbw [GB/s] = algbw * 2 * (N - 1) / N
busbw [Gbit/s] = busbw [GB/s] * 8
equivalent per-rank TX+RX [Gbit/s] = busbw [GB/s] * 16
```

Нормализация bus bandwidth соответствует [определению производительности в NCCL tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md). Это приведённая пропускная способность коллективной операции на ранг. Выводимый TX+RX — вычисленная суммарная величина, а не измеренный счётчик сетевого адаптера или сумма по кластеру. Для `N=2` коэффициент равен `1`, для `N=16` — `1.875`. При одной GPU межкарточного обмена нет, поэтому `busbw` равен нулю.

По умолчанию P2P и SHM отключены, а сетевой бэкенд NCCL принудительно задан как IB. `--allow-local-transports` меняет набор разрешённых транспортов; учитывайте этот режим при сравнении результатов. Скрипт выводит выбранные строки логов NCCL; фактически использованные каналы/транспорты проверяйте в `nccl.*.log`. Это Python-бенчмарк со своими накладными расходами на вызовы и синхронизацию, а не бинарный инструмент `nccl-tests`.

### Прогресс и файлы результатов

Лаунчер печатает каталог запуска: `Reports: /home/jovyan/ring_allreduce_ID`. После отправки всех задач он примерно каждые пять секунд выводит число завершённых замеров по рангам, число завершившихся рангов и оставшееся время. Один rank trial — один замер для одного размера буфера на одной GPU; полный прогресс равен `world_size × число размеров × trials`.

После каждого замера воркер сохраняет `progress-RANK.json`, а после завершения размера буфера дописывает медиану. JSON-отчёты заменяются атомарно, поэтому их можно читать во время работы. Повторный запуск скрипта создаёт новый каталог; старые промежуточные результаты сохраняются для просмотра, но автоматически не возобновляются.

| Файл в каталоге запуска | Содержимое |
| --- | --- |
| `run.json` | Узлы, выделенные GPU, глобальные ранги и настройки бенчмарка. |
| `worker.py`, `launch-NODE.sh` | Копия скрипта и по одному скрипту запуска на узел. |
| `store` | Временный общий файл rendezvous для группы распределённых процессов; после завершения может удаляться. |
| `peers.json` | Собранные после инициализации группы метаданные GPU-рангов и хеши идентификаторов физических узлов. |
| `rank-RANK.log` | Стандартный вывод и ошибки отдельного GPU-воркера. |
| `nccl.HOST.PID.log` | Диагностика инициализации, сети и графа NCCL. |
| `progress-RANK.json` | Завершённые замеры и уже доступные медианы по размерам; появляется во время работы. |
| `done-RANK.json` | Итоговый результат GPU-ранга после успешного завершения распределённой группы. |
| `error-RANK.json`, `error-node-NODE.json` | Traceback исключения GPU-воркера или управляющего процесса узла. |
| `STOP` | Запрос управляющим процессам/воркерам завершить запуск. |
| `report.json` | Итоговая конфигурация и результаты всех рангов; создаётся после завершения всех рангов. |
| `report.txt` | Человекочитаемая сводка, также выводимая лаунчером. |

Замените `ID` фактическим значением, напечатанным при запуске:

```bash
RUN_DIR=/home/jovyan/ring_allreduce_ID

# Посмотреть сохранённую конфигурацию и промежуточные результаты.
/usr/bin/python3.12 -m json.tool "$RUN_DIR/run.json"
/usr/bin/python3.12 -m json.tool "$RUN_DIR/progress-0.json"

# Посмотреть итоговые результаты после успешного завершения.
cat "$RUN_DIR/report.txt"
/usr/bin/python3.12 -m json.tool "$RUN_DIR/report.json"
```

### Завершение и остановка

Бенчмарк автоматически завершается после всех измерений. GPU-воркеры и управляющие процессы узлов выходят, после чего завершаются соответствующие binary-задачи. Периода keep-alive после теста нет.

Чтобы прервать запуск из его терминала Jupyter, нажмите **Ctrl+C**. Лаунчер запишет `STOP` и попытается остановить все отправленные задачи. Чтобы запросить остановку из другого терминала, создайте `STOP` в существующем каталоге запуска:

```bash
RUN_DIR=/home/jovyan/ring_allreduce_ID
test -f "$RUN_DIR/run.json" && touch "$RUN_DIR/STOP"
```

Управляющие процессы узлов увидят файл и завершат локальные GPU-процессы. Лаунчер может сообщить об отмене как об ошибке; уже записанные результаты останутся на NFS. Проверьте завершение всех задач этого запуска в Cloud.ru. Если задачи ещё стоят в очереди или управляющие процессы не отвечают, остановите эти binary-задачи в интерфейсе Cloud.ru, чтобы освободить ресурсы.

### Диагностика и локальные проверки

- Нет завершённых замеров: проверьте, запустились ли задачи всех узлов, затем посмотрите логи платформы и `rank-*.log`. Группе распределённых процессов нужны все запрошенные ранги.
- `requested N GPUs but sees M`: число видимых GPU задачи отличается от запрошенного; проверьте тип инстанса и выделенные ресурсы.
- Таймаут инициализации/коллективной операции: проверьте общий путь NFS, доступность узлов, сетевую связность для TCP-инициализации, InfiniBand и `nccl.*.log`. Увеличение таймаута меняет время ожидания, а не связность.
- Истёк общий лимит: увеличьте `--timeout-minutes`, учитывая очередь и объём нагрузки; `--collective-timeout` управляет отдельным таймаутом.
- CUDA out of memory: уменьшите `--sizes-mib`; размеры применяются к каждой GPU отдельно, дополнительно нужна память для проверки данных и обмена.

В репозитории есть десять локальных тестов с имитацией облака/GPU. Они не выделяют облачные ресурсы и не измеряют производительность оборудования:

```bash
# Выполнять из каталога репозитория, содержащего ring_allreduce.py и tests/.
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover \
  -s tests -p test_ring_allreduce.py -v
```

Служебные аргументы `--run-dir PATH`, `--_node INDEX` и `--_rank RANK` автоматически используются внутри созданных задач. Режим воркера требует каталог запуска и ровно один селектор: `--_node` запускает управляющий процесс узла, а `--_rank` — один глобальный GPU-ранг. Для обычного запуска они не нужны.
