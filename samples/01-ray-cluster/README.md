# 01 · Ray cluster and NCCL smoke test

[English](#english) · [Русский](#русский) · [All samples / Все примеры](../README.md)

## English

### What this sample provides

Creates a two-node Ray cluster on Cloud.ru SR009 and checks GPU scheduling, CUDA execution, Ray object transfers in both directions and a data-checked NCCL AllReduce. The result is a `PASSED.json` report with resource information and timing/bandwidth measurements. This is a functional infrastructure smoke test; its measurements are not a peak-bandwidth benchmark.

The default uses one H200 on each node (two GPUs total); `--gpus-per-node 8` checks all 16 GPUs. No model checkpoint is needed. By default both jobs stop after the test; `--keep-alive` leaves Ray available for a bounded period.

### Included files

- `submit_ray.py`: management entrypoint; submits jobs, waits for the report and requests shutdown.
- `bootstrap_ray.py`, `ray_node.py`: isolated environment installation, physical-node/GPU checks and Ray supervision.
- `smoke_test.py`, `cluster_common.py`: GPU actors, communication checks, job configuration and shared-file helpers.
- `requirements-ray.txt`: Ray version pin; `tests/test_cluster.py`: local tests with mocked cloud operations.

### Run

Use Cloud.ru management Jupyter with the **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`** image, which provides **`/usr/bin/python3.12`** and the platform-provided `client_lib`. Copy the repository to shared NFS, normally `/home/jovyan/samples`; that path must also be visible inside jobs. The default hosts must be available in your SR009 project; use `--nodes` for your allocation. Jobs need package-download access and working InfiniBand. Ray is installed inside jobs, not in Jupyter.

```bash
cd /home/jovyan/samples/01-ray-cluster

# Inspect the job plan without allocating resources.
/usr/bin/python3.12 submit_ray.py --dry-run

# Run the default two-GPU check and stop automatically.
/usr/bin/python3.12 submit_ray.py

# Alternatively, check all 16 GPUs and retain Ray for up to 30 minutes.
/usr/bin/python3.12 submit_ray.py --gpus-per-node 8 --keep-alive --keep-minutes 30
```

Use the exact run directory printed by the command; replace `ID` below. `head.json` contains the Ray address and private dashboard URL. `PASSED.json` is written only after the checks succeed.

```bash
RUN_DIR=/home/jovyan/cloud-ray-ID
/usr/bin/python3.12 -m json.tool "$RUN_DIR/PASSED.json"
/usr/bin/python3.12 -m json.tool "$RUN_DIR/head.json"
/usr/bin/python3.12 submit_ray.py --stop "$RUN_DIR"
```

`--prepare-only` creates a run snapshot without submitting jobs. Each run has independent files: `run.json`, `launch-*.sh`, `node-*.json`, `head.json`, `PASSED.json`, `install-*.log`, `ray-start-*.log`, `smoke.log` and `nccl.*.log`. Failures are saved as `*-error*.json`; shutdown creates `STOP` and then `stopped-*.json`.

### Launch parameters

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

### Worker entrypoints

The submit script starts these automatically inside jobs; they are not additional getting-started steps. All accept `-h` / `--help`.

| Script | Arguments | Description |
| --- | --- | --- |
| `bootstrap_ray.py` | `--run-dir PATH --rank RANK` | Required: run snapshot; rank `0`, `1` or `auto` (MPI environment). |
| `ray_node.py` | `--run-dir PATH --rank 0\|1` | Required: run snapshot and node rank. |
| `smoke_test.py` | `--run-dir PATH` | Required: run snapshot with head and node metadata. |

### Local checks

Run from this directory; no cloud allocation or model weights are needed.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover -s tests -v
```

## Русский

### Что даёт пример

Создаёт двухузловой кластер Ray в Cloud.ru SR009 и проверяет размещение GPU-акторов, выполнение CUDA, передачу объектов Ray в обе стороны и NCCL AllReduce с проверкой данных. Результат — отчёт `PASSED.json` с ресурсами и измерениями времени/пропускной способности. Это функциональная проверка инфраструктуры; её измерения не являются тестом пиковой пропускной способности.

По умолчанию используется по одной H200 на узел, всего две GPU; `--gpus-per-node 8` проверяет все 16 GPU. Веса модели не нужны. После теста обе задачи автоматически завершаются; с `--keep-alive` Ray продолжает работать ограниченное время.

### Состав

- `submit_ray.py`: точка входа из управляющего Jupyter; отправка задач, ожидание отчёта и остановка.
- `bootstrap_ray.py`, `ray_node.py`: установка изолированного окружения, проверка физических узлов/GPU и управление Ray.
- `smoke_test.py`, `cluster_common.py`: GPU-акторы, проверки связи, конфигурация задач и работа с общими файлами.
- `requirements-ray.txt`: зафиксированная версия Ray; `tests/test_cluster.py`: локальные тесты с имитацией облачных операций.

### Запуск

Используйте управляющий Jupyter Cloud.ru на образе **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`**. Именно в этом образе доступен **`/usr/bin/python3.12`** и установлен платформенный `client_lib`. Скопируйте репозиторий на общий NFS, обычно в `/home/jovyan/samples`; этот путь должен быть доступен и внутри задач. Узлы по умолчанию должны быть доступны в вашем проекте SR009; для других укажите `--nodes`. Задачам нужны доступ к загрузке пакетов и рабочий InfiniBand. Ray устанавливается внутри задач, а не в Jupyter.

```bash
cd /home/jovyan/samples/01-ray-cluster

# Посмотреть план без выделения ресурсов.
/usr/bin/python3.12 submit_ray.py --dry-run

# Проверить две GPU и автоматически остановить кластер.
/usr/bin/python3.12 submit_ray.py

# Другой вариант: проверить 16 GPU и оставить Ray максимум на 30 минут.
/usr/bin/python3.12 submit_ray.py --gpus-per-node 8 --keep-alive --keep-minutes 30
```

Используйте точный каталог запуска, напечатанный командой; ниже замените `ID`. В `head.json` находятся адрес Ray и URL внутреннего dashboard. `PASSED.json` появляется только после успешных проверок.

```bash
RUN_DIR=/home/jovyan/cloud-ray-ID
/usr/bin/python3.12 -m json.tool "$RUN_DIR/PASSED.json"
/usr/bin/python3.12 -m json.tool "$RUN_DIR/head.json"
/usr/bin/python3.12 submit_ray.py --stop "$RUN_DIR"
```

`--prepare-only` создаёт файлы запуска без отправки задач. Для каждого запуска отдельно сохраняются `run.json`, `launch-*.sh`, `node-*.json`, `head.json`, `PASSED.json`, `install-*.log`, `ray-start-*.log`, `smoke.log` и `nccl.*.log`. Ошибки записываются в `*-error*.json`; при остановке появляются `STOP`, затем `stopped-*.json`.

### Параметры запуска

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

### Служебные точки входа

Скрипт отправки запускает их автоматически внутри задач; вручную запускать их для начала работы не требуется. Все поддерживают `-h` / `--help`.

| Скрипт | Аргументы | Описание |
| --- | --- | --- |
| `bootstrap_ray.py` | `--run-dir PATH --rank RANK` | Обязательны: каталог запуска; ранг `0`, `1` или `auto` (из окружения MPI). |
| `ray_node.py` | `--run-dir PATH --rank 0\|1` | Обязательны: каталог запуска и ранг узла. |
| `smoke_test.py` | `--run-dir PATH` | Обязателен каталог запуска с метаданными головного процесса и узлов. |

### Локальные проверки

Выполняются из этого каталога без выделения облачных ресурсов и без весов модели.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover -s tests -v
```
