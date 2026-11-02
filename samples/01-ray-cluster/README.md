# Ray cluster and GPU smoke test

[English](#english) · [Русский](#русский)

## English

This sample creates a Ray cluster with a configurable node count and verifies CUDA on every GPU, object transfer in both directions around a ring of nodes, and NCCL AllReduce across all GPUs. No model is loaded. Defaults: two nodes with one GPU each. A single node is supported; cross-node checks are then not applicable. DP/TP/EP are inference settings and do not apply to this infrastructure sample.

### Requirements

Run from the management Jupyter image **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`**, which provides **`/usr/bin/python3.12`** and the platform `client_lib`. Copy the sample onto shared NFS, for example `/home/jovyan/samples`; absolute paths must match in Jupyter and worker containers. The run root must already exist.

Workers use **`cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42`**. Cloud.ru selects physical hosts: one binary job with `n_workers=N` and `processes_per_worker=1`, without `include_nodes`/`exclude_nodes`. Node ranks come from MPI and addresses are discovered at launch. Resource shape: `h200.Ggpu.(12*G)C.(244*G)G`. The platform enforces available shapes, allocation limits and queue permissions. Rejected submissions fail visibly; there are no automatic submission retries.

### Included files

- Launcher, bootstrap and per-node supervisors; `cluster_common.py` and `topology.py` handle placement, discovery and validation.
- `requirements-*.txt` pins runtime; `tests/` checks plans and lifecycle without cloud GPUs.

### Launch

```bash
cd /home/jovyan/samples/01-ray-cluster

/usr/bin/python3.12 submit_ray.py --num-nodes 3 --gpus-per-node 2 --dry-run
/usr/bin/python3.12 submit_ray.py --num-nodes 3 --gpus-per-node 2 --keep-alive --keep-minutes 60
```

### All launcher parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--num-nodes N` | `2` | Number of automatically placed workers. |
| `--gpus-per-node N` | `1` | Uniform H200 count per worker, 1..8. |
| `--cpus-per-node N` | `12*GPUs` | CPU resources advertised to Ray; at least one CPU per GPU. Platform shape remains 12 CPUs/GPU. |
| `--allocation-name NAME` | `workspace default` | Training allocation name. |
| `--queue-name NAME` | `shared` | Target queue; shared is selected explicitly by default. |
| `--image IMAGE` | `cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42` | Existing worker container image. |
| `--nfs-root PATH` | `/home/jovyan` | Existing shared run root. |
| `--startup-timeout SECONDS` | `900` | Queue/install/registration timeout. |
| `--test-timeout SECONDS` | `600` | Smoke-test timeout after the head starts. |
| `--keep-alive` | `off` | Keep Ray running after a successful test. |
| `--keep-minutes MINUTES` | `30` | Maximum keep-alive lease after the test. |
| `--dry-run` | `off` | Print job plan; no files, SDK imports or cloud resources. |
| `--prepare-only` | `off` | Write run snapshot without submitting jobs. |
| `--stop RUN_DIR` | `—` | Ask all node supervisors to terminate Ray. |
| `-h / --help` | `—` | Show help. |

### Results and stopping

Each launch creates a separate directory and snapshots its source. `run.json` stores configuration, `platform-job.json` the job ID/submission reply, and `node-RANK.json`/`stopped-RANK.json` record discovered workers and shutdown. Expect records for all N node ranks. A rejected submission may create no platform job.

`PASSED.json` contains results and metadata. Without `--keep-alive`, Ray stops after the test; otherwise GPUs remain allocated until lease expiry or explicit shutdown.

```bash
RUN_DIR=/home/jovyan/cloud-ray-ID
/usr/bin/python3.12 submit_ray.py --stop "$RUN_DIR"
```

**Stop the service when finished and verify that the Cloud.ru binary job actually ends.** STOP is an NFS request; if supervisors hang, stop the platform job using the ID from `platform-job.json`. Ctrl+C while the launcher is active invokes cleanup. Check every `stopped-RANK.json` and a terminal platform status; Terminating does not yet confirm resource release.

### Versions and validation

Ray 2.59.0. The original two-node preset was exercised on GPUs. The generalized version has local plan/lifecycle tests; this does not validate every topology on hardware. All source comments and messages are in English.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover -s tests -v
```

## Русский

Пример создаёт кластер Ray на настраиваемом числе узлов, проверяет CUDA на каждой GPU, передачу объектов между соседними узлами в обоих направлениях и NCCL AllReduce на всех GPU. Модель не загружается. По умолчанию — два узла по одной GPU. Один узел также поддерживается; межузловая проверка тогда неприменима. DP/TP/EP относятся к инференсу и в этом инфраструктурном примере не используются.

### Требования

Запускайте из управляющего Jupyter на образе **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`**: именно там доступен **`/usr/bin/python3.12`** и платформенный `client_lib`. Скопируйте каталог примера в общий NFS, например `/home/jovyan/samples`; абсолютные пути должны совпадать в Jupyter и контейнерах задач. Корень запусков должен существовать.

Воркеры используют **`cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42`**. Физические серверы выбирает Cloud.ru: одна binary-задача, `n_workers=N`, `processes_per_worker=1`, без `include_nodes`/`exclude_nodes`. Ранг берётся из MPI; адреса обнаруживаются при запуске. Тип ресурсов — `h200.Ggpu.(12*G)C.(244*G)G`. Доступность типов ресурсов, лимиты аллокации и права очереди проверяет платформа. Ошибка `submit()` выводится сразу; запросы автоматически не повторяются.

### Состав

- Лаунчер, bootstrap и управляющий процесс на каждый узел; `cluster_common.py` и `topology.py` для размещения, обнаружения и проверок.
- `requirements-*.txt` фиксирует runtime; `tests/` проверяет планы и жизненный цикл без облачных GPU.

### Запуск

```bash
cd /home/jovyan/samples/01-ray-cluster

/usr/bin/python3.12 submit_ray.py --num-nodes 3 --gpus-per-node 2 --dry-run
/usr/bin/python3.12 submit_ray.py --num-nodes 3 --gpus-per-node 2 --keep-alive --keep-minutes 60
```

### Все параметры лаунчера

| Параметр | По умолчанию | Описание |
| --- | --- | --- |
| `--num-nodes N` | `2` | Число автоматически размещаемых воркеров. |
| `--gpus-per-node N` | `1` | Одинаковое число H200 на воркер, 1..8. |
| `--cpus-per-node N` | `12*GPUs` | CPU, объявленные Ray; не меньше одного на GPU. Платформа выделяет 12 CPU/GPU. |
| `--allocation-name NAME` | `workspace default` | Имя аллокации задач обучения. |
| `--queue-name NAME` | `shared` | Целевая очередь; по умолчанию явно выбирается shared. |
| `--image IMAGE` | `cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42` | Готовый образ контейнера воркеров. |
| `--nfs-root PATH` | `/home/jovyan` | Существующий общий каталог запусков. |
| `--startup-timeout SECONDS` | `900` | Таймаут очереди, установки и регистрации. |
| `--test-timeout SECONDS` | `600` | Таймаут smoke-теста после старта головного узла. |
| `--keep-alive` | `off` | Оставить Ray работать после успешного теста. |
| `--keep-minutes MINUTES` | `30` | Максимальный срок keep-alive после теста. |
| `--dry-run` | `off` | Показать план; без файлов, импорта SDK и облачных ресурсов. |
| `--prepare-only` | `off` | Записать копию запуска без отправки задач. |
| `--stop RUN_DIR` | `—` | Попросить управляющие процессы всех узлов остановить Ray. |
| `-h / --help` | `—` | Показать справку. |

### Результаты и остановка

Каждый запуск создаёт отдельный каталог и копирует туда исходники. `run.json` хранит конфигурацию, `platform-job.json` — ID/ответ отправки, `node-RANK.json` и `stopped-RANK.json` — фактические узлы и завершение. Нужны записи для всех N рангов. При ошибке отправки задача может не существовать на платформе.

`PASSED.json` содержит результаты и метаданные. Без `--keep-alive` пример сам завершает Ray после теста; с ним GPU заняты до конца lease или явной остановки.

```bash
RUN_DIR=/home/jovyan/cloud-ray-ID
/usr/bin/python3.12 submit_ray.py --stop "$RUN_DIR"
```

**После работы обязательно остановите сервис и проверьте завершение binary-задачи в Cloud.ru.** STOP — запрос через NFS; если воркеры зависли, остановите задачу платформы по ID из `platform-job.json`. Ctrl+C во время активного лаунчера включает его очистку. Проверьте `stopped-RANK.json` для всех узлов и конечный статус платформы; Terminating ещё не означает освобождение ресурсов.

### Версии и проверка

Ray 2.59.0. Исходный пресет на двух узлах ранее проверен на GPU. Новая параметризованная версия проверяется локальными тестами; это не подтверждение работоспособности всех топологий на оборудовании. Все комментарии и сообщения исходного кода — на английском.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover -s tests -v
```

