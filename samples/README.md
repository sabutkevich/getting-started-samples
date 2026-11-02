# Cloud.ru distributed GPU samples

[English](#english) · [Русский](#русский)

## English

Four self-contained H200/SR009 samples with configurable node counts and automatic placement without physical server names. Inference supports configurable DP, TP, PP and EP. Every directory contains its own source, requirements, tests and bilingual README.

| Sample | Purpose | API |
| --- | --- | --- |
| [01-ray-cluster](01-ray-cluster/README.md) | Ray: CUDA, object transfer and NCCL smoke test | none |
| [02-vllm-ray](02-vllm-ray/README.md) | DeepSeek inference with Ray; default TP8/DP2/EP16 | one on head |
| [03-vllm-mp-internal-lb](03-vllm-mp-internal-lb/README.md) | Native multiprocessing and internal DP balancing | one on node 0 |
| [04-vllm-mp-hybrid-lb](04-vllm-mp-hybrid-lb/README.md) | Native multiprocessing and hybrid DP balancing | one per replica-host group |

Run from the management Jupyter image **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`**, which provides **`/usr/bin/python3.12`** and the platform `client_lib`. Copy the sample onto shared NFS, for example `/home/jovyan/samples`; absolute paths must match in Jupyter and worker containers. The run root must already exist.

Workers use **`cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42`**. Cloud.ru selects physical hosts: one binary job with `n_workers=N` and `processes_per_worker=1`, without `include_nodes`/`exclude_nodes`. Node ranks come from MPI and addresses are discovered at launch. Resource shape: `h200.Ggpu.(12*G)C.(244*G)G`. The platform enforces available shapes, allocation limits and queue permissions. Rejected submissions fail visibly; there are no automatic submission retries.

### Topology rules

- Each of N workers requests the same 1–8 H200 GPUs. N is any positive count accepted by the platform.
- `DP * TP * PP = N * GPUs_per_node`. DP is calculated by default; an explicit DP must use all requested GPUs.
- With EP enabled: `EP = TP * DP`; with EP disabled its group size is 1. vLLM 0.32.0 does not expose an independent arbitrary EP size. `--ep-size` asserts the expected result.
- A TP*PP replica must fit uniformly within a worker or span a whole number of complete workers. Local DP is calculated automatically.
- Topology arithmetic and available checkpoint geometry, including attention/o_groups TP divisibility, are checked before submission. Uneven expert placement is permitted without requiring experts % EP == 0.
- Planning checks cannot guarantee GPU memory capacity or model-kernel support for every combination. Runtime remains stock, without patches. New cluster sizes and parallelism combinations need GPU validation.

[Official EP documentation](https://docs.vllm.ai/en/v0.32.0/serving/expert_parallel_deployment/) · [DP and balancing](https://docs.vllm.ai/en/v0.32.0/serving/data_parallel_deployment/).

### Getting started

```bash
cd /home/jovyan/samples/03-vllm-mp-internal-lb
/usr/bin/python3.12 submit_vllm.py \
  --num-nodes 10 --gpus-per-node 8 --tp 1 --dp 80 --ep-size 80 \
  --allocation-name alloc-huawei-rnd-sr009 --queue-name shared --dry-run
```

Remove `--dry-run` after reviewing the plan. Each sample README lists every launch/query/stop parameter. Sample 01 loads no model and does not use DP/TP/EP. Sample 04 has an API on every node for local replicas, or only on replica leaders for spanning replicas. No external HTTP balancer is included.

### Lifecycle and limitations

Inference launchers return after first generation while jobs keep using GPUs. Default lease: 120 minutes after readiness; `--serve-minutes 720` gives 12 hours. Use `--stop RUN_DIR` and verify platform termination. If supervisors hang, stop the binary job using `platform-job.json`. An NFS path is not an API address; find the endpoint and key-file path in `READY.json`.

Runtime versions are preserved: worker image py3.12-torch2.9.0:0.0.42, Ray 2.59.0, vLLM 0.32.0 and CUDA toolkit 13.0.3. New sizes are not hardware-validated configurations: verify memory capacity and model kernels separately. Older archives in the parent directory are not updated automatically; this directory contains the current source.

```bash
cd /home/jovyan/samples
for sample in 01-ray-cluster 02-vllm-ray 03-vllm-mp-internal-lb 04-vllm-mp-hybrid-lb; do
  (cd "$sample" && PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover -s tests -v) || exit 1
done
```

## Русский

Четыре самостоятельных примера для H200 в SR009 с настраиваемым числом узлов и автоматическим размещением без имён серверов. У инференса настраиваются DP, TP, PP и EP. В каждом каталоге собственные исходники, зависимости, тесты и README на двух языках.

| Пример | Назначение | API |
| --- | --- | --- |
| [01-ray-cluster](01-ray-cluster/README.md) | Ray: проверка CUDA, обмена объектами и NCCL | нет |
| [02-vllm-ray](02-vllm-ray/README.md) | Инференс DeepSeek через Ray; по умолчанию TP8/DP2/EP16 | один на головном узле |
| [03-vllm-mp-internal-lb](03-vllm-mp-internal-lb/README.md) | Multiprocessing и внутренняя балансировка DP | один на узле 0 |
| [04-vllm-mp-hybrid-lb](04-vllm-mp-hybrid-lb/README.md) | Multiprocessing и гибридная балансировка DP | один на группу узлов реплики |

Запускайте из управляющего Jupyter на образе **`jupyter-cuda13.3-pt2.13.0-gpu:0.0.102`**: именно там доступен **`/usr/bin/python3.12`** и платформенный `client_lib`. Скопируйте каталог примера в общий NFS, например `/home/jovyan/samples`; абсолютные пути должны совпадать в Jupyter и контейнерах задач. Корень запусков должен существовать.

Воркеры используют **`cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42`**. Физические серверы выбирает Cloud.ru: одна binary-задача, `n_workers=N`, `processes_per_worker=1`, без `include_nodes`/`exclude_nodes`. Ранг берётся из MPI; адреса обнаруживаются при запуске. Тип ресурсов — `h200.Ggpu.(12*G)C.(244*G)G`. Доступность типов ресурсов, лимиты аллокации и права очереди проверяет платформа. Ошибка `submit()` выводится сразу; запросы автоматически не повторяются.

### Правила топологии

- Выделяются одинаковые 1–8 H200 на каждом из N узлов. Число узлов — любое положительное в пределах платформы.
- `DP * TP * PP = N * GPU_per_node`. DP по умолчанию вычисляется; явно заданный DP должен использовать все запрошенные GPU.
- При включённом EP: `EP = TP * DP`; при выключенном EP размер группы EP равен 1. Независимый произвольный EP vLLM 0.32.0 не предоставляет. `--ep-size` проверяет ожидаемый размер.
- Реплика TP*PP должна целиком помещаться в узле с равномерным размещением либо занимать целое число полных узлов. Локальный DP вычисляется автоматически.
- Перед отправкой проверяются арифметика топологии и доступная геометрия checkpoint, включая TP для attention/o_groups. Разрешённое размещение экспертов не требует делимости числа экспертов на EP.
- Проверки планирования не гарантируют достаточную память или поддержку каждой комбинации ядрами модели. Runtime остаётся штатным, без патчей. Новые размеры кластера и комбинации нужно проверить на GPU.

[Официальная документация EP](https://docs.vllm.ai/en/v0.32.0/serving/expert_parallel_deployment/) · [DP и балансировка](https://docs.vllm.ai/en/v0.32.0/serving/data_parallel_deployment/).

### Начало работы

```bash
cd /home/jovyan/samples/03-vllm-mp-internal-lb
/usr/bin/python3.12 submit_vllm.py \
  --num-nodes 10 --gpus-per-node 8 --tp 1 --dp 80 --ep-size 80 \
  --allocation-name alloc-huawei-rnd-sr009 --queue-name shared --dry-run
```

После проверки плана уберите `--dry-run`. Подробные параметры запуска, запросов и остановки — в README выбранного примера. Пример 01 не загружает модель и не использует DP/TP/EP. В примере 04 локальная реплика даёт API на каждом узле, а реплика на нескольких узлах — API только на ведущем. Внешний HTTP-балансировщик не входит в набор.

### Жизненный цикл и ограничения

Лаунчеры инференса возвращаются после первой генерации, но задачи продолжают занимать GPU. По умолчанию lease — 120 минут после готовности; `--serve-minutes 720` — 12 часов. Используйте `--stop RUN_DIR` и проверяйте завершение задачи на платформе. При зависании остановите binary-задачу по ID из `platform-job.json`. Путь NFS не является адресом API; адрес и путь к API-ключу находятся в `READY.json`.

Версии runtime сохранены: worker image py3.12-torch2.9.0:0.0.42, Ray 2.59.0, vLLM 0.32.0, CUDA toolkit 13.0.3. Новый размер кластера не является подтверждённой GPU-конфигурацией: память и ядра модели необходимо проверять отдельно. Старые архивы в родительском каталоге не обновляются автоматически; актуальные исходники находятся здесь.

```bash
cd /home/jovyan/samples
for sample in 01-ray-cluster 02-vllm-ray 03-vllm-mp-internal-lb 04-vllm-mp-hybrid-lb; do
  (cd "$sample" && PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -m unittest discover -s tests -v) || exit 1
done
```

