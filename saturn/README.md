# Lyra 0.1 на Saturn Cloud — обучение с нуля на максимуме

Ноутбук `saturn/train_lyra_saturn.ipynb` использует репозиторные `scripts/train.py` и `LyraModel` (отдельного цикла обучения нет). Обучение всегда идёт **с нуля** (случайная инициализация весов), если не указан `--resume`.

## Как запустить обучение с нуля

1. В Saturn Cloud создай проект и выбери **GPU-инстанс**:
   минимум T4 16GB; чем больше VRAM — тем большую модель ноутбук выберет сам (таблица ниже).
2. Открой JupyterLab, склонируй репозиторий на персистентный диск:
   ```bash
   cd /home/jovyan/project
   git clone https://github.com/er1kosfunpay-sketch/Lyra0.1.git
   cd Lyra0.1
   ```
   (если папка проекта уже является клоном — просто перейди в неё).
3. Открой `saturn/train_lyra_saturn.ipynb` и в первой ячейке проверь:
   - `AUTO_SIZE = True` — автовыбор максимальной модели под твой GPU;
   - `RUN_NAME` — **новое имя** для чистого запуска (например `"stage1_max"`),
     либо удали старый каталог `checkpoints/<RUN_NAME>`;
   - `AUTO_RESUME = True` + отсутствие `checkpoints/<RUN_NAME>/latest.pt` = старт с нуля.
     Уже лежащий `latest.pt` означает продолжение — для старта с нуля его не должно быть.
4. Прогони ячейки сверху вниз. Ячейка GPU-check сама выберет конфиг и напечатает
   `VRAM=... -> CONFIG=...`. Дальше всё автоматически: зависимости, датасет-максимум,
   токенизатор, preflight, обучение, verify, экспорт в `lyra_export/`.
5. Остановка/рестарт инстанса ничего не стирает: для продолжения просто прогони
   ноутбук снова — он подхватит `latest.pt`. Для повторного старта с нуля задай
   новый `RUN_NAME` (или удали каталог рана) и прогони снова.

Ручной вариант из терминала (эквивалент ячейки 11, с нуля — без `--resume`):

```bash
python scripts/train.py --config configs/saturn_t4.json \
  --tokenizer artifacts/tokenizer/tokenizer.json \
  --data data/processed/saturn_stage1/train.jsonl \
  --validation data/processed/saturn_stage1/validation.jsonl \
  --stage pretrain --steps 2000 --batch-size 1 --grad-accum 8 \
  --lr 3e-4 --warmup-steps 100 --save-every 100 --archive-every 100 \
  --keep-last-checkpoints 3 --eval-every 250 \
  --out checkpoints/stage1_t4
```

## Тиры «максимум под GPU» (автовыбор по VRAM)

Оценка памяти (консервативная): веса bf16 + градиенты + fp32 Adam + покомпонентный
activation checkpointing + запас. Тир считается подходящим при оценке ≤ 85% VRAM.
Проверено скриптом `python saturn/sizing.py --table --vram <GB>`:

| VRAM GPU | Тир / конфиг | Параметры | Оценка VRAM | Шаги |
|---|---|---|---|---|
| 12–20 GB (T4 16GB, бесплатный/стартовый Saturn `g4dn.xlarge`) | `configs/saturn_t4.json` | 528,545,280 | ~10.2 GB | 2000 |
| 20–34 GB (L4, A10G 24GB) | `configs/saturn_a10g.json` | 1,025,599,488 | ~18.3 GB | 2500 |
| 34–70 GB (A100 40GB) | `configs/saturn_a100.json` | 1,848,250,880 | ~31.7 GB | 3000 |
| ≥70 GB (H100/H200 80GB+) | `configs/saturn_h100.json` | 3,221,425,152 | ~53.9 GB | 4000 |

Архитектура везде одинаковая по духу (RMSNorm, RoPE, GQA, SwiGLU, tied embeddings,
вокабуляр 16384, контекст 1024, микробатч 1, grad_accum 8, LR 3e-4) — растёт только
ширина/глубина. `configs/saturn.json` (248M) оставлен как безопасный запасной вариант
при `AUTO_SIZE=False`. Оценки — эвристика: перед длинным раном смотри на
`nvidia-smi` в первые сотни шагов; при OOM уменьши `--batch-size`/контекст или
возьми тир ниже.

## Датасет — единственный источник

Единственный источник данных проекта — `Den4ikAI/russian_dialogues_2`.
Микса из нескольких датасетов больше нет, лимитов на источник тоже нет:

```bash
python scripts/prepare_data.py --out data/processed/saturn_stage1
```

Скрипт берёт весь корпус, чистит его (пустые/битые/бессмысленные строки, ссылочные
заглушки, неточно-русское, точные дубликаты), делит 98/1/1 на train/validation/test и
пишет `dataset_stats.json` (сколько исходных, удалено, осталось, сообщений и символов
по сплитам). Если скачать датасет не удалось — сборка падает с явной ошибкой и
**не подменяет** данные другим корпусом. Повторный запуск с теми же параметрами
использует уже собранные сплиты (`--force` — пересобрать).

## Продолжить / SFT

- Продолжить: тот же `RUN_NAME`, прогнать ноутбук — подхватит `latest.pt`
  (модель + оптимизатор + scheduler + scaler + RNG + шаг).
- SFT с сохранением предобученных весов: новый `RUN_NAME='stage3_sft'`,
  в команде ячейки 11 сменить `--stage` на `sft` и добавить
  `--resume <путь-к-latest.pt> --reset-stage`, `--out` — в отдельный каталог.
