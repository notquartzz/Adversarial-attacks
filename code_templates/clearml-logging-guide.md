# Как залогировать свой эксперимент в ClearML

За основу — уже рабочий кусок кода из FGSM-ноутбука (ResNet vs ViT, CIFAR-10). Этот файл объясняет, что в нём происходит и как перенести ту же структуру на **свой** эксперимент — обучение, атаку, что угодно.

---

## Главный принцип: сначала локально, потом ClearML

Результаты **всегда** сначала печатаются/отображаются в самом ноутбуке, и только потом — попытка отправить их в ClearML, обёрнутая в `try/except`. Если у кого-то ClearML не настроен (нет доступа, не тот workspace, не подключены секреты) — он всё равно увидит все свои результаты в ноутбуке, просто без общего дашборда. ClearML — это *дополнение*, а не единственное место, где видны результаты.

```python
# --- локальный вывод: делается ВСЕГДА, до ClearML ---
print(summary_df)   # или display(summary_df)
plt.show()           # все графики

# --- ClearML: опционально, обёрнуто так, чтобы не ломать то, что выше ---
try:
    from clearml import Task
    ...
except Exception as e:
    print(f"ClearML недоступен ({e}) — результаты выше видны локально.")
```

---

## Разбор рабочего примера

```python
try:
    from clearml import Task

    task = Task.init(project_name="adversarial-attacks", task_name="fgsm-resnet-vs-vit-cifar")
    logger = task.get_logger()

    for name, accs in results.items():
        for eps_num, acc in zip(EPS_LIST, accs):
            logger.report_scalar(title="robust_accuracy_vs_eps", series=name, iteration=eps_num, value=acc)

    logger.report_table(title="fgsm_comparison", series="summary", iteration=0, table_plot=summary_df)
    logger.report_matplotlib_figure(title="accuracy_vs_epsilon", series="comparison", figure=fig, iteration=0)
    for name, ex_fig in example_figs.items():
        logger.report_matplotlib_figure(title="adversarial_examples", series=name, figure=ex_fig, iteration=0)

    print("\nРезультаты также залогированы в ClearML (project: adversarial-attacks).")
except Exception as e:
    print(f"\nClearML недоступен ({e}) — результаты выше видны локально в ноутбуке, в общий дашборд не ушли.")
```

По кусочкам:

### `Task.init(project_name=..., task_name=...)`
`project_name` — всегда `"adversarial-attacks"`, один на всех. `task_name` — по схеме `<модель>-<датасет>-<что делаем>`, например: `vit-tiny-imagenet100-train`, `resnet18-cifar10-train`, `pgd-resnet-vs-vit`. Смотрите `clearml-howto.md` — там та же договорённость.

### `logger.report_scalar(title=..., series=..., iteration=..., value=...)`
Один вызов — одна точка на графике. Вызывается в цикле, чтобы получилась целая кривая.

- **`title`** — имя всего графика (осей). У всех, кто рисует "то же самое", должно совпадать, чтобы линии легли на один график.
- **`series`** — имя конкретной линии *внутри* графика. В примере — `name` (`resnet18`/`vit_tiny`), поэтому обе модели оказываются на одном графике, а не на двух разных.
- **`iteration`** — это **не обязательно эпоха**, несмотря на название. Это то, что должно быть на оси X — у вас может быть `eps_num` (как в примере), номер эпохи, размер патча, число пикселей — что угодно возрастающее числом.

### `logger.report_table(title=..., series=..., iteration=0, table_plot=<DataFrame>)`
Кладёт обычный `pandas.DataFrame` как табличку. `iteration=0`, если это не часть серии, а разовая итоговая таблица.

### `logger.report_matplotlib_figure(title=..., series=..., figure=<fig>, iteration=0)`
Любой `matplotlib`-график/картинка целиком — можно логировать хоть график сравнения, хоть картинки с примерами.

### `try/except` вокруг всего
Если `Task.init()` упадёт (нет доступа/не настроено) — ловится, печатается понятное сообщение, но выполнение ноутбука не прерывается, и то, что напечатано/показано до этого блока, никуда не девается.

---

## Шаблон — скопируйте и заполните под свой эксперимент

```python
# --- локальный вывод: ВСЕГДА до ClearML ---
my_summary_df = pd.DataFrame({...})   # TODO: ваша итоговая таблица
print("Итоговая таблица:")
display(my_summary_df)

my_fig = plt.figure(...)
# TODO: ваш график
plt.show()

# --- ClearML ---
try:
    from clearml import Task

    task = Task.init(project_name="adversarial-attacks", task_name="TODO-осмысленное-имя")
    logger = task.get_logger()

    # TODO: report_scalar в цикле, если у вас есть кривая (метрика vs что-то возрастающее)
    for x_value, y_value in zip(X_AXIS_VALUES, Y_VALUES):
        logger.report_scalar(title="TODO-название-графика", series="TODO-название-линии",
                              iteration=x_value, value=y_value)

    logger.report_table(title="TODO-название-таблицы", series="summary", iteration=0, table_plot=my_summary_df)
    logger.report_matplotlib_figure(title="TODO-название-графика", series="TODO", figure=my_fig, iteration=0)

    print("\nРезультаты также залогированы в ClearML (project: adversarial-attacks).")
except Exception as e:
    print(f"\nClearML недоступен ({e}) — результаты выше видны локально в ноутбуке.")
```

---

## Примеры под разные типы экспериментов

**Обучение (loss/accuracy по эпохам)** — если используете `LightningModule` с `self.log(...)` и вызвали `Task.init()` до создания `Trainer` — этот блок вообще не нужен, метрики уходят в ClearML автоматически (см. `clearml-howto.md`, раздел 4). Ручное логирование через `report_scalar` нужно, только если считаете метрики **вне** Lightning-цикла.

**Сравнение атак с варьируемым параметром** (как FGSM/PGD/Patch/OnePixel) — ровно шаблон выше, `iteration` = epsilon/размер патча/число пикселей, `series` = имя модели.

**Разовый результат без кривой** (например, просто итоговая таблица финальных чисел, без графика по возрастающему параметру) — можно вообще пропустить цикл с `report_scalar` и оставить только `report_table` с `iteration=0`.

---

## Частые ошибки

- **`title` разный у одного и того же сравнения** — если один человек залогирует `"robust_accuracy_vs_eps"`, а другой (для той же атаки, тех же моделей) — `"accuracy_vs_epsilon"`, получатся два разных графика вместо одного общего.
- **`series` = что-то неинформативное** (`"run1"`, `"test"`) — потом невозможно понять на графике, какая линия чья. Используйте имя модели/конфигурации.
- **`Task.init()` внутри цикла** — создаёт новую задачу на каждой итерации вместо одной. Вызывается один раз в начале.
