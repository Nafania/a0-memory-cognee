# User Flow: memory_cognee в Agent Zero

Документ описывает ожидаемый flow глазами пользователя и оператора Agent Zero.
Это не backend/runtime contract: здесь важны видимые действия, статусы, логи и
момент, когда память реально usable.

## Роли и точки наблюдения

- Пользователь чата видит ответы агента, tool output, временные сообщения recall
  и содержимое Memory Dashboard.
- Оператор видит startup/background logs и может проверить статус rebuild через
  dashboard API `cognify_status` или внутренний статус worker.
- Agent Zero не должен требовать от пользователя понимать Cognee internals.
  Видимые состояния должны объяснять: память готова, временно устарела,
  перестраивается, недоступна или сломана.

## Запуск и первичный статус

1. Пользователь устанавливает plugin через **Settings -> Plugins -> Install**,
   Git URL или вручную в `usr/plugins/memory_cognee`.
2. После restart/toggle Agent Zero запускает `memory_cognee`, отключает builtin
   `_memory` и инициализирует Cognee storage.
3. Оператор в логах должен увидеть один итоговый startup status:
   - `Cognee startup readiness: READY; recall enabled.`
   - `Cognee startup readiness: DEGRADED; Cognee recall is enabled ...`
   - `Cognee startup readiness: BLOCKED; recall may be unavailable.`
   - `Cognee startup readiness: UNKNOWN; could not read dataset graph status: ...`
4. Если graph status удалось прочитать, дополнительно ожидаем строку
   `Cognee dataset graph status: ready=...; empty=...; errors=...; unknown=...`.
   При `UNKNOWN` или раннем `BLOCKED` из-за config/storage ошибки этой строки
   может не быть; тогда root cause должен быть виден в startup readiness log.
5. Если есть dirty readable dataset, нормальный лог:
   `Cognee memory graph rebuild pending for readable dataset(s); recall remains enabled while background rebuild catches up: [...]`.
6. Если readable snapshot нет, нормальный warning:
   `Cognee memory graph rebuild required before recall; background rebuild pending for dataset(s): [...]`.

Память usable, когда status `READY`, либо `DEGRADED` с readable graph. В
`dirty/readable=true` старая память usable, но новые/удаленные записи могут еще
не попасть в graph search. В `BLOCKED`, `UNKNOWN`, `dirty/readable=false` или
`failed/readable=false` память нельзя считать usable для recall.

## Memory Dashboard

Пользователь открывает Memory Dashboard из Agent Zero UI.

Ожидаемо видно:

- **Table View** и **Knowledge Graph**.
- **Memory Directory** с `default` и доступными subdir.
- Filters: area, limit, search.
- Loading text: `Initializing memory database...` при первом открытии subdir,
  затем `Searching memories...`.
- Status bar: `Total`, `Filtered`, `Knowledge`, `Conversation`.
- Для записи: area badge, source `Knowledge` или `Conversation`, timestamp,
  preview, details modal.
- Delete/copy/export actions и success/error toast, например
  `Memory deleted successfully`, `Successfully deleted N memories`,
  `Failed to search memories`.

Dashboard без search query показывает сохраненные source items. Search box в
dashboard использует memory search и зависит от readable graph. Он не обязан
доказывать, что graph уже перестроен и auto-recall увидит свежую запись в
текущем turn. Для rebuild/readiness оператор смотрит logs/status.

## Обычный chat turn с auto-recall

1. Пользователь отправляет сообщение в чат.
2. Если включен `memory_recall_enabled`, Agent Zero периодически создает util log
   `Searching memories...` по `memory_recall_interval`.
3. Синхронно для текущего turn допустимы только короткая query prep и bounded
   search. Rebuild/migration/cognify не должны выполняться на пути ответа.
4. Если recall успел и graph searchable, найденные memories/solutions попадают в
   context текущего ответа. Пользователь не обязан видеть отдельный tool output.
5. Если recall ничего не нашел при доступном graph, это обычный empty result.
6. Если recall заблокирован, ответ чата продолжается без памяти, а util log
   получает один из headings:
   - `Memory rebuild pending; skipping recall`
   - `Memory rebuild failed; skipping recall`
   - `Memory rebuild stale; skipping recall`
   - `Memory rebuild in progress; skipping recall`
7. Если включен delayed recall (`memory_recall_delayed=true`) и search еще идет,
   пользователь видит:
   `Info: auto memory recall set to delayed mode. auto memories will be available after next message. if manual memory check is required use memory tools.`

Правильное поведение: chat остается responsive. Нельзя ждать минуты ради rebuild,
нельзя маскировать blocked/failed graph как "memories not found".

## Explicit `memory_load`

Пользователь или агент явно вызывает `memory_load` с query, threshold, limit и
опциональным filter.

Синхронно видно:

- Если search доступен и есть результаты: plain-text список memories.
- Если search доступен и результатов нет: стандартное `memories not found`.
- Если graph blocked/failed/unavailable:
  `Memory search unavailable: <reason>`.

`memory_load` должен отличать "ничего не найдено" от "искать сейчас нельзя".
Для оператора `<reason>` должен указывать pending/running/failed rebuild, а не
прятаться в generic exception.

## `memory_save`, delete и forget

`memory_save` нужен, когда пользователь явно просит запомнить факт.

Синхронно видно:

- Tool возвращает `Memory saved with id <memory_id>`.
- Source data сохранена, metadata/area записаны.
- Dataset помечен `dirty`.

Async после ответа:

- Background worker позже запускает rebuild/cognify для dirty dataset.
- Если dataset был readable, old snapshot остается usable до rebuild. Новая
  память должна стать доступна auto-recall/graph search после успешного rebuild
  и readiness verification.
- Если readable snapshot не было, search blocked до успешного rebuild.

Delete/forget:

- `memory_delete` удаляет по id, `memory_forget` удаляет по query/threshold.
- Синхронно видно количество удаленных memories через prompt удаления или UI
  toast.
- Если query-delete не может выполнить search из-за blocked/failed graph, user
  должен увидеть explicit unavailable/failure, а не успешный `0 deleted`.
- Dataset становится `dirty`; удаление может еще отображаться в старом readable
  graph до rebuild, но source list/dashboard должен обновляться по факту delete.

## Автоматическое запоминание после turn

После ответа Agent Zero может асинхронно извлекать fragments/solutions/session
memory.

В логах ожидаемо:

- `Memorizing new information...`
- `N entries queued for memory write.`
- `Memorizing succesful solutions...`
- `N solutions queued for memory write.`

Это background work. Пользовательский ответ уже отдан. Новая автоматическая
память не обязана быть доступна в том же turn и становится searchable только
после write + rebuild. Если queue/consolidation/search unavailable, это должно
быть видно в warning/status logs, а не пропадать молча.

## Dirty, readable, rebuild

- `ready/readable=true`: память usable.
- `dirty/readable=true`: память usable из последнего readable snapshot; свежие
  writes/deletes ждут background rebuild.
- `dirty/readable=false`: source data есть, но recall/search blocked до rebuild.
- `rebuilding/readable=true`: может быть usable, но active Cognee operation gate
  иногда заставит search быстро skip/timeout с явной причиной.
- `rebuilding/readable=false`: unavailable до successful rebuild.
- `failed/readable=true`: degraded; old snapshot usable, retry pending.
- `failed/readable=false`: blocked; explicit tools возвращают unavailable,
  auto-recall skip.

Rebuild считается успешным не по факту завершения job, а когда graph/vector
readiness подтверждает non-empty readable graph для dataset с data.

## Что делать при degraded path

`DEGRADED`:

- Chat можно использовать.
- Оператор проверяет, что readable graph есть.
- Incomplete FAISS migration или non-blocking maintenance не должны заставлять
  user path делать full rebuild.

`BLOCKED` / `dirty readable=false`:

- Не обещать пользователю, что memory recall работает.
- Ждать background rebuild или запустить операторскую диагностику worker status.
- `memory_load` должен вернуть `Memory search unavailable: ...`.

`FAILED`:

- Если readable snapshot есть, можно продолжать chat с предупреждением, что
  свежесть не гарантирована.
- Если readable snapshot нет, memory unavailable до retry/fix.
- Оператор смотрит `last_error`, retry state, graph errors, embedding config и
  startup logs.

`UNKNOWN`:

- Не считать READY.
- Оператор проверяет доступность Cognee graph/vector storage и повторяет startup
  или статусный check.

`UNAVAILABLE` на explicit tool:

- Пользователю/агенту нужно продолжить без памяти или повторить позже.
- Сообщение должно объяснять причину, например rebuild pending/running/failed.

## Happy path

1. Plugin installed, Agent Zero restarted.
2. Logs: `_memory` disabled, `Cognee startup readiness: READY; recall enabled.`
3. Пользователь открывает Memory Dashboard, видит memories, counts, graph/table.
4. Пользователь спрашивает вопрос, связанный с прошлой памятью.
5. Util log: `Searching memories...`; текущий turn получает relevant memories.
6. Пользователь просит: "запомни X".
7. Tool returns `Memory saved with id ...`.
8. Logs/status: dataset dirty/readable, recall remains enabled.
9. После idle + successful rebuild новая память появляется в `memory_load`,
   auto-recall и Knowledge Graph.

## Degraded path

1. Startup logs: `DEGRADED` или dirty readable dataset pending rebuild.
2. Chat работает, auto-recall использует last readable snapshot.
3. Fresh save/delete возвращает синхронный success, но свежесть search не
   гарантирована до rebuild.
4. Если active rebuild мешает search, auto-recall пишет `Memory rebuild in progress; skipping recall`; explicit `memory_load` возвращает unavailable.
5. После successful rebuild logs/status переходят к ready, dirty cleared.

## Blocked path

1. Startup logs: `BLOCKED` или `UNKNOWN`, либо worker status показывает
   `dirty/readable=false` или `failed/readable=false`.
2. Auto-recall не блокирует ответ и пишет skip heading.
3. `memory_load` возвращает `Memory search unavailable: <reason>`.
4. Dashboard может показать source items, но это не означает usable recall.
5. Оператор устраняет root cause и ждет successful rebuild/readiness. Только
   после этого память снова считается usable.

## Acceptance Criteria

- Startup flow документирован через видимые `READY`, `DEGRADED`, `BLOCKED`,
  `UNKNOWN` logs.
- Указано, когда память usable: `READY` или readable snapshot; blocked states не
  трактуются как empty result.
- Обычный chat turn описывает auto-recall, delayed mode, skip headings и
  отсутствие long rebuild wait на user path.
- `memory_load` описан отдельно от auto-recall и возвращает
  `Memory search unavailable: ...` при blocked/failed graph.
- `memory_save`, delete и forget разделяют synchronous user-visible result и
  async rebuild availability.
- Dirty/readable/rebuilding/failed semantics понятны оператору без чтения кода.
- Dashboard expectations ограничены реально видимыми table/graph/search/counts/
  toast элементами; readiness не выдуман как UI-only indicator.
- Happy, degraded и blocked paths дают проверяемые шаги и ожидаемые сообщения.
- Документ не требует изменений кода и не вводит новый backend/runtime contract.
