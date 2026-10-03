# Сервис баз знаний (knowledge_service)

Самостоятельный компонент продукта (наравне с AgentsCore, AgentsApp, MCP-сервером и планировщиком): загрузка документов в базы знаний, разбор, разбиение на фрагменты, эмбеддинги и поиск для RAG.

- **AgentsApp** обращается к сервису напрямую: создаёт базы знаний, добавляет контент, смотрит статус, документы и фрагменты, проверяет поиск.
- **AgentsCore** вызывает `POST /api/v1/retrieval`, когда у чата включено «Использовать RAG», подставляет найденные фрагменты в запрос к модели и сохраняет источники в ответе.

Генерацию ответа сервис не делает. В API база знаний называется **коллекцией** (`/api/v1/collections`).

## Запуск

```bash
pip install -r requirements.txt
cp .env.example .env        # KB_ROOTS, OLLAMA_URL и т. д.
ollama pull qwen3-embedding:0.6b
python -m knowledge_service
```

По умолчанию сервис слушает `0.0.0.0:8003`. Swagger UI доступен на `http://<host>:8003/docs`. Данные хранятся в `KNOWLEDGE_DATA_DIR` (по умолчанию `knowledge_data/`): база `knowledge.sqlite` и копии загруженных файлов и страниц.

При старте сервис печатает:

- какой `.env` прочитан;
- каталог данных;
- модель эмбеддингов по умолчанию;
- доступность провайдеров эмбеддингов;
- модели-реранкеры из `RERANK_PROVIDERS_FILE`;
- разрешённые каталоги `KB_ROOTS`;
- внутренние хосты и секреты (только имена);
- поддерживаемые форматы.

Миграций базы нет. Если после обновления сервиса схема изменилась, сервис не стартует и пишет, каких колонок не хватает: остановите его, удалите `knowledge.sqlite` (и `-wal`, `-shm` рядом) и запустите заново — базы знаний нужно загрузить повторно.

AgentsCore указывает на сервис переменной `KNOWLEDGE_SERVICE_URL=http://<host>:8003`, AgentsApp — адресом «Адрес сервиса баз знаний» в «Настройках».

## Как это работает

1. Запрос на загрузку (файлы, источник, переиндексация) создаёт **задачу** и сразу возвращает `202` с `job_id`.
2. Рабочий поток (`KB_WORKERS`) проводит каждый документ через этапы: загрузка → разбор → разбиение → эмбеддинги → запись в индекс.
3. Прогресс отдаёт `GET /api/v1/jobs/{id}` или SSE-поток `GET /api/v1/jobs/{id}/events`.
4. Статусы документа: `queued`, `fetching`, `parsing`, `chunking`, `embedding`, `indexed`, `failed`, `skipped`, `cancelled`.
5. Повторная синхронизация источника переиндексирует только изменившиеся документы: сравнивается хеш содержимого, способ разбиения и модель. Пропавшие документы удаляются.
6. Документ с тем же текстом, что уже есть в базе, пропускается как дубликат.
7. При перезапуске прерванные задачи получают статус «прервано перезапуском», задачи из очереди выполняются.

### Способы передачи контента

| Способ | Как |
|---|---|
| Файлы | `POST /collections/{id}/documents`, `multipart/form-data`: одно или несколько полей `file`, необязательно `metadata`, `chunking`, `jsonl` (JSON-строки) |
| Путь, папка, glob | `{"source": {"type": "path", "path": "/data/docs"}}` или `"/data/**/*.md"`, маски `include`/`exclude`; только внутри `KB_ROOTS`, с проверкой символических ссылок |
| Ссылка | `{"source": {"type": "url", "url": "https://..."}}`, обход ссылок `crawl: {depth, same_domain, max_pages}`, авторизация `auth: {"secret": "ИМЯ", "header": "Authorization", "scheme": "Bearer"}` |
| Текст | `{"source": {"type": "text", "title": "...", "content": "...", "format": "markdown"}}` |

Git-репозитории и S3/GCS в первую итерацию не входят: API отвечает, что такой тип источника «пока не поддерживается».

**Ссылки.** Разрешены только публичные адреса, и это проверяется после каждого редиректа. Внутренние хосты (например, Confluence в локальной сети) добавляются в `KB_ALLOWED_HOSTS`.

**Секреты для авторизации** описываются в `KB_SECRETS`:

- формат — `ИМЯ@хост1|хост2`;
- значение секрета берётся из переменной окружения с тем же именем;
- заголовок отправляется только на привязанные к секрету хосты;
- в базе и в ответах API значение секрета не хранится.

Notion формирует страницы через JavaScript, поэтому читаются только те страницы, которые отдают текст в HTML.

### Форматы

| Группа | Расширения |
|---|---|
| Текст | `.txt`, `.md`, `.markdown`, `.rst`, `.html`, `.htm` |
| Документы | `.pdf` (постранично, без повторяющихся колонтитулов), `.docx`, `.pptx`, `.xlsx` |
| Код | по умолчанию `.py .pyi .kt .kts .java .js .ts .tsx .jsx .go .sql .yaml .yml .json .toml .xml .gradle .properties .sh`; список — настройка коллекции `file_types.code_extensions` |
| Структурированное | `.jsonl`: одна запись — один фрагмент; поля `jsonl.text_fields` (текст) и `jsonl.metadata_fields` (метаданные фрагмента) |

Любое расширение можно объявить текстовым через `file_types.extra_text_extensions`. Для скана без текстового слоя документ получает ошибку «нужно распознавание текста (OCR)».

### Разбиение

Способ разбиения — настройка коллекции `chunking.method`. Его можно переопределить для источника, для загрузки файлов или для одного документа при переиндексации.

- **`structure` («По структуре», по умолчанию).**
  - Разделы по заголовкам 1–3 уровня (Markdown, reStructuredText, HTML, DOCX).
  - Страницы и нумерованные заголовки PDF.
  - Слайды PPTX, листы XLSX.
  - Объявления кода: Python — через `ast`; Kotlin, Java, JS/TS, Go — классы, функции и методы.
  - SQL — по операторам; YAML, JSON, TOML — по ключам и секциям.
  - Записи JSONL.
  - Разделы длиннее `max_chunk_tokens` делятся по абзацам, затем строкам, затем предложениям. Короткие (меньше `min_chunk_tokens`) склеиваются с соседними.
- **`fixed` («Фиксированный размер»).** Окно `chunk_size_tokens` с перекрытием `chunk_overlap_tokens`. Граница окна сдвигается к концу абзаца или предложения.

Для эмбеддинга к тексту фрагмента добавляются название документа и путь раздела.

### Метаданные

У документа три поля:

- **`properties`** — свойства, извлечённые из самого файла:
  - PDF: страницы, автор, тема, ключевые слова, программа, даты создания и изменения;
  - DOCX и PPTX: автор, тема, ключевые слова, категория, даты, число таблиц или слайдов, версия (переходит в `doc_version`);
  - XLSX: листы, строки, автор;
  - Markdown: поля front matter (`version` переходит в `doc_version`);
  - HTML: `<meta>` description, author, keywords, даты статьи, язык страницы;
  - код: язык, число строк и объявлений;
  - JSONL: число записей;
  - у всех форматов — число разделов.
- **`user_metadata`** — поля, заданные пользователем: при загрузке (`metadata` в multipart), у источника или через `PATCH /documents/{id}`.
- **`metadata`** — итог: `properties`, поверх них `user_metadata`.

Дату и версию документа (`doc_date`, `doc_version`) можно переопределить. Поиск по ним выбирает более свежий документ.

У фрагмента `metadata` — полные метаданные:

- служебные поля: `chunk_id`, `collection_id`, `document_id`, `source_id`, `source_type`, `source`, `title`, `filename`, `section`, `page`, `char_start` / `char_end`, `tokens`, `chunking_method`, `origin`, `doc_type`, `code_language`, `doc_date`, `doc_version`, `language`, `content_hash`;
- метаданные документа;
- собственные поля фрагмента (поля записи JSONL) — они же отдельно в `chunk_metadata`.

Те же `metadata` возвращает поиск. Фильтр `filters.metadata` проверяет поля фрагмента, пользователя и свойства файла.

### Эмбеддинги и поиск

Модель задаётся при создании коллекции как «провайдер/модель»:

- **по умолчанию** — `ollama/qwen3-embedding:0.6b`;
- **любая другая модель Ollama** — `ollama/<модель>`;
- **OpenAI-совместимые провайдеры** (OpenAI, vLLM, LM Studio) — из файла `EMBEDDING_PROVIDERS_FILE` (см. `embedding_providers.example.json`).

Смена модели через `PATCH` запускает переиндексацию коллекции. Векторы нормируются по длине (L2) и кэшируются по хешу текста. Для Qwen3-Embedding, nomic-embed-text и E5 автоматически добавляются нужные префиксы запроса и документа.

Фильтры поиска: `document_ids`, `source_ids`, `doc_types`, `source_types`, `languages`, `doc_date: {gte, lte}`, `metadata: {ключ: значение | [значения]}`.

### Реранкинг и фильтрация

Поиск идёт в два этапа:

1. **Векторный поиск.** Косинусное сходство (скалярное произведение нормированных векторов, numpy по индексу в памяти) → топ `candidate_k` кандидатов (по умолчанию max(20, `top_k`), до 200) → порог `score_threshold` → схлопывание почти одинаковых фрагментов (одинаковый хеш текста или сходство векторов > 0,97).
2. **Реранкинг** (`rerank`) → порог `rerank_threshold` → топ `top_k`:
   - `none` — «Нет»: итоговый балл — векторное сходство, `rerank_threshold` не применяется;
   - `heuristic` — «Эвристика (без LLM)»: 0,6 × векторное сходство + 0,4 × BM25 по словам вопроса среди кандидатов (основа слова — первые 6 символов, без стоп-слов), нормированный к лучшему кандидату;
   - `model` — «Модель-реранкер»: cross-encoder оценивает пары «вопрос — фрагмент» (фрагмент передаётся с префиксом «документ › раздел»). Модель — `rerank_model` («провайдер/модель») или модель по умолчанию.

Баллы обоих способов лежат в 0..1: сырые логиты cross-encoder переводятся сигмоидой. Если модель-реранкер не настроена или недоступна, применяется эвристика, а в ответе будет `rerank.fallback=true` и причина.

Модели-реранкеры (одна или несколько) описываются в файле `RERANK_PROVIDERS_FILE` (см. `rerank_providers.example.json`). Для каждого провайдера указываются `base_url`, `models`, `title`, `api_key` (ссылкой `${ПЕРЕМЕННАЯ}`) и `api`:

- `jina` — `POST {base_url}/rerank` `{model, query, documents, top_n}` → `{results: [{index, relevance_score}]}`: llama.cpp server с `--reranking`, Infinity, Jina, Cohere;
- `tei` — `POST {base_url}/rerank` `{query, texts}` → `[{index, score}]`: Hugging Face Text Embeddings Inference.

Модель по умолчанию — `KB_DEFAULT_RERANK_MODEL`, иначе первая модель файла; таймаут — `KB_RERANK_TIMEOUT`. У Ollama официального API реранкинга нет, поэтому локально удобнее всего llama.cpp:

```bash
llama-server -m bge-reranker-v2-m3-Q8_0.gguf --reranking --port 8081
```

Ответ поиска, кроме фрагментов (`score` — итоговый балл, `vector_score`, `rerank_score`), содержит трассировку этапов:

```json
"stages": {"candidates": 20, "after_threshold": 12, "after_dedup": 11, "after_rerank": 4, "returned": 4},
"rerank": {"method": "heuristic", "model": null, "fallback": false, "error": null}
```

## REST API `/api/v1`

Ошибки возвращаются как `{"error": "..."}`, списки — как `{"items", "total", "limit", "offset"}`, время — в секундах Unix.

| Метод и путь | Назначение |
|---|---|
| `POST /collections` | создать базу: `name`, `description`, `embedding_model`, `chunking`, `file_types` |
| `GET /collections`, `GET /collections/{id}` | список со статусом (`empty`, `processing`, `ready`, `has_errors`) и статистикой; одна база |
| `PATCH /collections/{id}` | изменить настройки; смена модели эмбеддингов → `reindex_job_id` |
| `DELETE /collections/{id}` | удалить базу с документами и файлами |
| `POST /collections/{id}/reindex` | переиндексировать все документы → `202 {job_id}` |
| `POST /collections/{id}/documents` | загрузить файлы (multipart) → `202 {job_id, documents}` |
| `POST /collections/{id}/sources` | добавить источник → `202 {source, job_id}` |
| `GET /collections/{id}/sources`, `GET/PATCH/DELETE /sources/{id}` | источники |
| `POST /sources/{id}/sync` | синхронизировать источник → `202 {job_id}` |
| `GET /collections/{id}/documents` | документы: фильтры `status` (через запятую, `processing` — все в работе), `source_id`, `doc_type`, `q` |
| `GET/PATCH/DELETE /documents/{id}` | документ; `PATCH`: `title`, `doc_date`, `doc_version`, `metadata`, `enabled` |
| `GET /documents/{id}/content`, `GET /documents/{id}/file` | извлечённый текст; исходный файл |
| `PUT /documents/{id}/file` | заменить файл загруженного документа → `202` |
| `POST /documents/{id}/reindex` | переиндексировать документ, необязательно с `chunking` (`{}` — как у коллекции) → `202` |
| `GET /jobs/{id}`, `GET /jobs/{id}/events` | статус задачи; SSE-поток прогресса (`progress`, `done`) |
| `GET /collections/{id}/jobs`, `POST /jobs/{id}/cancel` | задачи коллекции; отмена |
| `GET /documents/{id}/chunks`, `POST /documents/{id}/chunks` | фрагменты; добавить фрагмент вручную (сохраняется при переиндексации) |
| `GET/PATCH/DELETE /chunks/{id}` | фрагмент; правка текста пересчитывает эмбеддинг |
| `POST /retrieval` | поиск: `query`, `collection_ids`, `top_k` (≤ 50), `candidate_k` (≥ `top_k`, ≤ 200), `score_threshold`, `rerank` (`none` / `heuristic` / `model`), `rerank_model`, `rerank_threshold`, `filters` → `results`, `stages`, `rerank` |
| `GET /rerank-models` | модели-реранкеры `{items: [{id, provider, model, provider_title, default}], default}` |
| `GET /health`, `/info`, `/embedding-models`, `/fs?path=` | состояние и доступность провайдеров; форматы, типы источников, способы реранкинга (`rerank_methods` с признаком `enabled`) и лимиты; модели эмбеддингов; просмотр каталогов внутри `KB_ROOTS` |

Если задан `KNOWLEDGE_API_KEY`, запросы (кроме `/health`) должны нести заголовок `Authorization: Bearer <ключ>`. В AgentsCore этот же ключ задаётся в `KNOWLEDGE_SERVICE_API_KEY`.

## Структура

- `config.py` — настройки из окружения и `.env`.
- `db.py` — SQLite.
- `formats.py`, `code_structure.py` — извлечение текста и структуры.
- `chunking.py` — оба способа разбиения.
- `embeddings.py` — провайдеры (Ollama, OpenAI-совместимые) и кэш.
- `rerank.py` — второй этап поиска: эвристика (BM25 + вектор) и модели-реранкеры.
- `fetch.py` — загрузка по ссылкам с защитой от внутренних адресов.
- `sources.py` — проверка и перечисление источников.
- `jobs.py` — рабочие потоки.
- `service.py` — вся логика.
- `api.py`, `app.py` — REST API.
- `__main__.py` — точка входа.

## Тесты

```bash
python -m unittest discover -s tests -v
```

Тесты работают без сети и без Ollama: эмбеддинги подменяются детерминированным «мешком слов», HTTP — `httpx.MockTransport`. Проверяются:

- все форматы и оба способа разбиения;
- все способы передачи контента, синхронизация, дубли, отмена и восстановление после перезапуска;
- защита путей и ссылок, фильтры поиска, REST API и SSE.

Тесты API требуют `fastapi`, тест PDF — `reportlab` и шрифт DejaVu; без них эти тесты пропускаются.
