-- Извлечение текущих обращений из SQLite (режим только чтения).
-- Все поля остаются TEXT: сырые значения нужны для аудита качества,
-- приведение типов и часовых поясов выполняется в stage2_prepare.py.
SELECT ticket_id, customer_id, created_at, available_at,
       topic, channel, resolved_at, resolution_available_at,
       satisfaction_score
FROM support_current_raw
ORDER BY rowid;
