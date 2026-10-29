-- name: source_rows
SELECT COUNT(*) AS source_rows FROM support_current_raw;

-- name: distinct_ticket_ids
SELECT COUNT(DISTINCT ticket_id) AS distinct_ticket_ids FROM support_current_raw;

-- name: ticket_id_copies
SELECT COUNT(*) AS ticket_ids_with_copies, COALESCE(SUM(copies - 1), 0) AS extra_copies
FROM (SELECT ticket_id, COUNT(*) AS copies FROM support_current_raw GROUP BY ticket_id HAVING COUNT(*) > 1);

-- name: empty_customer_id
SELECT COUNT(*) AS empty_customer_id
FROM support_current_raw
WHERE customer_id IS NULL OR TRIM(customer_id) = '';

-- name: customer_id_with_spaces
SELECT COUNT(*) AS customer_id_with_spaces
FROM support_current_raw
WHERE customer_id <> TRIM(customer_id);

-- name: created_at_without_zone
SELECT COUNT(*) AS created_at_without_zone
FROM support_current_raw
WHERE created_at NOT LIKE '%Z';

-- name: rows_by_topic
SELECT COALESCE(NULLIF(topic, ''), '<empty>') AS topic, COUNT(*) AS rows_by_topic
FROM support_current_raw
GROUP BY 1 ORDER BY 1;

-- name: satisfaction_out_of_range
SELECT COUNT(*) AS satisfaction_out_of_range
FROM support_current_raw
WHERE satisfaction_score <> '' AND CAST(satisfaction_score AS INTEGER) NOT BETWEEN 1 AND 5;
