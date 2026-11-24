-- name: budget_by_province
SELECT
    COALESCE(province, '未识别') AS province,
    COUNT(*) AS project_count,
    ROUND(SUM(budget_yuan), 2) AS budget_sum_yuan,
    ROUND(AVG(budget_yuan), 2) AS budget_avg_yuan,
    MAX(budget_yuan) AS budget_max_yuan
FROM v_project_flat
GROUP BY province
ORDER BY budget_sum_yuan DESC;

-- name: agency_ranking
WITH agency_budget AS (
    SELECT
        agency_name,
        COUNT(*) AS project_count,
        ROUND(SUM(budget_yuan), 2) AS budget_sum_yuan
    FROM v_project_flat
    WHERE agency_name IS NOT NULL
    GROUP BY agency_name
)
SELECT
    RANK() OVER (ORDER BY project_count DESC, budget_sum_yuan DESC) AS agency_rank,
    agency_name,
    project_count,
    budget_sum_yuan
FROM agency_budget
ORDER BY agency_rank, agency_name;

-- name: deadline_notice
SELECT
    doc_id,
    project_name,
    publish_time,
    bid_deadline,
    ROUND(julianday(bid_deadline) - julianday(publish_time), 2) AS days_to_deadline,
    CASE
        WHEN bid_deadline IS NOT NULL
         AND julianday(bid_deadline) - julianday(publish_time) < 20
        THEN 1 ELSE 0
    END AS under_20_days
FROM v_project_flat
ORDER BY days_to_deadline, doc_id;

-- name: field_quality
SELECT
    field,
    COUNT(*) AS extraction_count,
    SUM(in_review) AS review_count,
    ROUND(1.0 * SUM(in_review) / COUNT(*), 4) AS review_rate,
    SUM(correct) AS correct_count,
    ROUND(1.0 * SUM(correct) / COUNT(*), 4) AS correct_rate
FROM field_extraction
GROUP BY field
ORDER BY field;

-- name: cumulative_budget
WITH daily_budget AS (
    SELECT
        date(publish_time) AS publish_date,
        ROUND(SUM(budget_yuan), 2) AS daily_budget_yuan
    FROM v_project_flat
    GROUP BY date(publish_time)
)
SELECT
    publish_date,
    daily_budget_yuan,
    ROUND(
        SUM(daily_budget_yuan) OVER (
            ORDER BY publish_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS cumulative_budget_yuan
FROM daily_budget
ORDER BY publish_date;

-- name: organizations_with_aliases
SELECT
    organization.org_id,
    organization.canonical_name,
    organization.display_name,
    COUNT(organization_alias.alias) AS alias_count,
    GROUP_CONCAT(organization_alias.alias, ' | ') AS aliases
FROM organization
JOIN organization_alias USING (org_id)
GROUP BY organization.org_id, organization.canonical_name, organization.display_name
HAVING COUNT(organization_alias.alias) > 1
ORDER BY alias_count DESC, organization.canonical_name;

-- name: data_quality
SELECT 'notices' AS check_name, COUNT(*) AS actual, 150 AS expected,
       COUNT(*) = 150 AS passed
FROM notice
UNION ALL
SELECT 'gkzb_projects', COUNT(*), 60, COUNT(*) = 60
FROM tender_project
UNION ALL
SELECT 'field_rows', COUNT(*), 360, COUNT(*) = 360
FROM field_extraction
UNION ALL
SELECT 'missing_evidence_for_value', COUNT(*), 0, COUNT(*) = 0
FROM field_extraction
WHERE value IS NOT NULL AND (evidence IS NULL OR trim(evidence) = '');
