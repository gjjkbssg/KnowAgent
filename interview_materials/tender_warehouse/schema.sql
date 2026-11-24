PRAGMA foreign_keys = ON;

DROP VIEW IF EXISTS v_review_queue;
DROP VIEW IF EXISTS v_project_flat;
DROP TABLE IF EXISTS field_extraction;
DROP TABLE IF EXISTS tender_project;
DROP TABLE IF EXISTS organization_alias;
DROP TABLE IF EXISTS notice;
DROP TABLE IF EXISTS organization;

CREATE TABLE organization (
    org_id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    province TEXT
);

CREATE TABLE organization_alias (
    alias TEXT PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organization(org_id)
);

CREATE TABLE notice (
    doc_id TEXT PRIMARY KEY,
    notice_type TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    publish_time TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('dev', 'test'))
);

CREATE TABLE tender_project (
    doc_id TEXT PRIMARY KEY REFERENCES notice(doc_id),
    project_name TEXT,
    project_code TEXT,
    purchaser_id INTEGER REFERENCES organization(org_id),
    agency_id INTEGER REFERENCES organization(org_id),
    budget_yuan REAL,
    bid_deadline TEXT,
    needs_review INTEGER NOT NULL CHECK (needs_review IN (0, 1))
);

CREATE TABLE field_extraction (
    doc_id TEXT NOT NULL REFERENCES notice(doc_id),
    field TEXT NOT NULL,
    value TEXT,
    evidence TEXT,
    source TEXT,
    in_review INTEGER NOT NULL CHECK (in_review IN (0, 1)),
    gold TEXT,
    correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
    PRIMARY KEY (doc_id, field)
);

CREATE INDEX idx_notice_publish_time ON notice(publish_time);
CREATE INDEX idx_field_extraction_review ON field_extraction(in_review, correct);
CREATE INDEX idx_organization_alias_org ON organization_alias(org_id);

CREATE VIEW v_review_queue AS
SELECT
    field_extraction.doc_id,
    notice.title,
    field_extraction.field,
    field_extraction.value,
    field_extraction.evidence,
    field_extraction.source,
    field_extraction.in_review,
    field_extraction.gold,
    field_extraction.correct
FROM field_extraction
JOIN notice USING (doc_id)
WHERE field_extraction.in_review = 1
   OR field_extraction.correct = 0;

CREATE VIEW v_project_flat AS
SELECT
    tender_project.doc_id,
    tender_project.project_name,
    tender_project.project_code,
    tender_project.budget_yuan,
    tender_project.bid_deadline,
    tender_project.needs_review,
    notice.publish_time,
    purchaser.display_name AS purchaser_name,
    purchaser.province AS province,
    agency.display_name AS agency_name
FROM tender_project
JOIN notice USING (doc_id)
LEFT JOIN organization AS purchaser ON purchaser.org_id = tender_project.purchaser_id
LEFT JOIN organization AS agency ON agency.org_id = tender_project.agency_id;
