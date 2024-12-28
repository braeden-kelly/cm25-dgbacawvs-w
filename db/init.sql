CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS meals (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    description TEXT,
    embedding VECTOR,
    created_at TIMESTAMPTZ DEFAULT NOW()
);