import duckdb

con = duckdb.connect("metadata.duckdb")
con.execute("""
CREATE TABLE IF NOT EXISTS metadata (
    id INTEGER PRIMARY KEY,
    document_class TEXT,
    filename TEXT,
    page_number INTEGER
)
""")

# Example insert
con.execute("""
INSERT INTO metadata (id, document_class, filename, page_number)
VALUES (0, 'recipe', 'my_recipes.pdf', 1)
""")

con.close()