import math
import struct

import sqlite_vec
from django.db import DatabaseError, migrations


BATCH_SIZE = 500
NORM_TOLERANCE = 1e-4


def _write_embedding(cursor, rowid, embedding):
    try:
        cursor.execute("UPDATE chunk_embeddings_vec SET embedding = %s WHERE rowid = %s", [embedding, rowid])
    except DatabaseError:
        cursor.execute("DELETE FROM chunk_embeddings_vec WHERE rowid = %s", [rowid])
        cursor.execute("INSERT INTO chunk_embeddings_vec(rowid, embedding) VALUES (%s, %s)", [rowid, embedding])
        return
    if cursor.rowcount == 0:
        cursor.execute("DELETE FROM chunk_embeddings_vec WHERE rowid = %s", [rowid])
        cursor.execute("INSERT INTO chunk_embeddings_vec(rowid, embedding) VALUES (%s, %s)", [rowid, embedding])


def normalize_vector_table(connection, batch_size=BATCH_SIZE):
    if connection.vendor != "sqlite":
        return 0
    if "chunk_embeddings_vec" not in set(connection.introspection.table_names()):
        return 0

    updated = 0
    last_rowid = None
    with connection.cursor() as cursor:
        cursor.execute("SELECT vec_version()")
        cursor.fetchone()
        while True:
            if last_rowid is None:
                cursor.execute(
                    "SELECT rowid, embedding FROM chunk_embeddings_vec ORDER BY rowid LIMIT %s",
                    [batch_size],
                )
            else:
                cursor.execute(
                    "SELECT rowid, embedding FROM chunk_embeddings_vec WHERE rowid > %s ORDER BY rowid LIMIT %s",
                    [last_rowid, batch_size],
                )
            rows = cursor.fetchall()
            if not rows:
                break
            last_rowid = rows[-1][0]
            for rowid, packed_vector in rows:
                vector_bytes = bytes(packed_vector)
                if len(vector_bytes) % 4:
                    raise ValueError(f"Invalid float32 vector at rowid {rowid}")
                vector = struct.unpack(f"<{len(vector_bytes) // 4}f", vector_bytes)
                norm = math.sqrt(sum(value * value for value in vector))
                if not norm or abs(norm - 1.0) <= NORM_TOLERANCE:
                    continue
                normalized = [value / norm for value in vector]
                _write_embedding(cursor, rowid, sqlite_vec.serialize_float32(normalized))
                updated += 1
    return updated


def renormalize_vectors(apps, schema_editor):
    updated = normalize_vector_table(schema_editor.connection)
    print(f"Renormalized {updated} vectors in chunk_embeddings_vec")


class Migration(migrations.Migration):
    dependencies = [
        ("personal_knowledge_base", "0015_agent_tool_approval"),
    ]

    operations = [
        migrations.RunPython(renormalize_vectors, reverse_code=migrations.RunPython.noop),
    ]
