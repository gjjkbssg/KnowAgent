from django.db import migrations


def retokenize_fts(apps, schema_editor):
    connection = schema_editor.connection
    if "chunks_fts" not in set(connection.introspection.table_names()):
        return

    from personal_knowledge_base.search import retokenize_fts_rows

    with connection.cursor() as cursor:
        retokenize_fts_rows(cursor)


class Migration(migrations.Migration):
    dependencies = [("personal_knowledge_base", "0013_multimodal_document_images")]

    operations = [
        migrations.RunPython(retokenize_fts, reverse_code=migrations.RunPython.noop),
    ]
