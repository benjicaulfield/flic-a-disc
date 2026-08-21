import pytest


from django.db import connection

@pytest.mark.django_db
def test_show_actual_db():
    with connection.cursor() as cursor:
        cursor.execute("SELECT current_database(), current_schema()")
        db, schema = cursor.fetchone()
        print(f"\nDJANGO IS CONNECTED TO → database={db}, schema={schema}")



    