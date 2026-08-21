from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bandit', '0045_add_skipped_field'),
    ]

    operations = [
        migrations.AddField(
            model_name='record',
            name='is_six',
            field=models.BooleanField(default=False),
        ),
    ]
