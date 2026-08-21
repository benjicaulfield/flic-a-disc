# Generated migration for knapsack session comparison feature

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bandit', '0041_alter_record_format'),
    ]

    operations = [
        migrations.AddField(
            model_name='knapsacksession',
            name='saved_for_comparison',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='knapsacksession',
            name='notes',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AlterModelOptions(
            name='knapsacksession',
            options={'ordering': ['-created_at']},
        ),
    ]
