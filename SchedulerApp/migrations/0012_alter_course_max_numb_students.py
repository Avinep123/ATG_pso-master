# Generated by Django 3.2.25 on 2025-01-31 11:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SchedulerApp', '0011_course_max_numb_students'),
    ]

    operations = [
        migrations.AlterField(
            model_name='course',
            name='max_numb_students',
            field=models.IntegerField(default=50),
        ),
    ]
