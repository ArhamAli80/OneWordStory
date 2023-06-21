from django.db import models


# Create your models here.
class Story(models.Model):
    word = models.CharField(max_length=255)
    next_word = models.CharField(max_length=255)
# Create your models here.
