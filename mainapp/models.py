from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError

# Create your models here.

def file_size(value): 
    limit = 2 * 1024 * 1024
    if value.size > limit:
        raise ValidationError('File too large. Size should not exceed 2 MiB.')

def validate_file_extension(value): 
    if not value.name.endswith('csv'):
        raise ValidationError('Only CSV file is accepted')


class Data(models.Model):

  datatype_choices = [
    ('normal', 'Normal'),
    ('freq', 'High Frequency'),
    ('time', 'Time Series')
  ]

  question_choices = [
    ('classification', 'Classification'),
    ('regression', 'Regression'),
  ]

  name = models.CharField('Data Name', max_length=50)
  owner = models.ForeignKey(User, on_delete=models.CASCADE)
  rawdata = models.FileField(upload_to='rawdata/', validators=[file_size, validate_file_extension])
  attribute_data = models.FileField(upload_to='attribute/', validators=[file_size, validate_file_extension])
  datatype = models.CharField(max_length=10, choices=datatype_choices, default='normal')
  questiontype = models.CharField(max_length=15, choices=question_choices, default='classification')
  freq_config = models.JSONField(null=True, blank=True)

  def __str__(self):
      return self.name

  def save(self, *args, **kwargs):
    try:
        this = Data.objects.get(id=self.id)
        if this.rawdata != self.rawdata:
            this.rawdata.delete(save=False)
        if this.attribute_data != self.attribute_data:
            this.attribute_data.delete(save=False)
    except: pass
    super(Data, self).save(*args, **kwargs)

  def delete(self, *args, **kwargs):
      self.rawdata.delete()
      self.attribute_data.delete()
      super().delete(*args, **kwargs)

class HighFrequencyData(models.Model):
  name = models.CharField('Data Name', max_length=50)
  owner = models.ForeignKey(User, on_delete=models.CASCADE)
  rawdata = models.FileField(upload_to='HFD/rawdata/', validators=[validate_file_extension])
  config = models.FileField(upload_to='HFD/config/')
  transData = models.ManyToManyField(Data, blank=True)

  def __str__(self):
      return self.name

class Indicator(models.Model):
  name = models.OneToOneField(Data, on_delete=models.CASCADE, primary_key=True)
  completeness = models.IntegerField(default=0)
  accuracy = models.JSONField(null=True, blank=True)
  info_content = models.JSONField(null=True, blank=True)

  def __str__(self):
      return str(self.name)

class SecurityIndicator(models.Model):
  
  choices = [
    (1, 'YES'),
    (-1, 'NO'),
    (0, 'Unknown'),
    (2, 'Non Applicable'),
  ]
  data_name = models.OneToOneField(Data, on_delete=models.CASCADE, primary_key=True)
  question1 =  models.IntegerField(
    choices=choices,
  )
  question2 =  models.IntegerField(
    choices=choices,
  )
  question3 =  models.IntegerField(
    choices=choices,
  )
  question4 =  models.IntegerField(
    choices=choices,
  )
  question5 =  models.IntegerField(
    choices=choices,
  ) 
  question6 =  models.IntegerField(
    choices=choices,
  )
  question7 =  models.IntegerField(
    choices=choices,
  )
  question8 =  models.IntegerField(
    choices=choices,
  )
  question9 =  models.IntegerField(
    choices=choices,
  )
  question10 =  models.IntegerField(
    choices=choices,
  )
  question11 =  models.IntegerField(
    choices=choices,
  )
  question12 =  models.IntegerField(
    choices=choices,
  )
  question13 =  models.IntegerField(
    choices=choices,
  )
  question14 =  models.IntegerField(
    choices=choices,
  )
  question15 =  models.IntegerField(
    choices=choices,
  )
  question16 =  models.IntegerField(
    choices=choices,
  )
  question17 =  models.IntegerField(
    choices=choices,
  )
  question18 =  models.IntegerField(
    choices=choices,
  )
  question19 =  models.IntegerField(
    choices=choices,
  )
  question20 =  models.IntegerField(
    choices=choices,
  )
  question21 =  models.IntegerField(
    choices=choices,
  )
  question22 =  models.IntegerField(
    choices=choices,
  )
  question23 =  models.IntegerField(
    choices=choices,
  )
  def __str__(self):
      return str(self.data_name)

class TimelinessIndicator(models.Model):
  choices = [
    (1, 'YES'),
    (-1, 'NO'),
    (0, 'Unknown'),
    (2, 'Non Applicable'),
  ]
  data_name = models.OneToOneField(Data, on_delete=models.CASCADE, primary_key=True)
  question1 =  models.IntegerField(
    choices=choices,
  )
  question2 =  models.IntegerField(
    choices=choices,
  )
  

  def __str__(self):
      return str(self.data_name)