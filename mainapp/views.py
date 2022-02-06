from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect, FileResponse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile

from .forms import DataForm, SecurityIndicatorForm, TimelinessIndicatorForm, HighFrequencyDataForm
from .models import Data, SecurityIndicator, TimelinessIndicator, Indicator, HighFrequencyData
from .utils import Accuracy, InfoContent, Completeness, Overview

import pandas as pd
import os

from plotly.offline import plot
import plotly.graph_objs as go

from .freq_trans import Feature_all
import json
from datetime import datetime

# Create your views here.
@login_required(login_url='/members/login_user')
def upload_view(request):
  user = request.user

  submitted = False
  if request.method == "POST":
    form = DataForm(request.POST, request.FILES)
    if form.is_valid():
      form.save()
      return HttpResponseRedirect('upload?submitted=True')
  else: 
    form = DataForm(initial={"owner": user})
    if 'submitted' in request.GET:
      submitted = True

  context = {
    'form': form,
    'submitted': submitted
  }
  return render(request, 'mainapp/upload_view.html', context)

@login_required(login_url='/members/login_user')
def upload_hfd(request):
  user = request.user

  submitted = False
  if request.method == "POST":
    form = HighFrequencyDataForm(request.POST, request.FILES)
    if form.is_valid():
      form.save()
      return HttpResponseRedirect('upload_hfd?submitted=True')
  else: 
    form = HighFrequencyDataForm(initial={"owner": user})
    if 'submitted' in request.GET:
      submitted = True

  context = {
    'form': form,
    'submitted': submitted
  }
  return render(request, 'mainapp/upload_hfd.html', context)

@login_required(login_url='/members/login_user')
def update_data(request, data_id):
  data = get_object_or_404(Data, pk=data_id, owner=request.user)

  form = DataForm(request.POST or None, instance=data)

  if request.method == "POST":
    old_rawdata = data.rawdata
    old_attrdata = data.attribute_data

    if 'rawdata' in request.FILES.keys():
      data.rawdata = request.FILES['rawdata']
    if 'attribute_data' in request.FILES.keys():
      data.attribute_data = request.FILES['attribute_data']

    form = DataForm(request.POST, instance=data)
    if form.is_valid():
      if(os.path.exists(old_rawdata.path)):
        os.remove(old_rawdata.path)

      if(os.path.exists(old_attrdata.path)):
        os.remove(old_attrdata.path)
      
      form.save()
      return redirect('show-data', data_id)
    else:
      print('form is invalid')
  
  context = {
    'data': data,
    'data_id': data_id,
    'form': form
  }

  return render(request, 'mainapp/update_data.html', context)

@login_required(login_url='/members/login_user')
def update_hfd(request, hfd_id):
  data = get_object_or_404(HighFrequencyData, pk=hfd_id, owner=request.user)

  form = HighFrequencyDataForm(request.POST or None, instance=data)

  if request.method == "POST":
    old_rawdata = data.rawdata
    old_config = data.config

    if 'rawdata' in request.FILES.keys():
      data.rawdata = request.FILES['rawdata']
    if 'config' in request.FILES.keys():
      data.config = request.FILES['config']

    form = HighFrequencyDataForm(request.POST, instance=data)
    if form.is_valid():
      if(os.path.exists(old_rawdata.path)):
        os.remove(old_rawdata.path)

      if(os.path.exists(old_config.path)):
        os.remove(old_config.path)
      
      form.save()
      return redirect('show-hfd', hfd_id)
    else:
      print('form is invalid')
  
  context = {
    'data': data,
    'hfd_id': hfd_id,
    'form': form
  }

  return render(request, 'mainapp/update_hfd.html', context)

@login_required(login_url='/members/login_user')
def delete_data(request, data_id):
  data = get_object_or_404(Data, pk=data_id, owner=request.user)

  if request.method == "POST":
    data.delete()
    return redirect('list-data')

  context = {
    'data': data
  }
  return render(request, 'mainapp/delete_data.html', context)

@login_required(login_url='/members/login_user')
def delete_hfd(request, hfd_id):
  data = get_object_or_404(HighFrequencyData, pk=hfd_id, owner=request.user)

  if request.method == "POST":
    data.delete()
    return redirect('list-data')

  context = {
    'data': data
  }
  return render(request, 'mainapp/delete_hfd.html', context)

@login_required(login_url='/members/login_user')  
def fillout_security(request, data_id):
  
  data = get_object_or_404(Data, pk=data_id, owner=request.user)
  
  # update security indicator
  try:
    security_ind = SecurityIndicator.objects.get(data_name=data)
    form = SecurityIndicatorForm(request.POST or None, instance=security_ind)
    action = 'Update'
  # create new security indicator
  except:
    form = SecurityIndicatorForm(request.POST or None, initial={"data_name": data})
    action = 'Fill Out'
    
  if form.is_valid():
    form.save()
    return redirect('show-data', data_id)
  
  context = {
    'form': form,
    'action': action,
    'data_id': data_id,
    'data': data
  }
  return render(request, 'mainapp/security_view.html', context)

@login_required(login_url='/members/login_user')
def fillout_timeliness(request, data_id):

  data = get_object_or_404(Data, pk=data_id, owner=request.user)
  
  # update timeliness indicator
  try:
    timeliness_ind = TimelinessIndicator.objects.get(data_name=data)
    form = TimelinessIndicatorForm(request.POST or None, instance=timeliness_ind)
    action = 'Update'
  # create new timeliness indicator
  except:
    form = TimelinessIndicatorForm(request.POST or None, initial={"data_name": data})
    action = 'Fill Out'
    
  if form.is_valid():
    form.save()
    return redirect('show-data', data_id)
  
  context = {
    'form': form,
    'action': action,
    'data_id': data_id,
    'data': data
  }
  return render(request, 'mainapp/timeliness_view.html', context)

@login_required(login_url='/members/login_user')
def list_data(request):
  user = request.user
  data_list = Data.objects.filter(owner=user)
  hfd_list = HighFrequencyData.objects.filter(owner=user)
  
  context = {
    'data_list': data_list,
    'hfd_list': hfd_list,
  }

  return render(request, 'mainapp/data_list.html', context)

@login_required(login_url='/members/login_user')
def show_indicator(request, data_id):

  data = get_object_or_404(Data, pk=data_id, owner=request.user)
  
  # read data and attribute data
  raw_df = pd.read_csv(data.rawdata)
  column_df = pd.read_csv(data.attribute_data)

  # check raw data and column type format
  try:
    correctFormat = check_data_format(raw_df, column_df)
  except:
    correctFormat = False

  if correctFormat:
    # get security indicator score by data
    security_dict = get_security(data)
    
    # get timeliness indicator score by data
    timeliness_dict = get_timeliness(data)
    # change column attribute to json format
    column_dict = {}
    for _, row in column_df.iterrows():
      column_dict[row['Column']] = row['Attribute']

    # delete other type column
    raw_df, column_dict = handle_other_type(raw_df, column_dict)

    # get data summary
    result = get_overview_result(raw_df, column_dict) 
    overview = result.overview_data()
    summary_dict = result.get_summary()
    
    try:
      indicator = Indicator.objects.get(name=data)
      isNewInd = False
    except: 
      isNewInd = True

    if(isNewInd):
      # removing missing value
      raw_df = remove_missing_value(raw_df)
        
      # get accuracy indicator
      acc_dict = get_accuracy(raw_df, column_dict)

      # get infomation content dict
      infoContent = get_infoContent(raw_df, column_dict)

      # get completeness indicator
      comp_score = get_completeness_score(raw_df, column_dict)
      comp_score = int(comp_score*100)
      
      indicator = Indicator.objects.create(
                    name=data, 
                    completeness=comp_score,
                    accuracy=acc_dict, 
                    info_content=infoContent
                  )
      indicator.save()            
    else:
      # get accuracy indicator
      acc_dict = indicator.accuracy

      # get completeness indicator
      comp_score = indicator.completeness

      # get info content dict
      infoContent = indicator.info_content

    # indicator scores
    indicator_scores = [acc_dict['score'] , comp_score, int(infoContent['total_score']), security_dict['score'], timeliness_dict['score']]

    context = {
      'data': data,
      'data_id': data_id,
      'acc_res': acc_dict['acc_res'],
      'opposite_var_warning': acc_dict['opposite_var_warning'],
      'acc_detail': acc_dict['detail'],
      'sec_detail': security_dict['detail'],  
      'sec_checked': security_dict['checked_questions'],
      'sec_improved': security_dict['improved_questions'],
      'sec_finished': security_dict['finished'],
      'time_detail': timeliness_dict['detail'],
      'time_finished': timeliness_dict['finished'],
      'time_checked': timeliness_dict['checked_questions'],
      'time_improved': timeliness_dict['improved_questions'],
      'comp_score': comp_score,
      'gainRatio': infoContent['gain_ratio'],
      'var_score': infoContent['var_score'],
      'indicator_scores': indicator_scores,
      'summary': summary_dict,
      'correctFormat': correctFormat
    }

  else:
    indicator_scores = [0, 0, 0, 0, 0]
    context = {
      'data': data,
      'data_id': data_id,
      'indicator_scores': indicator_scores,
      'correctFormat': correctFormat
    }

  return render(request, 'mainapp/show_indicator.html', context)

@login_required(login_url='/members/login_user')
def recalculate_indicator(request, data_id):
  data = get_object_or_404(Data, pk=data_id, owner=request.user)

  indicator = Indicator.objects.get(name=data)

  # read data and attribute data
  raw_df = pd.read_csv(data.rawdata)
  column_df = pd.read_csv(data.attribute_data)

  # change column attribute to json format
  column_dict = {}
  for _, row in column_df.iterrows():
    column_dict[row['Column']] = row['Attribute']

  # delete other type column
  raw_df, column_dict = handle_other_type(raw_df, column_dict)

  # removing missing value
  raw_df = remove_missing_value(raw_df)
      
  # get accuracy indicator
  acc_dict = get_accuracy(raw_df, column_dict)

  # get infomation content dict
  infoContent = get_infoContent(raw_df, column_dict)
  
  # get completeness indicator
  comp_score = get_completeness_score(raw_df, column_dict)
  comp_score = int(comp_score*100)

  indicator.completeness = comp_score
  indicator.accuracy = acc_dict
  indicator.info_content = infoContent
  
  indicator.save()
  messages.success(request, ("Successfully recalculate indicator scores!"))
  return redirect('show-indicator', data_id)

@login_required(login_url='/members/login_user')
def transform_hfd(request, hfd_id):
  hfd = get_object_or_404(HighFrequencyData, pk=hfd_id, owner=request.user)

  # Load raw data and config
  df = pd.read_csv(hfd.rawdata)
  config = json.loads(hfd.config.read())
  
  # Data Transformation
  f_all = Feature_all(df, config['windowSize'], config['target_xs'], config['target_y'], config['samplingFrequency'])
  rawdata_file = ContentFile(f_all.to_csv(index=False))
  current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
  rawdata_file.name = f"{hfd.name}_Trans_{current_time}.csv"

  # Create attribute data
  cols = ['y', 'problem']
  col_attrs = [ config['target_y'][0], config['problem']]
  cols.extend(f_all.columns)
  col_attrs.extend([ 'I_ordinal' for _ in range(len(f_all.columns))])
  data = {
      'Column': cols,
      'Attribute': col_attrs
  }
  attribute_df = pd.DataFrame(data)
  attribute_file = ContentFile(attribute_df.to_csv(index=False))
  attribute_file.name = f"{hfd.name}_Trans_{current_time}_column.csv"

  # config
  freq_config = {
    "samplingFrequency": config["samplingFrequency"],
    "windowSize": config["windowSize"],
    "usedFilter": config["usedFilter"]
  }


  # Save transformated data and column attribute
  data = Data.objects.create(
    name = f"{hfd.name}_Trans_{current_time}",
    owner = request.user,
    rawdata = rawdata_file,
    attribute_data = attribute_file,
    datatype = 'freq',
    questiontype=config['problem'],
    freq_config=freq_config
  )
  data.save()

  hfd.transData.add(data)
  hfd.save()

  return redirect('show-hfd', hfd_id)

@login_required(login_url='/members/login_user')
def show_data(request, data_id):
  
  data = get_object_or_404(Data, pk=data_id, owner=request.user)
  # read data and attribute data
  raw_df = pd.read_csv(data.rawdata)
  column_df = pd.read_csv(data.attribute_data)

  try:
    correctFormat = check_data_format(raw_df, column_df)
  except:
    correctFormat = False
  
  if correctFormat:
    # change attribute data to json format
    column_dict = {}
    for _, row in column_df.iterrows():
      column_dict[row['Column']] = row['Attribute']

    result = get_overview_result(raw_df, column_dict) 
    overview = result.overview_data()

    try:
      cate_variables = result.overview_variable_cat()
    except :
      cate_variables = {}

    try:
      num_variables = result.overview_variable_num()
    except:
      num_variables = {}

    context = {
      'data': data,
      'data_id': data_id,
      'overview': overview,
      'cate_variables': cate_variables,
      'num_variables': num_variables,
      'correctFormat': correctFormat
    }  
  else:
    context = {
      'data': data,
      'data_id': data_id,
      'correctFormat': correctFormat
    }

  return render(request, 'mainapp/show_data.html', context)

@login_required(login_url='/members/login_user')
def show_hfd(request, hfd_id):
  data = get_object_or_404(HighFrequencyData, pk=hfd_id, owner=request.user)
  transdata_list = data.transData.all()

  context = {
    'data': data,
    'hfd_id': hfd_id,
    'transdata_list': transdata_list
  }
  return render(request, 'mainapp/show_hfd.html', context)

def download_rawdata(request):
  filename = 'media/example/titanic.csv'
  response = FileResponse(open(filename, 'rb'))
  return response

def download_attrdata(request):
  filename = 'media/example/titanic_column.csv'
  response = FileResponse(open(filename, 'rb'))
  return response

def check_data_format(raw_df, column_df):
  correct_dict = {
    'y': '',
    'problem': ''
  }

  for col in raw_df.columns:
    correct_dict[col] = ''

  column_dict = {}
  for _, row in column_df.iterrows():
    column_dict[row['Column']] = row['Attribute']

  for key in correct_dict.keys():
    if key not in column_dict:
      return False
  return True

def handle_other_type(raw_df, column_dict):
  drop_cols = []
  for key, value in column_dict.items():
    if value == 'other':
      drop_cols.append(key)
      
  new_df = raw_df.drop(columns=drop_cols)

  for col in drop_cols:
    del column_dict[col]

  return new_df, column_dict

def remove_missing_value(data):
    drop_cols = []
    for col in data.columns:
        if data[col].isnull().sum() >= (0.2*len(data)):
            drop_cols.append(col)

    data = data.drop(columns=drop_cols)

    data = data.dropna()

    return data

def get_overview_result(data, column_dict):
    result = Overview(data, column_dict)
    result.missingvalue_pickup()
    result.column_parse()
    return result

def get_accuracy(data, column_dict):
    res = {}

    aa = Accuracy(data, column_dict)
    res = aa.coef()

    total = 0
    same_dir_cnt = 0
    opposite_var = {}
    
    for key, value in res['acc_res'].items():
      total += 1
      if value == 1:
        same_dir_cnt += 1
      else:
        opposite_var[key] = res['acc_beta'][key]

    # calcualte accuray score
    score = int(100*(same_dir_cnt / total))

    # sort by value to descending order
    opposite_var_warning = dict(sorted(opposite_var.items(), key=lambda item: item[1], reverse=True))
    # var_beta = [ f'{key}({value:.2f})' for key, value in opposite_var.items() ]
    # opposite_var_warning = '>'.join(var_beta)
    
    res['score'] = score
    res['detail'] = f'{same_dir_cnt}/{total}'
    res['opposite_var_warning'] = opposite_var_warning

    return res

def get_infoContent(data, column_dict):
    result = {}
    
    infoContent = InfoContent(data, column_dict)
  
    if infoContent.problem == 'regression':
      result['var_score'] = infoContent.get_variance_score()
      result['gain_ratio'] = {}
      result['total_score'] = 0

    elif infoContent.problem == 'classification':
      result['gain_ratio'] = infoContent.get_gain_ratio()
      # infomation content score
      total_score = 0
      if(len(result['gain_ratio']) > 0):
        for v in result['gain_ratio'].values():
          total_score += v / len(result['gain_ratio'])

      result['total_score'] =  total_score
      result['var_score'] = None
    
    return result

def get_security(data):
    
    improved_questions = []
    checked_questions = []
    questions = {
      'question1': '1. Does the system support user identification?',
      'question2': '2. Does the application force “new” users to change their password upon first login into the application?',
      'question3': '3. Can the system administrator enforce password policy and/or complexity such as minimum length, numbers and alphabet requirements, and upper and lower case constraint, etc.?',
      'question4': '4. Can the application force password expiration and prevent users from reusing a password?',
      'question5': '5. Can the application be set to automatically lock a user’s account after a predetermined number of consecutive unsuccessful logon attempts?',
      'question6': '6. Does the application prohibit users from logging into the application on more than one workstation at the same time with the same user ID?',
      'question7': '7. Can the application be set to automatically log a user off the application after a predefined period of inactivity?',
      'question8': '8. Can access be defined based upon the user’s job role? (Role-based Access Controls (RBAC))? ',
      'question9': '9. Capturing user access activity such as successful logon, logoff, and unsuccessful logon attempts?',
      'question10': '10. Capturing data access inquiry activity such as screens viewed and reports printed?',
      'question11': '11. Capturing data entries, changes, and deletions? ',
      'question12': '12. Does the application allow a system administrator to set the inclusion or exclusion of audited events based on organizational policy and operating requirements or limits?',
      'question13': '13. Is functionality built into the application which allows remote user access and/or control?',
      'question14': '14. Is the application compatible with commercial off the shelf (COTS) virus scanning software products for removal and prevention from malicious code?',
      'question15': '15. Are updates to application software and/or the operating system controlled by a mutual agreement between the support vendor and the application owner?',
      'question16': '16. Do you provide documentation for guidance on establishing and managing security controls such as user access and auditing?',
      'question17': '17. Does the application maintain a journal of transactions or snapshots of data between backup intervals?',
      'question18': '18. Has the application security controls been tested by a third party?',
      'question19': '19. Does the application have ability to run a backup concurrently with the operation of the application?',
      'question20': '20. Does the application include documentation that explains error or messages to users and system administrators and information on what actions required?',
      'question21': '21. Is the relation schema in the first normal form?(Disallow composite attributes, multivalued attributes, and nested relations)',
      'question22': '22. Is the relation schema in the second normal form? (if every non-prime attribute is fully functionally dependent on the primary key)',
      'question23': '23. Is the relation schema in the third normal form? (if no non-prime attribute in the schema is transitively dependent on the primary key)',
    }
    # get security indicator by data
    try:
      security_score = 0
      sec_ind = SecurityIndicator.objects.get(data_name=data)

      # get all fields in security form
      fields = SecurityIndicator._meta.get_fields()

      # calculate the sum of all fields
      total = 0
      for field in fields:
        if field.name.startswith('question'):
          # value: selected field value
          value = getattr(sec_ind, field.name)
          if value != 2:
            security_score += value
            total += 1
          if value == -1:
            improved_questions.append(questions[field.name])
          if value == 0:
            checked_questions.append(questions[field.name])

      score = int(100*(security_score/total))
      
      res = {
        'finished': True,
        'detail': f"{security_score} / {total}",
        'score': score,
        'checked_questions': checked_questions,
        'improved_questions': improved_questions
      }

    except:
      res = {
        'finished': False,
        'detail': 'The score is undefiend,',
        'score': 0,
        'checked_questions': checked_questions,
        'improved_questions': improved_questions
      }

    return res

def get_timeliness(data):
    improved_questions = []
    checked_questions = []
    questions = {
      'question1': 'Whether the data is collected within 24 hours?',
      'question2': 'Whether the data is updated to the database within 24 hours?',
    }
    # get timeliness indicator by data
    try:
      timeliness_score = 0
      time_ind = TimelinessIndicator.objects.get(data_name=data)

      # get all fields in timeliness form
      fields = TimelinessIndicator._meta.get_fields()

      # calculate the sum of all fields
      total = 0
      for field in fields:
        if field.name.startswith('question'):
          # value: selected field value
          value = getattr(time_ind, field.name)
          if value != 2:
            timeliness_score += value
            total += 1
          if value == -1:
            improved_questions.append(questions[field.name])
          if value == 0:
            checked_questions.append(questions[field.name])

      score = int(100*(timeliness_score/total))

      res = {
        'finished': True,
        'detail': f"{timeliness_score} / {total}",
        'score': score,
        'checked_questions': checked_questions,
        'improved_questions': improved_questions
      }
 
    except:
      res = {
        'finished': False,
        'detail': 'The score is undefiend,',
        'score': 0,
        'checked_questions': checked_questions,
        'improved_questions': improved_questions
      }

    return res

def get_completeness_score(data, column_dict):
    comp = Completeness(data, column_dict)

    # try:
    #   mece_score = comp.mece()
    # except:
    #   mece_score = -1
    mece_score = comp.mece()
    
    return mece_score