from django import forms
from django.forms import ModelForm
from .models import Data, SecurityIndicator, TimelinessIndicator, HighFrequencyData


class DataForm(ModelForm):
  class Meta:
    model = Data
    fields = ['name', 'owner', 'rawdata', 'attribute_data']
    labels = {
      'name': 'Dataset Name',
      'owner': 'Owner Id',
      'rawdata': 'Raw Data (Upload file must be smaller than 2MB and only accept csv format)', 
      'attribute_data': 'Column data (Upload file must be smaller than 2MB and only accept csv format)', 
      # 'datatype': 'Data Type', 
      # 'questiontype': 'Question Type'
    }
    widgets = {
      'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'E.g. Titanic Classification'}),
      'owner': forms.TextInput(attrs={'class': 'form-control', 'readonly': True}),
      'rawdata': forms.FileInput(attrs={'class': 'form-control'}),
      'attribute_data': forms.FileInput(attrs={'class': 'form-control'}),
      # 'datatype': forms.Select(attrs={'class': 'form-select'}),
      # 'questiontype': forms.Select(attrs={'class': 'form-select'}),
    }

class HighFrequencyDataForm(ModelForm):
  class Meta:
    model = HighFrequencyData
    fields = ['name', 'owner', 'rawdata', 'config']
    labels = {
      'name': 'Dataset Name',
      'owner': 'Owner Id',
      'rawdata': 'Raw Data (Upload file must be smaller than 2MB and only accept csv format)', 
      'config': 'Config (Upload file must be smaller than 2MB and only accept csv format)', 
      
    }
    widgets = {
      'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'E.g. Titanic Classification'}),
      'owner': forms.TextInput(attrs={'class': 'form-control', 'readonly': True}),
      'rawdata': forms.FileInput(attrs={'class': 'form-control'}),
      'config': forms.FileInput(attrs={'class': 'form-control'}),
    }

class SecurityIndicatorForm(ModelForm):
  class Meta:
    model = SecurityIndicator
    fields = '__all__'
    labels = {
      'data_name': 'Data Id',
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
    widgets = {
      'data_name': forms.TextInput(attrs={'class':'form-control','readonly':True}),
      'question1': forms.Select(attrs={'class': 'form-select'}),
      'question2': forms.Select(attrs={'class': 'form-select'}),
      'question3': forms.Select(attrs={'class': 'form-select'}),
      'question4': forms.Select(attrs={'class': 'form-select'}),
      'question5': forms.Select(attrs={'class': 'form-select'}),
      'question6': forms.Select(attrs={'class': 'form-select'}),
      'question7': forms.Select(attrs={'class': 'form-select'}),
      'question8': forms.Select(attrs={'class': 'form-select'}),
      'question9': forms.Select(attrs={'class': 'form-select'}),
      'question10': forms.Select(attrs={'class': 'form-select'}),
      'question11': forms.Select(attrs={'class': 'form-select'}),
      'question12': forms.Select(attrs={'class': 'form-select'}),
      'question13': forms.Select(attrs={'class': 'form-select'}),
      'question14': forms.Select(attrs={'class': 'form-select'}),
      'question15': forms.Select(attrs={'class': 'form-select'}),
      'question16': forms.Select(attrs={'class': 'form-select'}),
      'question17': forms.Select(attrs={'class': 'form-select'}),
      'question18': forms.Select(attrs={'class': 'form-select'}),
      'question19': forms.Select(attrs={'class': 'form-select'}),
      'question20': forms.Select(attrs={'class': 'form-select'}),
      'question21': forms.Select(attrs={'class': 'form-select'}),
      'question22': forms.Select(attrs={'class': 'form-select'}),
      'question23': forms.Select(attrs={'class': 'form-select'}),
      
    }

class TimelinessIndicatorForm(ModelForm):
  class Meta:
    model = TimelinessIndicator
    fields = '__all__'
    labels = {
      'data_name': 'Data Id',
      'question1': 'Whether the data is collected within 24 hours?',
      'question2': 'Whether the data is updated to the database within 24 hours?',
    }
    widgets = {
      'data_name': forms.TextInput(attrs={'class':'form-control','readonly':True}),
      'question1': forms.Select(attrs={'class': 'form-select'}),
      'question2': forms.Select(attrs={'class': 'form-select'}),
    }