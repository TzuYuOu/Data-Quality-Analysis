from django.contrib import admin
from .models import Data, SecurityIndicator, TimelinessIndicator, Indicator, HighFrequencyData
# Register your models here.

admin.site.register(Data)
admin.site.register(SecurityIndicator)
admin.site.register(TimelinessIndicator)
admin.site.register(Indicator)
admin.site.register(HighFrequencyData)