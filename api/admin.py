from django.contrib import admin
from .models import ImagePrediction

@admin.register(ImagePrediction)
class ImagePredictionAdmin(admin.ModelAdmin):
    list_display = ('predicted_label', 'confidence', 'predicted_at')
    search_fields = ('predicted_label',)
    readonly_fields = ('predicted_label', 'confidence', 'predicted_at')

    def has_add_permission(self, request):
        # Prevent adding new entries directly from the admin
        return False
