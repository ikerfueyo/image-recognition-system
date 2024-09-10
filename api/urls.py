from django.urls import path
from .views import ImageClassificationView

urlpatterns = [
    path('classify/', ImageClassificationView.as_view(), name='image-classification'),
]
