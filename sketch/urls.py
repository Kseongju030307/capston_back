from django.urls import path
from .views import SketchAnalyzeView

urlpatterns = [
    path('analyze/', SketchAnalyzeView.as_view()),
]
