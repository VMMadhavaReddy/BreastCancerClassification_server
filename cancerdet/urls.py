from django.urls import path, re_path

from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('train', views.train,name="train"),
    path('test', views.test,name="test"),
]