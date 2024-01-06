from django.urls import path
from route import views

urlpatterns = [

    #path('login/', views.login,name = 'login'),
   # path('login', views.LoginView.as_view(), name='login'),
    path('index/', views.index, name='index'),
    path('add/',views.files,name='add'),
    #path('get_d1/', views.get_d1),
]
