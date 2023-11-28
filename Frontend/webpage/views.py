from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.core.exceptions import ObjectDoesNotExist

# from .models import TutoringAdmin,StudentInfo,Tutoring,WeeklyReport
from rest_framework import generics
# from .serializers import TutoringAdminSerializer,StudentInfoSerializer,TutoringSerializer,WeeklyReportSerializer
from django.utils.decorators import method_decorator
from rest_framework.parsers import JSONParser
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.views import View
from rest_framework.decorators import api_view, permission_classes
from rest_framework import permissions
from django.http import HttpResponseRedirect
from django.urls import reverse_lazy
from django.views.generic import TemplateView
from django.views.generic import DetailView
# from .form import TutorForm
from django.http import FileResponse

import numpy as np
import json_numpy
# Create your views here.

class Main(APIView):
    def get(self, request):
        return render(request, "index.html")

@api_view(["GET"])
@permission_classes((permissions.AllowAny,))
def get_data(request, stock_name):
    response = ""
    if stock_name == "samsung":
        response = "abc"

    return response