from django.shortcuts import render

def index(request):
    """Render the main page for uploading and classifying images."""
    return render(request, 'index.html')
