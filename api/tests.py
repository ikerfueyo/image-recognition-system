from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from PIL import Image
import io

class ImageClassificationAPITest(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.url = reverse('image-classification')

    def generate_image_file(self, format='JPEG'):
        """Generate a simple in-memory image for testing."""
        img = Image.new('RGB', (100, 100), color='red')
        img_file = io.BytesIO()
        img.save(img_file, format=format)
        img_file.seek(0)
        return img_file

    def test_image_classification_success(self):
        """Test successful image classification."""
        img_file = self.generate_image_file()
        response = self.client.post(self.url, {'image': img_file}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('predicted_label', response.data)
        self.assertIn('confidence', response.data)

    def test_image_classification_invalid_file(self):
        """Test invalid file format rejection."""
        img_file = io.BytesIO(b"Not an image")
        response = self.client.post(self.url, {'image': img_file}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
