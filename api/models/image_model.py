from django.db import models

class ImagePrediction(models.Model):
    image = models.ImageField(upload_to='uploaded_images/')
    predicted_label = models.CharField(max_length=100)
    confidence = models.FloatField()
    predicted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"ImagePrediction: {self.predicted_label} with {self.confidence * 100:.2f}% confidence"
