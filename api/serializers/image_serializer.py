from rest_framework import serializers
from .models import ImagePrediction

class ImageSerializer(serializers.ModelSerializer):
    image = serializers.ImageField()

    class Meta:
        model = ImagePrediction
        fields = ['image']

    def validate_image(self, value):
        """Validate that the uploaded file is an image."""
        if not value.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise serializers.ValidationError("Only .png, .jpg, and .jpeg images are supported.")
        return value
