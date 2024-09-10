from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import torch
from torchvision import transforms
from PIL import Image
from .serializers import ImageSerializer
from .models import ImagePrediction

model = torch.load(settings.MODEL_PATH, map_location=torch.device('cpu'))
model.eval()

class ImageClassificationView(APIView):
    """
    View to classify an uploaded image and return the predicted label.
    """
    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)
        
        if serializer.is_valid():
            image_file = serializer.validated_data['image']
            
            image = Image.open(image_file)
            image = self.preprocess_image(image)

            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs.data, 1)
                confidence = torch.softmax(outputs, dim=1)[0][predicted].item()

            predicted_label = self.get_label(predicted.item())

            prediction = ImagePrediction.objects.create(
                image=image_file, predicted_label=predicted_label, confidence=confidence
            )

            return Response({
                'predicted_label': predicted_label,
                'confidence': f"{confidence * 100:.2f}%"
            }, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def preprocess_image(self, image):
        """
        Preprocess the image to the format expected by the model.
        """
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = preprocess(image).unsqueeze(0) # Add batch dimension
        return image

    def get_label(self, index):
        """
        Map the prediction index to a human-readable label.
        """
        labels = {0: "cat", 1: "dog"} # Example label mapping
        return labels.get(index, "Unknown")
