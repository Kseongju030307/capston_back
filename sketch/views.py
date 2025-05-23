from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
import os

from .generate_caption import image_crop, image_classification #, generate_caption
from django.conf import settings

class SketchAnalyzeView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        try:
            image_file = request.FILES.get('image')
            if not image_file:
                return Response({"error": "이미지를 업로드 해주세요."}, status=400)

            image_save_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_image.png')
            os.makedirs(os.path.dirname(image_save_path), exist_ok=True)

            with open(image_save_path, 'wb+') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)
            cropped_images = image_crop(image_save_path)
            categories = image_classification(cropped_images)

            return Response({
                "categories": [{"name": c} for c in categories]
            })

        except Exception as e:
            # 에러 메시지를 클라이언트에도 보내도록 함
            return Response({"error": str(e)}, status=500)
"""
            # 핵심 파이프라인
            cropped_images = image_crop(image_save_path)
            categories = image_classification(cropped_images)
            captions = generate_caption(categories, image_save_path)

            return Response({
                "categories": categories,
                "captions": captions
            })
   """ 