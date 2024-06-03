from django.shortcuts import render
from .forms import UploadImageForm
from django.conf import settings
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image as PILImage
import os
from g4f.client import Client


# Modeli yükleyin
model = load_model(os.path.join(settings.BASE_DIR, 'diagnosis', 'assets', 'bitki_hastaligi_model.h5'))

def home(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_path = os.path.join(settings.MEDIA_ROOT, image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            
            prediction = predict_image(image_path)
            return render(request, 'result.html', {'image_path': os.path.join(settings.MEDIA_URL, image.name), 'prediction': prediction})
    else:
        form = UploadImageForm()
    return render(request, 'home.html', {'form': form})

def predict_image(image_path):
    image = PILImage.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    image = image.astype("float32")
    
    tahmin = model.predict(image)
    en_yuksek_olasilik_sinifi = np.argmax(tahmin)
    sinif_etiketleri = ["Bakteri", "Mantar", "Sağlıklı/Diğer", "Zararlı Haşere", "Virüs"]
    tahmin_etiket = sinif_etiketleri[en_yuksek_olasilik_sinifi]

    ##################################################
    hastalik = tahmin_etiket

    chat="Bitkimde ki {} sorununu çözemek için izleyeceğim adımlar nelerdir.".format(hastalik)
    client = Client()
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": chat}],
    #############################################
   
)
   

    

    
    return f"Tahmin edilen hastalık: {tahmin_etiket} İzleyebileceğiniz Çözüm Adımları: { response.choices[0].message.content} " 

    