#!/usr/bin/env python
import os
import sys
import django

# Django ayarlarını yükle
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'plant_disease_diagnosis.settings')
django.setup()

from forum.models import UserProfile
from django.contrib.auth.models import User

def fix_userprofiles():
    print("UserProfile kayıtları düzeltiliyor...")
    
    # Tüm kullanıcıları al
    users = User.objects.all()
    
    for user in users:
        try:
            # UserProfile var mı kontrol et
            profile = user.userprofile
            if profile.bio is None:
                profile.bio = ''
                profile.save()
                print(f"UserProfile düzeltildi: {user.username}")
        except UserProfile.DoesNotExist:
            # UserProfile yoksa oluştur
            UserProfile.objects.create(user=user, bio='')
            print(f"UserProfile oluşturuldu: {user.username}")
    
    print("UserProfile kayıtları başarıyla düzeltildi!")
    
    # UserProfile'ları listele
    print("\nMevcut UserProfile'lar:")
    for profile in UserProfile.objects.all():
        print(f"- {profile.user.username} (bio: '{profile.bio}')")

if __name__ == '__main__':
    fix_userprofiles()
