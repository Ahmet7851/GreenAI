#!/usr/bin/env python
"""
Eksik UserProfile kayıtlarını oluşturan script
"""
import os
import sys
import django

# Django ayarlarını yükle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'plant_disease_diagnosis.settings')
django.setup()

from django.contrib.auth.models import User
from forum.models import UserProfile

def create_missing_profiles():
    """Eksik UserProfile kayıtlarını oluşturur"""
    print("Eksik UserProfile kayıtları kontrol ediliyor...")
    
    # Tüm kullanıcıları al
    users = User.objects.all()
    created_count = 0
    updated_count = 0
    
    for user in users:
        try:
            # UserProfile var mı kontrol et
            profile = user.profile
            if profile.bio is None:
                profile.bio = ''
                profile.save()
                updated_count += 1
                print(f"UserProfile güncellendi: {user.username}")
        except UserProfile.DoesNotExist:
            # UserProfile yoksa oluştur
            UserProfile.objects.create(user=user, bio='')
            created_count += 1
            print(f"UserProfile oluşturuldu: {user.username}")
    
    print(f"\nToplam {created_count} UserProfile oluşturuldu")
    print(f"Toplam {updated_count} UserProfile güncellendi")
    print("İşlem tamamlandı!")
    
    # UserProfile'ları listele
    print(f"\nMevcut UserProfile sayısı: {UserProfile.objects.count()}")
    print("UserProfile'lar:")
    for profile in UserProfile.objects.all():
        avatar_status = "Var" if profile.avatar else "Yok"
        print(f"- {profile.user.username} (Avatar: {avatar_status})")

if __name__ == '__main__':
    create_missing_profiles()
