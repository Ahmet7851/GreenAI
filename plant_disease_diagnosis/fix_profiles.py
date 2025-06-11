#!/usr/bin/env python
"""
UserProfile sorunlarını düzelten script
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

def fix_profiles():
    """UserProfile sorunlarını düzeltir"""
    print("UserProfile sorunları düzeltiliyor...")
    
    # Tüm kullanıcıları al
    users = User.objects.all()
    created_count = 0
    updated_count = 0
    
    for user in users:
        try:
            # Önce profile ilişkisini kontrol et
            if hasattr(user, 'profile'):
                try:
                    profile = user.profile
                    if profile.bio is None:
                        profile.bio = ''
                        profile.save()
                        updated_count += 1
                        print(f"UserProfile güncellendi: {user.username}")
                except UserProfile.DoesNotExist:
                    # Profile yoksa oluştur
                    UserProfile.objects.create(user=user, bio='')
                    created_count += 1
                    print(f"UserProfile oluşturuldu (profile): {user.username}")
            else:
                # Profile ilişkisi yoksa oluştur
                UserProfile.objects.create(user=user, bio='')
                created_count += 1
                print(f"UserProfile oluşturuldu (yeni): {user.username}")
                
        except Exception as e:
            print(f"Hata {user.username} için: {e}")
            # Hata durumunda da oluşturmaya çalış
            try:
                UserProfile.objects.get_or_create(user=user, defaults={'bio': ''})
                created_count += 1
                print(f"UserProfile oluşturuldu (hata sonrası): {user.username}")
            except Exception as e2:
                print(f"Kritik hata {user.username} için: {e2}")
    
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
    fix_profiles()
