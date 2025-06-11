# Generated manually

from django.db import migrations
from django.contrib.auth import get_user_model

def create_missing_profiles(apps, schema_editor):
    """Eksik UserProfile kayıtlarını oluşturur"""
    User = get_user_model()
    UserProfile = apps.get_model('forum', 'UserProfile')
    
    # Tüm kullanıcıları al
    users = User.objects.all()
    created_count = 0
    
    for user in users:
        # UserProfile var mı kontrol et
        if not UserProfile.objects.filter(user=user).exists():
            # UserProfile yoksa oluştur
            UserProfile.objects.create(user=user, bio='')
            created_count += 1
            print(f"UserProfile oluşturuldu: {user.username}")
    
    print(f"Toplam {created_count} UserProfile oluşturuldu")

def reverse_create_missing_profiles(apps, schema_editor):
    """Geri alma işlemi - boş bırakılabilir"""
    pass

class Migration(migrations.Migration):

    dependencies = [
        ('forum', '0008_alter_userprofile_user_alter_userprofile_avatar'),
    ]

    operations = [
        migrations.RunPython(create_missing_profiles, reverse_create_missing_profiles),
    ]
