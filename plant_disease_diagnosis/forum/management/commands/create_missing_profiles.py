from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from forum.models import UserProfile

class Command(BaseCommand):
    help = 'Eksik UserProfile kayıtlarını oluşturur'

    def handle(self, *args, **options):
        self.stdout.write("Eksik UserProfile kayıtları kontrol ediliyor...")
        
        # Tüm kullanıcıları al
        users = User.objects.all()
        created_count = 0
        updated_count = 0
        
        for user in users:
            try:
                # UserProfile var mı kontrol et (yeni related_name ile)
                profile = user.profile
                if profile.bio is None:
                    profile.bio = ''
                    profile.save()
                    updated_count += 1
                    self.stdout.write(f"UserProfile güncellendi: {user.username}")
            except UserProfile.DoesNotExist:
                # UserProfile yoksa oluştur
                UserProfile.objects.create(user=user, bio='')
                created_count += 1
                self.stdout.write(f"UserProfile oluşturuldu: {user.username}")
        
        self.stdout.write(f"\nToplam {created_count} UserProfile oluşturuldu")
        self.stdout.write(f"Toplam {updated_count} UserProfile güncellendi")
        self.stdout.write(self.style.SUCCESS("İşlem tamamlandı!"))
        
        # UserProfile'ları listele
        self.stdout.write(f"\nMevcut UserProfile sayısı: {UserProfile.objects.count()}")
        self.stdout.write("UserProfile'lar:")
        for profile in UserProfile.objects.all():
            avatar_status = "Var" if profile.avatar else "Yok"
            self.stdout.write(f"- {profile.user.username} (Avatar: {avatar_status})")
