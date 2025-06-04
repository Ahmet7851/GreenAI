from django.core.management.base import BaseCommand
from django.utils.text import slugify
from forum.models import Category


class Command(BaseCommand):
    help = 'Kategorileri düzeltir ve slug alanlarını günceller'

    def handle(self, *args, **options):
        self.stdout.write('Kategoriler düzeltiliyor...')

        # Mevcut kategorileri kontrol et
        categories = Category.objects.all()
        
        if not categories.exists():
            # Kategoriler yoksa oluştur
            categories_data = [
                {'name': 'Bitki Hastalıkları', 'icon': 'fas fa-bug', 'description': 'Bitki hastalıkları hakkında tartışmalar'},
                {'name': 'Tarım Teknikleri', 'icon': 'fas fa-seedling', 'description': 'Modern tarım teknikleri'},
                {'name': 'Organik Tarım', 'icon': 'fas fa-leaf', 'description': 'Organik tarım yöntemleri'},
                {'name': 'Genel Tartışma', 'icon': 'fas fa-comments', 'description': 'Genel konular'}
            ]
            
            for cat_data in categories_data:
                slug = slugify(cat_data['name'])
                category, created = Category.objects.get_or_create(
                    name=cat_data['name'],
                    defaults={
                        'slug': slug,
                        'icon': cat_data['icon'],
                        'description': cat_data['description']
                    }
                )
                if created:
                    self.stdout.write(f'Kategori oluşturuldu: {category.name} -> {category.slug}')
        else:
            # Mevcut kategorilerin slug'larını güncelle
            for category in categories:
                if not category.slug:
                    category.slug = slugify(category.name)
                    category.save()
                    self.stdout.write(f'Slug güncellendi: {category.name} -> {category.slug}')
                else:
                    self.stdout.write(f'Slug zaten mevcut: {category.name} -> {category.slug}')

        self.stdout.write(self.style.SUCCESS('Kategoriler başarıyla düzeltildi!'))
