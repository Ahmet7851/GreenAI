from django.core.management.base import BaseCommand
from django.utils.text import slugify
from forum.models import Category


class Command(BaseCommand):
    help = 'Kategoriler için slug alanlarını günceller'

    def handle(self, *args, **options):
        self.stdout.write('Kategori slug\'ları güncelleniyor...')

        categories = Category.objects.all()
        updated_count = 0
        
        for category in categories:
            if not category.slug:
                category.slug = slugify(category.name)
                category.save()
                self.stdout.write(f'Slug oluşturuldu: {category.name} -> {category.slug}')
                updated_count += 1
            else:
                self.stdout.write(f'Slug zaten mevcut: {category.name} -> {category.slug}')

        self.stdout.write(
            self.style.SUCCESS(f'Toplam {updated_count} kategori slug\'ı güncellendi!')
        )
