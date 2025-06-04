#!/usr/bin/env python
import os
import sys
import django

# Django ayarlarını yükle
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'plant_disease_diagnosis.settings')
django.setup()

from forum.models import Category
from django.utils.text import slugify

def update_categories():
    print("Kategoriler güncelleniyor...")
    
    # Mevcut kategorileri al
    categories = Category.objects.all()
    
    if not categories.exists():
        print("Kategori bulunamadı, yeni kategoriler oluşturuluyor...")
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
                print(f'Kategori oluşturuldu: {category.name} -> {category.slug}')
    else:
        print("Mevcut kategoriler güncelleniyor...")
        for category in categories:
            old_slug = category.slug
            category.slug = slugify(category.name)
            category.save()
            print(f'Kategori güncellendi: {category.name} -> {old_slug} => {category.slug}')
    
    print("Kategoriler başarıyla güncellendi!")
    
    # Kategorileri listele
    print("\nMevcut kategoriler:")
    for category in Category.objects.all():
        print(f"- {category.name} (slug: {category.slug})")

if __name__ == '__main__':
    update_categories()
