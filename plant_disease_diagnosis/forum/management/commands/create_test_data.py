from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from forum.models import Category, Post, Comment, UserProfile


class Command(BaseCommand):
    help = 'Forum için test verileri oluşturur'

    def handle(self, *args, **options):
        self.stdout.write('Test verileri oluşturuluyor...')

        # Test kullanıcısı oluştur
        user, created = User.objects.get_or_create(
            username='test_user',
            defaults={
                'email': 'test@example.com',
                'first_name': 'Test',
                'last_name': 'User'
            }
        )
        if created:
            user.set_password('testpass123')
            user.save()
            self.stdout.write(f'Kullanıcı oluşturuldu: {user.username}')
        else:
            self.stdout.write(f'Kullanıcı zaten mevcut: {user.username}')

        # UserProfile oluştur
        profile, created = UserProfile.objects.get_or_create(
            user=user,
            defaults={'bio': 'Test kullanıcısı'}
        )

        # Kategoriler oluştur
        categories_data = [
            {'name': 'Bitki Hastalıkları', 'icon': 'fas fa-bug', 'description': 'Bitki hastalıkları hakkında tartışmalar'},
            {'name': 'Tarım Teknikleri', 'icon': 'fas fa-seedling', 'description': 'Modern tarım teknikleri'},
            {'name': 'Organik Tarım', 'icon': 'fas fa-leaf', 'description': 'Organik tarım yöntemleri'},
            {'name': 'Genel Tartışma', 'icon': 'fas fa-comments', 'description': 'Genel konular'}
        ]

        for cat_data in categories_data:
            category, created = Category.objects.get_or_create(
                name=cat_data['name'],
                defaults={
                    'icon': cat_data['icon'],
                    'description': cat_data['description']
                }
            )
            if created:
                self.stdout.write(f'Kategori oluşturuldu: {category.name}')

        # Test postları oluştur
        posts_data = [
            {
                'title': 'Domates Bitkimde Sarı Lekeler Var',
                'content': 'Domates bitkilerimin yapraklarında sarı lekeler belirdi. Bu ne olabilir? Nasıl tedavi edebilirim?',
                'category': 'Bitki Hastalıkları'
            },
            {
                'title': 'Organik Gübre Tavsiyeleri',
                'content': 'Evde organik gübre nasıl hazırlanır? Deneyimlerinizi paylaşabilir misiniz?',
                'category': 'Organik Tarım'
            },
            {
                'title': 'Sulama Sistemi Kurulumu',
                'content': 'Bahçem için damla sulama sistemi kurmayı planlıyorum. Önerileriniz neler?',
                'category': 'Tarım Teknikleri'
            },
            {
                'title': 'Bu Yıl Hangi Sebzeleri Ektiniz?',
                'content': 'Bu sezon hangi sebzeleri ekmeyi planlıyorsunuz? Deneyimlerinizi paylaşalım.',
                'category': 'Genel Tartışma'
            },
            {
                'title': 'Yaprak Biti ile Mücadele',
                'content': 'Bitkilerimde yaprak biti problemi var. Doğal yöntemlerle nasıl mücadele edebilirim?',
                'category': 'Bitki Hastalıkları'
            }
        ]

        for post_data in posts_data:
            category = Category.objects.get(name=post_data['category'])
            post, created = Post.objects.get_or_create(
                title=post_data['title'],
                defaults={
                    'content': post_data['content'],
                    'author': user,
                    'category': category
                }
            )
            if created:
                self.stdout.write(f'Post oluşturuldu: {post.title}')

                # Her post için birkaç yorum oluştur
                comments_data = [
                    'Çok faydalı bir konu, teşekkürler!',
                    'Benzer bir problem yaşamıştım, bu yöntem işe yaradı.',
                    'Daha detaylı bilgi verebilir misiniz?'
                ]

                for i, comment_text in enumerate(comments_data[:2]):  # Her post için 2 yorum
                    Comment.objects.create(
                        post=post,
                        author=user,
                        content=comment_text
                    )

        self.stdout.write(self.style.SUCCESS('Test verileri başarıyla oluşturuldu!'))
