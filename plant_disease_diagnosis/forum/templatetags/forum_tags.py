from django import template
from django.templatetags.static import static
from forum.models import UserProfile

register = template.Library()

@register.simple_tag
def user_avatar_url(user):
    """
    Kullanıcının avatar URL'sini güvenli bir şekilde döndürür.
    Eğer avatar yoksa default avatar döndürür.
    """
    try:
        # Basit yaklaşım - sadece default avatar döndür şimdilik
        return static('forum/images/default-avatar.png')
    except Exception:
        return static('forum/images/default-avatar.png')

@register.inclusion_tag('forum/partials/user_avatar.html')
def user_avatar(user, size='40', css_class='user-avatar'):
    """
    Kullanıcı avatar'ını render eden inclusion tag
    """
    avatar_url = user_avatar_url(user)
    return {
        'avatar_url': avatar_url,
        'username': user.username,
        'size': size,
        'css_class': css_class
    }
