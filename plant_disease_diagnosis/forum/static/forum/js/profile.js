document.addEventListener('DOMContentLoaded', function() {
    // Tab değiştirme
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Aktif tab'ı değiştir
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // İçeriği göster
            const tabId = button.dataset.tab;
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === tabId) {
                    content.classList.add('active');
                }
            });
        });
    });

    // Modal işlemleri
    const editProfileBtn = document.getElementById('editProfileBtn');
    const modal = document.getElementById('editProfileModal');
    const closeModalButtons = document.querySelectorAll('.close-modal');

    if (editProfileBtn && modal) {
        editProfileBtn.addEventListener('click', () => {
            modal.style.display = 'block';
            document.body.style.overflow = 'hidden';
        });
    }

    closeModalButtons.forEach(button => {
        button.addEventListener('click', () => {
            if (modal) {
                modal.style.display = 'none';
                document.body.style.overflow = 'auto';
            }
        });
    });

    // Modal dışına tıklandığında kapat
    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    });

    // Avatar yükleme önizleme
    const avatarInput = document.querySelector('input[name="avatar"]');
    const avatarPreview = document.getElementById('avatarPreview');

    if (avatarInput && avatarPreview) {
        avatarInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    avatarPreview.src = e.target.result;
                };
                reader.readAsDataURL(file);
            }
        });
    }

    // İstatistik sayılarını animasyonla güncelleme (Opsiyonel)
    const statNumbers = document.querySelectorAll('.stat-number');
    
    function animateNumber(element) {
        const target = parseInt(element.textContent);
        const duration = 2000;
        const step = target / (duration / 16);
        let current = 0;

        const timer = setInterval(() => {
            current += step;
            if (current >= target) {
                element.textContent = target;
                clearInterval(timer);
            } else {
                element.textContent = Math.floor(current);
            }
        }, 16);
    }

    statNumbers.forEach(number => animateNumber(number));

    // Profil güncelleme formu
    const profileForm = document.querySelector('#editProfileModal form');
    if (profileForm) {
        profileForm.addEventListener('submit', async function(e) {
            e.preventDefault();

            const formData = new FormData(this);
            const submitBtn = this.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;

            // Loading state
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Kaydediliyor...';
            submitBtn.disabled = true;

            try {
                const response = await fetch(this.action, {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
                    }
                });

                const data = await response.json();

                if (data.success) {
                    // Başarılı mesaj göster
                    showMessage('Profil başarıyla güncellendi!', 'success');

                    // Modal'ı kapat
                    modal.style.display = 'none';
                    document.body.style.overflow = 'auto';

                    // Sayfayı yenile (profil bilgilerini güncellemek için)
                    setTimeout(() => {
                        window.location.reload();
                    }, 1000);
                } else {
                    showMessage(data.error || 'Bir hata oluştu!', 'error');
                }
            } catch (error) {
                showMessage('Bağlantı hatası!', 'error');
            } finally {
                // Button'ı eski haline getir
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
            }
        });
    }

    // Mesaj gösterme fonksiyonu
    function showMessage(message, type) {
        // Mevcut mesajları temizle
        const existingMessages = document.querySelectorAll('.alert-message');
        existingMessages.forEach(msg => msg.remove());

        const alertDiv = document.createElement('div');
        alertDiv.className = `alert-message alert-${type}`;
        alertDiv.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
            ${message}
        `;

        // Sayfanın üstüne ekle
        document.body.insertBefore(alertDiv, document.body.firstChild);

        // 5 saniye sonra kaldır
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
});