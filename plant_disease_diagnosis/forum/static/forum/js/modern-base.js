// Modern Base JavaScript for GreenAI Forum

class GreenAIForum {
    constructor() {
        this.init();
    }

    init() {
        this.initTheme();
        this.initToasts();
        this.initNavbar();
        this.initLoadingStates();
        this.initAnimations();
        this.initFormValidation();
        this.initLazyLoading();
    }

    // Theme Management
    initTheme() {
        const themeToggle = document.getElementById('themeToggle');
        const currentTheme = localStorage.getItem('theme') || 'light';
        
        // Set initial theme
        document.documentElement.setAttribute('data-theme', currentTheme);
        this.updateThemeIcon(currentTheme);

        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                const currentTheme = document.documentElement.getAttribute('data-theme');
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                
                document.documentElement.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
                this.updateThemeIcon(newTheme);
                
                // Add smooth transition effect
                document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
                setTimeout(() => {
                    document.body.style.transition = '';
                }, 300);
            });
        }
    }

    updateThemeIcon(theme) {
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            const icon = themeToggle.querySelector('i');
            if (icon) {
                icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
            }
        }
    }

    // Toast Messages
    initToasts() {
        const toastElements = document.querySelectorAll('.toast');
        toastElements.forEach(toastElement => {
            const toast = new bootstrap.Toast(toastElement);
            toast.show();
        });
    }

    // Navbar Enhancements
    initNavbar() {
        const navbar = document.querySelector('.modern-navbar');
        let lastScrollTop = 0;

        // Navbar scroll effect
        window.addEventListener('scroll', () => {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            
            if (scrollTop > 100) {
                navbar.classList.add('navbar-scrolled');
            } else {
                navbar.classList.remove('navbar-scrolled');
            }

            // Hide/show navbar on scroll
            if (scrollTop > lastScrollTop && scrollTop > 200) {
                navbar.style.transform = 'translateY(-100%)';
            } else {
                navbar.style.transform = 'translateY(0)';
            }
            
            lastScrollTop = scrollTop;
        });

        // Mobile menu close on link click
        const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
        const navbarCollapse = document.querySelector('.navbar-collapse');
        
        navLinks.forEach(link => {
            link.addEventListener('click', () => {
                if (window.innerWidth < 992) {
                    const bsCollapse = new bootstrap.Collapse(navbarCollapse, {
                        toggle: false
                    });
                    bsCollapse.hide();
                }
            });
        });
    }

    // Loading States
    initLoadingStates() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        
        // Show loading for AJAX requests
        document.addEventListener('ajaxStart', () => {
            this.showLoading();
        });
        
        document.addEventListener('ajaxComplete', () => {
            this.hideLoading();
        });

        // Show loading for form submissions
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            form.addEventListener('submit', (e) => {
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn) {
                    this.setButtonLoading(submitBtn, true);
                }
            });
        });
    }

    showLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.classList.remove('d-none');
        }
    }

    hideLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.classList.add('d-none');
        }
    }

    setButtonLoading(button, isLoading) {
        if (isLoading) {
            button.disabled = true;
            const originalText = button.innerHTML;
            button.dataset.originalText = originalText;
            button.innerHTML = `
                <span class="spinner-border spinner-border-sm me-2" role="status"></span>
                Yükleniyor...
            `;
        } else {
            button.disabled = false;
            button.innerHTML = button.dataset.originalText || button.innerHTML;
        }
    }

    // Animations
    initAnimations() {
        // Intersection Observer for fade-in animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-fade-in');
                }
            });
        }, observerOptions);

        // Observe elements for animation
        const animateElements = document.querySelectorAll('.forum-card, .topic-item, .blog-item');
        animateElements.forEach(el => {
            observer.observe(el);
        });

        // Add CSS for fade-in animation
        if (!document.querySelector('#fadeInStyles')) {
            const style = document.createElement('style');
            style.id = 'fadeInStyles';
            style.textContent = `
                .animate-fade-in {
                    animation: fadeInUp 0.6s ease-out forwards;
                }
                
                @keyframes fadeInUp {
                    from {
                        opacity: 0;
                        transform: translateY(30px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
            `;
            document.head.appendChild(style);
        }
    }

    // Form Validation
    initFormValidation() {
        const forms = document.querySelectorAll('form');
        
        forms.forEach(form => {
            form.addEventListener('submit', (e) => {
                if (!form.checkValidity()) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                form.classList.add('was-validated');
            });

            // Real-time validation
            const inputs = form.querySelectorAll('input, textarea, select');
            inputs.forEach(input => {
                input.addEventListener('blur', () => {
                    this.validateField(input);
                });

                input.addEventListener('input', () => {
                    if (input.classList.contains('is-invalid')) {
                        this.validateField(input);
                    }
                });
            });
        });
    }

    validateField(field) {
        const isValid = field.checkValidity();
        field.classList.remove('is-valid', 'is-invalid');
        field.classList.add(isValid ? 'is-valid' : 'is-invalid');

        // Custom validation messages
        const feedback = field.parentElement.querySelector('.invalid-feedback');
        if (!isValid && !feedback) {
            const feedbackDiv = document.createElement('div');
            feedbackDiv.className = 'invalid-feedback';
            feedbackDiv.textContent = field.validationMessage;
            field.parentElement.appendChild(feedbackDiv);
        }
    }

    // Lazy Loading for Images
    initLazyLoading() {
        const images = document.querySelectorAll('img[data-src]');
        
        const imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.classList.remove('lazy');
                    imageObserver.unobserve(img);
                }
            });
        });

        images.forEach(img => {
            imageObserver.observe(img);
        });
    }

    // Utility Methods
    showToast(message, type = 'info') {
        const toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) return;

        const toastId = 'toast-' + Date.now();
        const toastHTML = `
            <div id="${toastId}" class="toast modern-toast toast-${type}" role="alert">
                <div class="toast-header">
                    <div class="toast-icon me-2">
                        <i class="fas fa-${this.getToastIcon(type)}"></i>
                    </div>
                    <strong class="me-auto">${this.getToastTitle(type)}</strong>
                    <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">${message}</div>
            </div>
        `;

        toastContainer.insertAdjacentHTML('beforeend', toastHTML);
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement);
        toast.show();

        // Remove toast element after it's hidden
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    }

    getToastIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || icons.info;
    }

    getToastTitle(type) {
        const titles = {
            success: 'Başarılı',
            error: 'Hata',
            warning: 'Uyarı',
            info: 'Bilgi'
        };
        return titles[type] || titles.info;
    }

    // AJAX Helper
    async fetchData(url, options = {}) {
        this.showLoading();
        
        try {
            const response = await fetch(url, {
                headers: {
                    'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            this.showToast('Bir hata oluştu: ' + error.message, 'error');
            throw error;
        } finally {
            this.hideLoading();
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.greenAI = new GreenAIForum();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GreenAIForum;
}
