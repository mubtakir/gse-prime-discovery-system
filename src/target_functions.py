"""
دوال الهدف المتقدمة لنظرية الأعداد
Advanced Target Functions for Number Theory
"""

import numpy as np
import matplotlib.pyplot as plt
from number_theory_utils import NumberTheoryUtils

class TargetFunctions:
    """مجموعة دوال الهدف المتقدمة لتدريب نموذج GSE"""
    
    @staticmethod
    def prime_indicator_function(x_range):
        """
        دالة مؤشر الأعداد الأولية
        y(x) = 1 إذا كان x أولي، 0 إذا لم يكن
        """
        x_vals = np.arange(x_range[0], x_range[1] + 1)
        y_vals = np.array([1 if NumberTheoryUtils.is_prime(int(x)) else 0 for x in x_vals])
        
        return x_vals, y_vals
    
    @staticmethod
    def prime_counting_function(x_range):
        """
        دالة عد الأعداد الأولية π(x)
        عدد الأعداد الأولية ≤ x
        """
        x_vals = np.arange(x_range[0], x_range[1] + 1)
        y_vals = []
        count = 0
        
        for x in x_vals:
            if NumberTheoryUtils.is_prime(int(x)):
                count += 1
            y_vals.append(count)
        
        return x_vals, np.array(y_vals)
    
    @staticmethod
    def smoothed_prime_counting(x_range, smoothing_factor=0.1):
        """
        نسخة مُنعمة من دالة عد الأعداد الأولية
        أسهل للنمذجة من الدالة الدرجية الحادة
        """
        x_vals, pi_vals = TargetFunctions.prime_counting_function(x_range)
        
        # تطبيق تنعيم باستخدام متوسط متحرك
        window_size = max(1, int(len(x_vals) * smoothing_factor))
        smoothed_vals = np.convolve(pi_vals, np.ones(window_size)/window_size, mode='same')
        
        return x_vals, smoothed_vals
    
    @staticmethod
    def von_mangoldt_function(x_range):
        """
        دالة فون مانجولت Λ(n)
        Λ(n) = log(p) إذا كان n = p^k، 0 عدا ذلك
        """
        x_vals = np.arange(x_range[0], x_range[1] + 1)
        lambda_vals = []
        
        for n in x_vals:
            if n <= 1:
                lambda_vals.append(0)
                continue
            
            # البحث عن أصغر عامل أولي
            for p in range(2, int(np.sqrt(n)) + 1):
                if n % p == 0:
                    # فحص ما إذا كان n = p^k
                    temp_n = n
                    while temp_n % p == 0:
                        temp_n //= p
                    if temp_n == 1:  # n = p^k
                        lambda_vals.append(np.log(p))
                    else:
                        lambda_vals.append(0)
                    break
            else:
                # n عدد أولي
                lambda_vals.append(np.log(n))
        
        return x_vals, np.array(lambda_vals)
    
    @staticmethod
    def prime_density_function(x_range, window_size=10):
        """
        دالة كثافة الأعداد الأولية المحلية
        نسبة الأعداد الأولية في نافذة متحركة
        """
        x_vals = np.arange(x_range[0], x_range[1] + 1)
        density_vals = []
        
        for i, x in enumerate(x_vals):
            # تحديد النافذة
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(x_vals), i + window_size // 2 + 1)
            
            window_vals = x_vals[start_idx:end_idx]
            prime_count = sum(1 for val in window_vals if NumberTheoryUtils.is_prime(int(val)))
            
            density = prime_count / len(window_vals)
            density_vals.append(density)
        
        return x_vals, np.array(density_vals)
    
    @staticmethod
    def logarithmic_integral_approximation(x_range):
        """
        تقريب التكامل اللوغاريتمي لدالة عد الأعداد الأولية
        Li(x) ≈ x / ln(x) (تقريب أولي)
        """
        x_vals = np.arange(max(2, x_range[0]), x_range[1] + 1)
        li_vals = x_vals / np.log(x_vals)
        
        # إضافة نقاط x < 2 إذا لزم الأمر
        if x_range[0] < 2:
            extra_x = np.arange(x_range[0], 2)
            extra_y = np.zeros(len(extra_x))  # Li(x) ≈ 0 for x < 2
            x_vals = np.concatenate([extra_x, x_vals])
            li_vals = np.concatenate([extra_y, li_vals])
        
        return x_vals, li_vals
    
    @staticmethod
    def riemann_prime_approximation(x_range, num_zeros=10):
        """
        تقريب الأعداد الأولية باستخدام صيغة ريمان الصريحة
        (نسخة مبسطة)
        """
        x_vals = np.arange(max(2, x_range[0]), x_range[1] + 1)
        
        # الحد الرئيسي: Li(x)
        main_term = x_vals / np.log(x_vals)
        
        # إضافة تصحيحات من أصفار زيتا (مبسطة)
        zeros = NumberTheoryUtils.riemann_zeta_zeros_approximation(num_zeros)
        corrections = np.zeros_like(x_vals, dtype=complex)
        
        for zero in zeros:
            rho = zero
            # تصحيح من الصفر rho
            correction = np.power(x_vals.astype(complex), rho) / rho
            corrections += correction.real  # أخذ الجزء الحقيقي فقط
        
        approximation = main_term + corrections.real
        
        # إضافة نقاط x < 2 إذا لزم الأمر
        if x_range[0] < 2:
            extra_x = np.arange(x_range[0], 2)
            extra_y = np.zeros(len(extra_x))
            x_vals = np.concatenate([extra_x, x_vals])
            approximation = np.concatenate([extra_y, approximation])
        
        return x_vals, approximation.real
    
    @staticmethod
    def composite_target_function(x_range, weights=None):
        """
        دالة هدف مركبة تجمع عدة دوال
        """
        if weights is None:
            weights = {'pi': 0.4, 'density': 0.3, 'von_mangoldt': 0.2, 'li': 0.1}
        
        x_vals = np.arange(x_range[0], x_range[1] + 1)
        composite_vals = np.zeros(len(x_vals))
        
        # دالة عد الأعداد الأولية
        if 'pi' in weights:
            _, pi_vals = TargetFunctions.prime_counting_function(x_range)
            # تطبيع
            pi_normalized = pi_vals / np.max(pi_vals) if np.max(pi_vals) > 0 else pi_vals
            composite_vals += weights['pi'] * pi_normalized
        
        # كثافة الأعداد الأولية
        if 'density' in weights:
            _, density_vals = TargetFunctions.prime_density_function(x_range)
            composite_vals += weights['density'] * density_vals
        
        # دالة فون مانجولت
        if 'von_mangoldt' in weights:
            _, vm_vals = TargetFunctions.von_mangoldt_function(x_range)
            # تطبيع
            vm_normalized = vm_vals / np.max(vm_vals) if np.max(vm_vals) > 0 else vm_vals
            composite_vals += weights['von_mangoldt'] * vm_normalized
        
        # التكامل اللوغاريتمي
        if 'li' in weights:
            _, li_vals = TargetFunctions.logarithmic_integral_approximation(x_range)
            # تطبيع
            li_normalized = li_vals / np.max(li_vals) if np.max(li_vals) > 0 else li_vals
            composite_vals += weights['li'] * li_normalized
        
        return x_vals, composite_vals
    
    @staticmethod
    def plot_target_functions(x_range=(2, 100)):
        """رسم جميع دوال الهدف للمقارنة"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('دوال الهدف المتقدمة لنظرية الأعداد', fontsize=16, fontweight='bold')
        
        # دالة مؤشر الأعداد الأولية
        x1, y1 = TargetFunctions.prime_indicator_function(x_range)
        axes[0, 0].plot(x1, y1, 'ro-', markersize=2, linewidth=1)
        axes[0, 0].set_title('دالة مؤشر الأعداد الأولية')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('مؤشر(x)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # دالة عد الأعداد الأولية
        x2, y2 = TargetFunctions.prime_counting_function(x_range)
        axes[0, 1].plot(x2, y2, 'b-', linewidth=2)
        axes[0, 1].set_title('دالة عد الأعداد الأولية π(x)')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('π(x)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # دالة عد الأعداد الأولية المُنعمة
        x3, y3 = TargetFunctions.smoothed_prime_counting(x_range)
        axes[0, 2].plot(x3, y3, 'g-', linewidth=2)
        axes[0, 2].set_title('π(x) المُنعمة')
        axes[0, 2].set_xlabel('x')
        axes[0, 2].set_ylabel('π(x) منعمة')
        axes[0, 2].grid(True, alpha=0.3)
        
        # دالة فون مانجولت
        x4, y4 = TargetFunctions.von_mangoldt_function(x_range)
        axes[1, 0].plot(x4, y4, 'm-', linewidth=1)
        axes[1, 0].set_title('دالة فون مانجولت Λ(n)')
        axes[1, 0].set_xlabel('n')
        axes[1, 0].set_ylabel('Λ(n)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # كثافة الأعداد الأولية
        x5, y5 = TargetFunctions.prime_density_function(x_range)
        axes[1, 1].plot(x5, y5, 'c-', linewidth=2)
        axes[1, 1].set_title('كثافة الأعداد الأولية المحلية')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('الكثافة')
        axes[1, 1].grid(True, alpha=0.3)
        
        # مقارنة π(x) مع Li(x)
        x6, y6 = TargetFunctions.prime_counting_function(x_range)
        x7, y7 = TargetFunctions.logarithmic_integral_approximation(x_range)
        axes[1, 2].plot(x6, y6, 'r-', label='π(x)', linewidth=2)
        axes[1, 2].plot(x7, y7, 'b--', label='Li(x)', linewidth=2)
        axes[1, 2].set_title('مقارنة π(x) مع Li(x)')
        axes[1, 2].set_xlabel('x')
        axes[1, 2].set_ylabel('القيمة')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def get_difficulty_ranking():
        """ترتيب دوال الهدف حسب صعوبة النمذجة"""
        return {
            'سهل': [
                'logarithmic_integral_approximation',
                'smoothed_prime_counting'
            ],
            'متوسط': [
                'prime_counting_function',
                'prime_density_function'
            ],
            'صعب': [
                'von_mangoldt_function',
                'prime_indicator_function'
            ],
            'متقدم جداً': [
                'riemann_prime_approximation',
                'composite_target_function'
            ]
        }
    
    @staticmethod
    def recommend_target_for_model(model_complexity='medium'):
        """اقتراح دالة هدف مناسبة حسب تعقيد النموذج"""
        recommendations = {
            'simple': 'smoothed_prime_counting',
            'medium': 'prime_counting_function', 
            'complex': 'von_mangoldt_function',
            'advanced': 'composite_target_function'
        }
        
        return recommendations.get(model_complexity, 'prime_counting_function')
