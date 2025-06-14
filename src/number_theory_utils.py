"""
أدوات نظرية الأعداد المتقدمة
لتوليد البيانات المرجعية والتحليل الرياضي
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta
import sympy as sp

class NumberTheoryUtils:
    """أدوات نظرية الأعداد المتقدمة"""
    
    @staticmethod
    def is_prime(n):
        """فحص ما إذا كان العدد أولياً - محسن"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        # فحص الأعداد الفردية فقط حتى الجذر التربيعي
        for i in range(3, int(np.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    @staticmethod
    def generate_primes(limit):
        """توليد جميع الأعداد الأولية حتى حد معين باستخدام غربال إراتوستينس"""
        if limit < 2:
            return []
        
        # إنشاء مصفوفة منطقية
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        # تطبيق غربال إراتوستينس
        for i in range(2, int(np.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        # استخراج الأعداد الأولية
        primes = [i for i in range(2, limit + 1) if sieve[i]]
        return primes
    
    @staticmethod
    def generate_prime_data(start, end):
        """توليد بيانات الأعداد الأولية (x, y) حيث y=1 للأولي و y=0 لغير الأولي"""
        x = np.arange(start, end + 1)
        y = np.array([1 if NumberTheoryUtils.is_prime(int(xi)) else 0 for xi in x])
        return x, y
    
    @staticmethod
    def prime_counting_function(x_max):
        """دالة عد الأعداد الأولية π(x)"""
        x_vals = np.arange(2, x_max + 1)
        pi_vals = []
        count = 0
        
        for x in x_vals:
            if NumberTheoryUtils.is_prime(x):
                count += 1
            pi_vals.append(count)
        
        return x_vals, np.array(pi_vals)
    
    @staticmethod
    def prime_indicator_function(x_max):
        """دالة مؤشر الأعداد الأولية"""
        x_vals = np.arange(2, x_max + 1)
        indicator_vals = [1 if NumberTheoryUtils.is_prime(x) else 0 for x in x_vals]
        
        return x_vals, np.array(indicator_vals)
    
    @staticmethod
    def von_mangoldt_function(x_max):
        """دالة فون مانجولت Λ(n)"""
        x_vals = np.arange(1, x_max + 1)
        lambda_vals = []
        
        for n in x_vals:
            if n == 1:
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
    def mobius_function(x_max):
        """دالة موبيوس μ(n)"""
        x_vals = np.arange(1, x_max + 1)
        mu_vals = []
        
        for n in x_vals:
            if n == 1:
                mu_vals.append(1)
                continue
            
            # تحليل العدد إلى عوامل أولية
            factors = []
            temp_n = n
            
            for p in range(2, int(np.sqrt(temp_n)) + 1):
                count = 0
                while temp_n % p == 0:
                    temp_n //= p
                    count += 1
                if count > 0:
                    factors.append((p, count))
            
            if temp_n > 1:
                factors.append((temp_n, 1))
            
            # حساب μ(n)
            if any(count > 1 for _, count in factors):
                mu_vals.append(0)  # يحتوي على مربع عدد أولي
            elif len(factors) % 2 == 0:
                mu_vals.append(1)   # عدد زوجي من العوامل الأولية المختلفة
            else:
                mu_vals.append(-1)  # عدد فردي من العوامل الأولية المختلفة
        
        return x_vals, np.array(mu_vals)
    
    @staticmethod
    def euler_totient_function(x_max):
        """دالة أويلر φ(n)"""
        x_vals = np.arange(1, x_max + 1)
        phi_vals = []
        
        for n in x_vals:
            if n == 1:
                phi_vals.append(1)
                continue
            
            result = n
            
            # تطبيق صيغة أويلر
            p = 2
            while p * p <= n:
                if n % p == 0:
                    while n % p == 0:
                        n //= p
                    result -= result // p
                p += 1
            
            if n > 1:
                result -= result // n
            
            phi_vals.append(result)
        
        return x_vals, np.array(phi_vals)
    
    @staticmethod
    def riemann_zeta_zeros_approximation(num_zeros=10):
        """تقريب أول عدة أصفار لدالة زيتا ريمان"""
        # الأصفار غير البديهية المعروفة (الجزء التخيلي)
        known_zeros = [
            14.134725142,
            21.022039639,
            25.010857580,
            30.424876126,
            32.935061588,
            37.586178159,
            40.918719012,
            43.327073281,
            48.005150881,
            49.773832478
        ]
        
        zeros = []
        for i in range(min(num_zeros, len(known_zeros))):
            zeros.append(complex(0.5, known_zeros[i]))
        
        return zeros
    
    @staticmethod
    def plot_number_theory_functions(x_max=100):
        """رسم دوال نظرية الأعداد المختلفة"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('دوال نظرية الأعداد الأساسية', fontsize=16)
        
        # دالة مؤشر الأعداد الأولية
        x1, y1 = NumberTheoryUtils.prime_indicator_function(x_max)
        axes[0, 0].plot(x1, y1, 'ro-', markersize=3)
        axes[0, 0].set_title('دالة مؤشر الأعداد الأولية')
        axes[0, 0].set_xlabel('n')
        axes[0, 0].set_ylabel('π(n)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # دالة عد الأعداد الأولية
        x2, y2 = NumberTheoryUtils.prime_counting_function(x_max)
        axes[0, 1].plot(x2, y2, 'b-', linewidth=2)
        axes[0, 1].set_title('دالة عد الأعداد الأولية π(x)')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('π(x)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # دالة فون مانجولت
        x3, y3 = NumberTheoryUtils.von_mangoldt_function(x_max)
        axes[0, 2].plot(x3, y3, 'g-', linewidth=1)
        axes[0, 2].set_title('دالة فون مانجولت Λ(n)')
        axes[0, 2].set_xlabel('n')
        axes[0, 2].set_ylabel('Λ(n)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # دالة موبيوس
        x4, y4 = NumberTheoryUtils.mobius_function(x_max)
        axes[1, 0].plot(x4, y4, 'mo-', markersize=2)
        axes[1, 0].set_title('دالة موبيوس μ(n)')
        axes[1, 0].set_xlabel('n')
        axes[1, 0].set_ylabel('μ(n)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # دالة أويلر
        x5, y5 = NumberTheoryUtils.euler_totient_function(x_max)
        axes[1, 1].plot(x5, y5, 'co-', markersize=2)
        axes[1, 1].set_title('دالة أويلر φ(n)')
        axes[1, 1].set_xlabel('n')
        axes[1, 1].set_ylabel('φ(n)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # مقارنة π(x) مع x/ln(x)
        x6 = np.arange(2, x_max + 1)
        pi_x = [sum(1 for p in range(2, x+1) if NumberTheoryUtils.is_prime(p)) for x in x6]
        li_x = x6 / np.log(x6)  # تقريب لوغاريتمي
        
        axes[1, 2].plot(x6, pi_x, 'r-', label='π(x)', linewidth=2)
        axes[1, 2].plot(x6, li_x, 'b--', label='x/ln(x)', linewidth=2)
        axes[1, 2].set_title('مقارنة π(x) مع التقريب اللوغاريتمي')
        axes[1, 2].set_xlabel('x')
        axes[1, 2].set_ylabel('القيمة')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class PrimeAnalyzer:
    """محلل متقدم للأعداد الأولية"""
    
    def __init__(self):
        self.primes_cache = {}
    
    def analyze_prime_gaps(self, limit=1000):
        """تحليل الفجوات بين الأعداد الأولية"""
        primes = NumberTheoryUtils.generate_primes(limit)
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
        
        analysis = {
            'total_primes': len(primes),
            'average_gap': np.mean(gaps),
            'max_gap': max(gaps),
            'min_gap': min(gaps),
            'gap_distribution': np.histogram(gaps, bins=20)
        }
        
        return analysis, gaps
    
    def twin_primes_analysis(self, limit=1000):
        """تحليل الأعداد الأولية التوأم"""
        primes = NumberTheoryUtils.generate_primes(limit)
        twin_primes = []
        
        for i in range(len(primes)-1):
            if primes[i+1] - primes[i] == 2:
                twin_primes.append((primes[i], primes[i+1]))
        
        return twin_primes
    
    def prime_density_analysis(self, limit=1000, window_size=100):
        """تحليل كثافة الأعداد الأولية"""
        densities = []
        positions = []
        
        for start in range(2, limit - window_size, window_size // 2):
            end = start + window_size
            count = sum(1 for n in range(start, end) if NumberTheoryUtils.is_prime(n))
            density = count / window_size
            densities.append(density)
            positions.append(start + window_size // 2)
        
        return positions, densities
