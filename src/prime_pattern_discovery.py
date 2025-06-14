"""
اكتشاف الأنماط والقوانين الجديدة في الأعداد الأولية
Prime Pattern Discovery - New Laws and Conjectures
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from scipy import stats
from scipy.optimize import curve_fit

# إضافة مسار المصدر
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gse_advanced_model import AdvancedGSEModel
from optimizer_advanced import GSEOptimizer
from target_functions import TargetFunctions
from number_theory_utils import NumberTheoryUtils

class PrimePatternDiscovery:
    """اكتشاف الأنماط والقوانين الجديدة في الأعداد الأولية"""
    
    def __init__(self):
        self.primes = []
        self.prime_gaps = []
        self.prime_densities = []
        self.discovered_patterns = {}
        self.conjectures = []
        self.experiment_log = []
    
    def log_message(self, message):
        """تسجيل رسالة مع الوقت"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.experiment_log.append(log_entry)
    
    def analyze_prime_distribution_patterns(self, max_n=2000):
        """تحليل أنماط توزيع الأعداد الأولية"""
        
        self.log_message("🔍 تحليل أنماط توزيع الأعداد الأولية...")
        
        # توليد الأعداد الأولية
        self.primes = NumberTheoryUtils.generate_primes(max_n)
        self.log_message(f"   تم توليد {len(self.primes)} عدد أولي حتى {max_n}")
        
        # حساب الفجوات
        self.prime_gaps = [self.primes[i+1] - self.primes[i] for i in range(len(self.primes)-1)]
        
        # حساب الكثافات المحلية
        window_size = 100
        self.prime_densities = []
        
        for i in range(window_size, max_n, window_size):
            primes_in_window = len([p for p in self.primes if i-window_size < p <= i])
            density = primes_in_window / window_size
            self.prime_densities.append((i, density))
        
        # تحليل الأنماط
        patterns = {
            'gap_analysis': self.analyze_gap_patterns(),
            'density_analysis': self.analyze_density_patterns(),
            'modular_analysis': self.analyze_modular_patterns(),
            'clustering_analysis': self.analyze_clustering_patterns(),
            'spiral_analysis': self.analyze_spiral_patterns()
        }
        
        self.discovered_patterns = patterns
        return patterns
    
    def analyze_gap_patterns(self):
        """تحليل أنماط فجوات الأعداد الأولية"""
        
        self.log_message("   📏 تحليل فجوات الأعداد الأولية...")
        
        gaps = np.array(self.prime_gaps)
        
        # إحصائيات أساسية
        gap_stats = {
            'mean': np.mean(gaps),
            'std': np.std(gaps),
            'max': np.max(gaps),
            'min': np.min(gaps),
            'median': np.median(gaps)
        }
        
        # توزيع الفجوات
        unique_gaps, gap_counts = np.unique(gaps, return_counts=True)
        gap_distribution = dict(zip(unique_gaps, gap_counts))
        
        # الفجوات الأكثر شيوعاً
        most_common_gaps = sorted(gap_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # تحليل الاتجاه العام للفجوات
        x_positions = np.arange(len(gaps))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_positions, gaps)
        
        # البحث عن أنماط دورية في الفجوات
        gap_autocorr = np.correlate(gaps, gaps, mode='full')
        gap_autocorr = gap_autocorr[gap_autocorr.size // 2:]
        
        return {
            'statistics': gap_stats,
            'distribution': gap_distribution,
            'most_common': most_common_gaps,
            'trend': {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value
            },
            'autocorrelation': gap_autocorr[:100].tolist()  # أول 100 قيمة
        }
    
    def analyze_density_patterns(self):
        """تحليل أنماط كثافة الأعداد الأولية"""
        
        self.log_message("   📊 تحليل كثافة الأعداد الأولية...")
        
        positions = [d[0] for d in self.prime_densities]
        densities = [d[1] for d in self.prime_densities]
        
        # مقارنة مع النظرية (1/ln(x))
        theoretical_densities = [1/np.log(x) if x > 1 else 0 for x in positions]
        
        # حساب الانحراف عن النظرية
        deviations = [abs(d - t) for d, t in zip(densities, theoretical_densities)]
        avg_deviation = np.mean(deviations)
        
        # البحث عن أنماط دورية في الكثافة
        density_fft = np.fft.fft(densities)
        dominant_frequencies = np.argsort(np.abs(density_fft))[-5:]
        
        # تحليل الاتجاه
        slope, intercept, r_value, p_value, std_err = stats.linregress(positions, densities)
        
        return {
            'positions': positions,
            'densities': densities,
            'theoretical_densities': theoretical_densities,
            'average_deviation': avg_deviation,
            'trend': {
                'slope': slope,
                'r_squared': r_value**2
            },
            'dominant_frequencies': dominant_frequencies.tolist(),
            'fft_magnitudes': np.abs(density_fft).tolist()
        }
    
    def analyze_modular_patterns(self):
        """تحليل الأنماط المعيارية (modular patterns)"""
        
        self.log_message("   🔢 تحليل الأنماط المعيارية...")
        
        modular_patterns = {}
        
        # تحليل الأعداد الأولية modulo أعداد مختلفة
        for mod in [6, 10, 12, 30, 60]:
            residues = [p % mod for p in self.primes if p > mod]
            unique_residues, counts = np.unique(residues, return_counts=True)
            
            modular_patterns[f'mod_{mod}'] = {
                'residues': unique_residues.tolist(),
                'counts': counts.tolist(),
                'distribution': dict(zip(unique_residues, counts))
            }
        
        # البحث عن أنماط خاصة
        # نمط 6n±1
        mod6_pattern = modular_patterns['mod_6']
        mod6_analysis = {
            'pattern_6n_plus_1': mod6_pattern['distribution'].get(1, 0),
            'pattern_6n_minus_1': mod6_pattern['distribution'].get(5, 0),
            'other_residues': sum(mod6_pattern['counts']) - mod6_pattern['distribution'].get(1, 0) - mod6_pattern['distribution'].get(5, 0)
        }
        
        return {
            'modular_distributions': modular_patterns,
            'mod6_analysis': mod6_analysis
        }
    
    def analyze_clustering_patterns(self):
        """تحليل أنماط التجمع في الأعداد الأولية"""
        
        self.log_message("   🎯 تحليل أنماط التجمع...")
        
        # البحث عن الأعداد الأولية التوأم
        twin_primes = []
        for i in range(len(self.primes)-1):
            if self.primes[i+1] - self.primes[i] == 2:
                twin_primes.append((self.primes[i], self.primes[i+1]))
        
        # البحث عن الأعداد الأولية الثلاثية
        triplet_primes = []
        for i in range(len(self.primes)-2):
            if (self.primes[i+1] - self.primes[i] == 2 and 
                self.primes[i+2] - self.primes[i+1] == 4) or \
               (self.primes[i+1] - self.primes[i] == 4 and 
                self.primes[i+2] - self.primes[i+1] == 2):
                triplet_primes.append((self.primes[i], self.primes[i+1], self.primes[i+2]))
        
        # تحليل المسافات بين التجمعات
        twin_positions = [tp[0] for tp in twin_primes]
        if len(twin_positions) > 1:
            twin_gaps = [twin_positions[i+1] - twin_positions[i] for i in range(len(twin_positions)-1)]
            avg_twin_gap = np.mean(twin_gaps)
        else:
            twin_gaps = []
            avg_twin_gap = 0
        
        return {
            'twin_primes': twin_primes[:20],  # أول 20 زوج
            'twin_count': len(twin_primes),
            'triplet_primes': triplet_primes[:10],  # أول 10 ثلاثيات
            'triplet_count': len(triplet_primes),
            'twin_gaps': twin_gaps[:50],  # أول 50 فجوة
            'average_twin_gap': avg_twin_gap
        }
    
    def analyze_spiral_patterns(self):
        """تحليل أنماط الحلزون (Ulam Spiral)"""
        
        self.log_message("   🌀 تحليل أنماط الحلزون...")
        
        # إنشاء حلزون أولام
        size = 50  # حجم الشبكة
        spiral = np.zeros((size, size))
        
        # ملء الحلزون
        x, y = size // 2, size // 2
        spiral[x, y] = 1
        
        directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # يمين، أعلى، يسار، أسفل
        direction = 0
        steps = 1
        num = 2
        
        while num <= size * size:
            for _ in range(2):  # كل اتجاه يتكرر مرتين
                for _ in range(steps):
                    if num > size * size:
                        break
                    dx, dy = directions[direction]
                    x, y = x + dx, y + dy
                    if 0 <= x < size and 0 <= y < size:
                        spiral[x, y] = num
                        num += 1
                direction = (direction + 1) % 4
                if num > size * size:
                    break
            steps += 1
        
        # تحديد الأعداد الأولية في الحلزون
        prime_positions = []
        for i in range(size):
            for j in range(size):
                if spiral[i, j] > 0 and NumberTheoryUtils.is_prime(int(spiral[i, j])):
                    prime_positions.append((i, j, int(spiral[i, j])))
        
        # تحليل الأنماط الخطية
        diagonal_primes = []
        for i in range(size):
            # القطر الرئيسي
            if spiral[i, i] > 0 and NumberTheoryUtils.is_prime(int(spiral[i, i])):
                diagonal_primes.append(int(spiral[i, i]))
            # القطر الثانوي
            if spiral[i, size-1-i] > 0 and NumberTheoryUtils.is_prime(int(spiral[i, size-1-i])):
                diagonal_primes.append(int(spiral[i, size-1-i]))
        
        return {
            'spiral_size': size,
            'total_primes_in_spiral': len(prime_positions),
            'prime_positions': prime_positions[:50],  # أول 50 موقع
            'diagonal_primes': diagonal_primes,
            'prime_density_in_spiral': len(prime_positions) / (size * size)
        }
    
    def discover_new_conjectures(self):
        """اكتشاف فرضيات جديدة بناءً على الأنماط المكتشفة"""
        
        self.log_message("🔬 اكتشاف فرضيات جديدة...")
        
        conjectures = []
        
        # فرضية 1: نمط الفجوات المتزايدة
        gap_analysis = self.discovered_patterns['gap_analysis']
        if gap_analysis['trend']['slope'] > 0:
            conjecture1 = {
                'name': 'فرضية الفجوات المتزايدة المعدلة',
                'statement': f'متوسط فجوات الأعداد الأولية يزيد بمعدل {gap_analysis["trend"]["slope"]:.6f} لكل عدد أولي',
                'confidence': gap_analysis['trend']['r_squared'],
                'evidence': f'R² = {gap_analysis["trend"]["r_squared"]:.4f}'
            }
            conjectures.append(conjecture1)
        
        # فرضية 2: نمط الكثافة المحسن
        density_analysis = self.discovered_patterns['density_analysis']
        if density_analysis['average_deviation'] < 0.01:
            conjecture2 = {
                'name': 'فرضية الكثافة المحسنة',
                'statement': f'كثافة الأعداد الأولية تتبع نمط 1/ln(x) مع انحراف متوسط {density_analysis["average_deviation"]:.6f}',
                'confidence': 1 - density_analysis['average_deviation'],
                'evidence': f'متوسط الانحراف = {density_analysis["average_deviation"]:.6f}'
            }
            conjectures.append(conjecture2)
        
        # فرضية 3: نمط الأعداد الأولية التوأم
        clustering_analysis = self.discovered_patterns['clustering_analysis']
        if clustering_analysis['twin_count'] > 0:
            twin_density = clustering_analysis['twin_count'] / len(self.primes)
            conjecture3 = {
                'name': 'فرضية كثافة الأعداد الأولية التوأم',
                'statement': f'نسبة الأعداد الأولية التوأم تقارب {twin_density:.4f} من إجمالي الأعداد الأولية',
                'confidence': 0.8,  # ثقة متوسطة
                'evidence': f'{clustering_analysis["twin_count"]} زوج توأم من أصل {len(self.primes)} عدد أولي'
            }
            conjectures.append(conjecture3)
        
        # فرضية 4: نمط معياري جديد
        modular_analysis = self.discovered_patterns['modular_analysis']
        mod6_analysis = modular_analysis['mod6_analysis']
        total_mod6 = mod6_analysis['pattern_6n_plus_1'] + mod6_analysis['pattern_6n_minus_1']
        if total_mod6 > 0:
            ratio_6n_plus_1 = mod6_analysis['pattern_6n_plus_1'] / total_mod6
            conjecture4 = {
                'name': 'فرضية التوزيع المعياري 6n±1',
                'statement': f'من الأعداد الأولية الكبيرة، {ratio_6n_plus_1:.2%} تأتي في شكل 6n+1 والباقي 6n-1',
                'confidence': 0.9,
                'evidence': f'{mod6_analysis["pattern_6n_plus_1"]} عدد 6n+1 و {mod6_analysis["pattern_6n_minus_1"]} عدد 6n-1'
            }
            conjectures.append(conjecture4)
        
        # فرضية 5: نمط الحلزون
        spiral_analysis = self.discovered_patterns['spiral_analysis']
        spiral_density = spiral_analysis['prime_density_in_spiral']
        conjecture5 = {
            'name': 'فرضية كثافة حلزون أولام',
            'statement': f'كثافة الأعداد الأولية في حلزون أولام تقارب {spiral_density:.4f}',
            'confidence': 0.7,
            'evidence': f'{spiral_analysis["total_primes_in_spiral"]} عدد أولي في شبكة {spiral_analysis["spiral_size"]}×{spiral_analysis["spiral_size"]}'
        }
        conjectures.append(conjecture5)
        
        self.conjectures = conjectures
        return conjectures
    
    def formulate_gse_prime_law(self):
        """صياغة قانون GSE للأعداد الأولية"""
        
        self.log_message("⚖️ صياغة قانون GSE للأعداد الأولية...")
        
        # تدريب نموذج GSE على البيانات المكتشفة
        x_data = np.array(range(2, len(self.primes) + 2))
        y_data = np.array(self.primes)
        
        # إنشاء نموذج GSE متقدم
        gse_model = AdvancedGSEModel()
        gse_model.add_sigmoid(alpha=complex(1.0, 0.1), n=2.0, z=complex(1.0, 0.0), x0=10.0)
        gse_model.add_sigmoid(alpha=complex(0.8, -0.1), n=1.8, z=complex(0.9, 0.1), x0=50.0)
        gse_model.add_sigmoid(alpha=complex(0.6, 0.2), n=1.5, z=complex(1.1, -0.1), x0=100.0)
        
        # التدريب
        optimizer = GSEOptimizer(gse_model)
        result = optimizer.optimize_differential_evolution(x_data, y_data, max_iter=200, verbose=False)
        
        # تقييم النموذج
        y_pred = gse_model.evaluate(x_data)
        r2 = 1 - (np.sum((y_data - y_pred) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2))
        
        # استخراج المعاملات
        components = []
        for i, comp in enumerate(gse_model.sigmoid_components):
            components.append({
                'component': i + 1,
                'alpha': comp['alpha'],
                'n': comp['n'],
                'z': comp['z'],
                'x0': comp['x0']
            })
        
        # صياغة القانون
        gse_law = {
            'name': 'قانون GSE للأعداد الأولية',
            'formula': 'P(n) = Σ[αᵢ * (1 + e^(-nᵢ*(x-x₀ᵢ)*zᵢ))^(-1)]',
            'components': components,
            'accuracy': r2,
            'domain': f'n ∈ [1, {len(self.primes)}]',
            'description': 'قانون رياضي يصف العدد الأولي رقم n باستخدام مجموع دوال سيجمويد معقدة'
        }
        
        return gse_law
    
    def generate_comprehensive_report(self):
        """إنتاج تقرير شامل للاكتشافات"""
        
        report = []
        report.append("=" * 80)
        report.append("تقرير اكتشاف الأنماط والقوانين الجديدة في الأعداد الأولية")
        report.append("=" * 80)
        
        # ملخص البيانات
        report.append(f"\n📊 ملخص البيانات:")
        report.append(f"   عدد الأعداد الأولية المحللة: {len(self.primes)}")
        report.append(f"   أكبر عدد أولي: {max(self.primes) if self.primes else 'غير محدد'}")
        report.append(f"   عدد الفجوات المحللة: {len(self.prime_gaps)}")
        
        # الأنماط المكتشفة
        if 'gap_analysis' in self.discovered_patterns:
            gap_stats = self.discovered_patterns['gap_analysis']['statistics']
            report.append(f"\n📏 تحليل الفجوات:")
            report.append(f"   متوسط الفجوة: {gap_stats['mean']:.2f}")
            report.append(f"   أكبر فجوة: {gap_stats['max']}")
            report.append(f"   أصغر فجوة: {gap_stats['min']}")
        
        # الفرضيات المكتشفة
        if self.conjectures:
            report.append(f"\n🔬 الفرضيات المكتشفة:")
            for i, conjecture in enumerate(self.conjectures, 1):
                report.append(f"   {i}. {conjecture['name']}")
                report.append(f"      البيان: {conjecture['statement']}")
                report.append(f"      الثقة: {conjecture['confidence']:.2%}")
                report.append(f"      الدليل: {conjecture['evidence']}")
                report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_complete_discovery(self, max_n=2000):
        """تشغيل عملية الاكتشاف الكاملة"""
        
        self.log_message("🚀 بدء عملية اكتشاف الأنماط الشاملة")
        self.log_message("=" * 60)
        
        # 1. تحليل الأنماط
        patterns = self.analyze_prime_distribution_patterns(max_n)
        
        # 2. اكتشاف الفرضيات
        conjectures = self.discover_new_conjectures()
        
        # 3. صياغة قانون GSE
        gse_law = self.formulate_gse_prime_law()
        
        # 4. إنتاج التقرير
        report = self.generate_comprehensive_report()
        
        self.log_message("\n" + "=" * 60)
        self.log_message("🎉 انتهت عملية الاكتشاف!")
        
        return {
            'patterns': patterns,
            'conjectures': conjectures,
            'gse_law': gse_law,
            'report': report
        }

def main():
    """تشغيل اكتشاف الأنماط"""
    
    discovery = PrimePatternDiscovery()
    
    # تشغيل الاكتشاف الكامل
    results = discovery.run_complete_discovery(max_n=3000)
    
    # طباعة التقرير
    print("\n" + results['report'])
    
    # طباعة الفرضيات
    print("\n🔬 الفرضيات المكتشفة:")
    for conjecture in results['conjectures']:
        print(f"\n📋 {conjecture['name']}")
        print(f"   {conjecture['statement']}")
        print(f"   الثقة: {conjecture['confidence']:.2%}")
    
    # طباعة قانون GSE
    print(f"\n⚖️ {results['gse_law']['name']}")
    print(f"   الصيغة: {results['gse_law']['formula']}")
    print(f"   الدقة: {results['gse_law']['accuracy']:.4f}")
    
    return discovery, results

if __name__ == "__main__":
    discovery, results = main()
