#!/usr/bin/env python3
"""
مجموعة أدوات البحث العلمي المتقدمة
أدوات متخصصة للبحث في نظرية الأعداد والأنماط الرياضية
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import pandas as pd
from datetime import datetime
import json
import sys
import os

# إضافة مسار المشروع
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_hybrid_system import AdvancedHybridSystem
    from enhanced_matrix_sieve import enhanced_matrix_sieve
    print("✅ تم تحميل مجموعة أدوات البحث")
except ImportError as e:
    print(f"❌ خطأ في تحميل المكونات: {e}")
    sys.exit(1)

class PrimeResearchToolkit:
    """
    مجموعة أدوات البحث في الأعداد الأولية
    """
    
    def __init__(self):
        self.hybrid_system = AdvancedHybridSystem()
        self.research_data = {}
        self.hypotheses = []
        self.experiments = []
        
        print("🔬 تم إنشاء مجموعة أدوات البحث العلمي")
    
    def prime_distribution_analysis(self, max_num=1000, intervals=10):
        """
        تحليل توزيع الأعداد الأولية
        """
        
        print(f"\n📊 تحليل توزيع الأعداد الأولية حتى {max_num}")
        print("="*60)
        
        # الحصول على الأعداد الأولية
        primes = self._get_primes(max_num)
        
        # تقسيم النطاق إلى فترات
        interval_size = max_num // intervals
        interval_data = []
        
        for i in range(intervals):
            start = i * interval_size + 1
            end = (i + 1) * interval_size
            if i == intervals - 1:
                end = max_num
            
            primes_in_interval = [p for p in primes if start <= p <= end]
            density = len(primes_in_interval) / interval_size
            
            interval_data.append({
                'interval': f"{start}-{end}",
                'start': start,
                'end': end,
                'count': len(primes_in_interval),
                'density': density,
                'theoretical_density': 1 / np.log(end) if end > 1 else 0
            })
        
        # تحليل الاتجاهات
        densities = [d['density'] for d in interval_data]
        theoretical = [d['theoretical_density'] for d in interval_data]
        
        # حساب الارتباط
        correlation = np.corrcoef(densities, theoretical)[0, 1]
        
        print(f"📈 نتائج التحليل:")
        print(f"   إجمالي الأعداد الأولية: {len(primes)}")
        print(f"   الكثافة العامة: {len(primes)/max_num:.6f}")
        print(f"   الارتباط مع النظرية: {correlation:.4f}")
        
        # عرض تفاصيل الفترات
        print(f"\n📊 تفاصيل الفترات:")
        for data in interval_data:
            print(f"   {data['interval']}: {data['count']} أعداد، كثافة = {data['density']:.6f}")
        
        return {
            'primes': primes,
            'interval_data': interval_data,
            'correlation': correlation,
            'total_primes': len(primes),
            'overall_density': len(primes)/max_num
        }
    
    def gap_analysis(self, max_num=1000):
        """
        تحليل الفجوات بين الأعداد الأولية
        """
        
        print(f"\n🔍 تحليل الفجوات بين الأعداد الأولية حتى {max_num}")
        print("="*60)
        
        primes = self._get_primes(max_num)
        
        if len(primes) < 2:
            print("❌ عدد غير كافٍ من الأعداد الأولية للتحليل")
            return None
        
        # حساب الفجوات
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
        
        # إحصائيات الفجوات
        try:
            mode_result = stats.mode(gaps, keepdims=True)
            mode_value = mode_result.mode[0] if len(mode_result.mode) > 0 else gaps[0] if gaps else 0
        except:
            # حساب المنوال يدوي<|im_start|>
            unique_gaps, gap_counts = np.unique(gaps, return_counts=True)
            mode_value = unique_gaps[np.argmax(gap_counts)] if len(unique_gaps) > 0 else 0

        gap_stats = {
            'mean': np.mean(gaps),
            'median': np.median(gaps),
            'std': np.std(gaps),
            'min': np.min(gaps),
            'max': np.max(gaps),
            'mode': mode_value
        }
        
        # توزيع الفجوات
        unique_gaps, gap_counts = np.unique(gaps, return_counts=True)
        gap_distribution = dict(zip(unique_gaps, gap_counts))
        
        # تحليل الأنماط
        even_gaps = [g for g in gaps if g % 2 == 0]
        odd_gaps = [g for g in gaps if g % 2 == 1]
        
        print(f"📊 إحصائيات الفجوات:")
        print(f"   إجمالي الفجوات: {len(gaps)}")
        print(f"   متوسط الفجوة: {gap_stats['mean']:.2f}")
        print(f"   الوسيط: {gap_stats['median']:.2f}")
        print(f"   الانحراف المعياري: {gap_stats['std']:.2f}")
        print(f"   أصغر فجوة: {gap_stats['min']}")
        print(f"   أكبر فجوة: {gap_stats['max']}")
        print(f"   الفجوة الأكثر شيوع<|im_start|>: {gap_stats['mode']}")
        
        print(f"\n🔢 تحليل الأنماط:")
        print(f"   فجوات زوجية: {len(even_gaps)} ({len(even_gaps)/len(gaps)*100:.1f}%)")
        print(f"   فجوات فردية: {len(odd_gaps)} ({len(odd_gaps)/len(gaps)*100:.1f}%)")
        
        # الفجوات الأكثر شيوع<|im_start|>
        print(f"\n📈 أكثر الفجوات شيوع<|im_start|>:")
        sorted_gaps = sorted(gap_distribution.items(), key=lambda x: x[1], reverse=True)
        for gap, count in sorted_gaps[:10]:
            percentage = count / len(gaps) * 100
            print(f"   فجوة {gap}: {count} مرة ({percentage:.1f}%)")
        
        return {
            'gaps': gaps,
            'statistics': gap_stats,
            'distribution': gap_distribution,
            'even_gaps': len(even_gaps),
            'odd_gaps': len(odd_gaps),
            'primes': primes
        }
    
    def twin_prime_analysis(self, max_num=1000):
        """
        تحليل الأعداد الأولية التوأم
        """
        
        print(f"\n👥 تحليل الأعداد الأولية التوأم حتى {max_num}")
        print("="*60)
        
        primes = self._get_primes(max_num)
        
        # العثور على الأعداد الأولية التوأم
        twin_primes = []
        for i in range(len(primes)-1):
            if primes[i+1] - primes[i] == 2:
                twin_primes.append((primes[i], primes[i+1]))
        
        # تحليل التوزيع
        twin_count = len(twin_primes)
        twin_density = twin_count / len(primes) if primes else 0
        
        # تحليل الاتجاهات
        intervals = 10
        interval_size = max_num // intervals
        interval_twins = []
        
        for i in range(intervals):
            start = i * interval_size + 1
            end = (i + 1) * interval_size
            if i == intervals - 1:
                end = max_num
            
            twins_in_interval = [(p1, p2) for p1, p2 in twin_primes if start <= p1 <= end]
            interval_twins.append({
                'interval': f"{start}-{end}",
                'count': len(twins_in_interval),
                'density': len(twins_in_interval) / interval_size
            })
        
        print(f"📊 نتائج تحليل الأعداد التوأم:")
        print(f"   إجمالي الأعداد الأولية: {len(primes)}")
        print(f"   أزواج الأعداد التوأم: {twin_count}")
        print(f"   نسبة الأعداد التوأم: {twin_density:.4f}")
        
        if twin_primes:
            print(f"\n👥 أمثلة على الأعداد التوأم:")
            for i, (p1, p2) in enumerate(twin_primes[:10]):
                print(f"   {i+1:2d}. ({p1}, {p2})")
            
            if len(twin_primes) > 10:
                print(f"   ... و {len(twin_primes)-10} زوج آخر")
        
        print(f"\n📈 توزيع الأعداد التوأم عبر الفترات:")
        for data in interval_twins:
            print(f"   {data['interval']}: {data['count']} أزواج")
        
        return {
            'twin_primes': twin_primes,
            'twin_count': twin_count,
            'twin_density': twin_density,
            'interval_analysis': interval_twins,
            'total_primes': len(primes)
        }
    
    def prime_patterns_discovery(self, max_num=500):
        """
        اكتشاف الأنماط في الأعداد الأولية
        """
        
        print(f"\n🔍 اكتشاف الأنماط في الأعداد الأولية حتى {max_num}")
        print("="*60)
        
        primes = self._get_primes(max_num)
        
        patterns = {}
        
        # 1. تحليل الأرقام الأخيرة
        last_digits = [p % 10 for p in primes if p > 10]
        last_digit_dist = {}
        for digit in last_digits:
            last_digit_dist[digit] = last_digit_dist.get(digit, 0) + 1
        
        patterns['last_digits'] = last_digit_dist
        
        # 2. تحليل الأرقام الأولى
        first_digits = [int(str(p)[0]) for p in primes if p >= 10]
        first_digit_dist = {}
        for digit in first_digits:
            first_digit_dist[digit] = first_digit_dist.get(digit, 0) + 1
        
        patterns['first_digits'] = first_digit_dist
        
        # 3. تحليل مجموع الأرقام
        digit_sums = [sum(int(d) for d in str(p)) for p in primes]
        digit_sum_dist = {}
        for s in digit_sums:
            digit_sum_dist[s] = digit_sum_dist.get(s, 0) + 1
        
        patterns['digit_sums'] = digit_sum_dist
        
        # 4. تحليل الأنماط الحسابية
        arithmetic_progressions = self._find_arithmetic_progressions(primes)
        patterns['arithmetic_progressions'] = arithmetic_progressions
        
        # 5. تحليل الأنماط الهندسية (تقريبية)
        geometric_patterns = self._find_geometric_patterns(primes)
        patterns['geometric_patterns'] = geometric_patterns
        
        print(f"📊 نتائج اكتشاف الأنماط:")
        
        print(f"\n🔢 توزيع الأرقام الأخيرة:")
        for digit, count in sorted(last_digit_dist.items()):
            percentage = count / len(last_digits) * 100
            print(f"   الرقم {digit}: {count} مرة ({percentage:.1f}%)")
        
        print(f"\n🔢 توزيع الأرقام الأولى:")
        for digit, count in sorted(first_digit_dist.items()):
            percentage = count / len(first_digits) * 100
            print(f"   الرقم {digit}: {count} مرة ({percentage:.1f}%)")
        
        print(f"\n➕ أكثر مجاميع الأرقام شيوع<|im_start|>:")
        sorted_sums = sorted(digit_sum_dist.items(), key=lambda x: x[1], reverse=True)
        for s, count in sorted_sums[:10]:
            percentage = count / len(digit_sums) * 100
            print(f"   المجموع {s}: {count} مرة ({percentage:.1f}%)")
        
        if arithmetic_progressions:
            print(f"\n📐 المتتاليات الحسابية المكتشفة:")
            for i, prog in enumerate(arithmetic_progressions[:5]):
                print(f"   {i+1}. {prog['sequence']} (فرق = {prog['difference']})")
        
        return patterns
    
    def _find_arithmetic_progressions(self, primes, min_length=3):
        """
        العثور على المتتاليات الحسابية في الأعداد الأولية
        """
        
        progressions = []
        
        for i in range(len(primes)):
            for j in range(i+1, len(primes)):
                diff = primes[j] - primes[i]
                sequence = [primes[i], primes[j]]
                
                # البحث عن المزيد من الأعداد في نفس المتتالية
                next_expected = primes[j] + diff
                for k in range(j+1, len(primes)):
                    if primes[k] == next_expected:
                        sequence.append(primes[k])
                        next_expected += diff
                    elif primes[k] > next_expected:
                        break
                
                if len(sequence) >= min_length:
                    progressions.append({
                        'sequence': sequence,
                        'difference': diff,
                        'length': len(sequence)
                    })
        
        # إزالة التكرارات وترتيب حسب الطول
        unique_progressions = []
        seen = set()
        
        for prog in progressions:
            key = tuple(prog['sequence'])
            if key not in seen:
                seen.add(key)
                unique_progressions.append(prog)
        
        return sorted(unique_progressions, key=lambda x: x['length'], reverse=True)
    
    def _find_geometric_patterns(self, primes, tolerance=0.1):
        """
        العثور على الأنماط الهندسية التقريبية
        """
        
        patterns = []
        
        for i in range(len(primes)-2):
            for j in range(i+1, len(primes)-1):
                for k in range(j+1, len(primes)):
                    p1, p2, p3 = primes[i], primes[j], primes[k]
                    
                    # فحص النسبة الهندسية
                    if p1 > 0 and p2 > 0:
                        ratio1 = p2 / p1
                        ratio2 = p3 / p2
                        
                        if abs(ratio1 - ratio2) / ratio1 < tolerance:
                            patterns.append({
                                'sequence': [p1, p2, p3],
                                'ratio': (ratio1 + ratio2) / 2,
                                'error': abs(ratio1 - ratio2) / ratio1
                            })
        
        return sorted(patterns, key=lambda x: x['error'])[:10]
    
    def _get_primes(self, max_num):
        """
        الحصول على الأعداد الأولية
        """
        
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        return [n for n in range(2, max_num + 1) if is_prime(n)]
    
    def generate_research_report(self, max_num=1000):
        """
        إنشاء تقرير بحثي شامل
        """
        
        print(f"\n📋 إنشاء تقرير بحثي شامل للأعداد الأولية حتى {max_num}")
        print("="*80)
        
        # تشغيل جميع التحليلات
        distribution_analysis = self.prime_distribution_analysis(max_num)
        gap_analysis = self.gap_analysis(max_num)
        twin_analysis = self.twin_prime_analysis(max_num)
        pattern_analysis = self.prime_patterns_discovery(max_num)
        
        # إنشاء التقرير
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'max_number': max_num,
                'analysis_type': 'comprehensive_prime_research'
            },
            'distribution_analysis': distribution_analysis,
            'gap_analysis': gap_analysis,
            'twin_prime_analysis': twin_analysis,
            'pattern_analysis': pattern_analysis
        }
        
        # حفظ التقرير
        filename = f"prime_research_report_{max_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            # تحويل numpy arrays إلى lists للحفظ
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {str(k): convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, tuple):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(report), f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 تم حفظ التقرير البحثي في: {filename}")
        
        return report

def main():
    """
    الدالة الرئيسية لأدوات البحث
    """
    
    print("🔬 مجموعة أدوات البحث العلمي في الأعداد الأولية")
    print("أدوات متقدمة لاكتشاف الأنماط والتحليل الرياضي")
    print("="*80)
    
    try:
        toolkit = PrimeResearchToolkit()
        
        # إنشاء تقرير بحثي شامل
        report = toolkit.generate_research_report(500)
        
        print(f"\n🎉 تم الانتهاء من التحليل البحثي الشامل!")
        print(f"📊 النتائج محفوظة ومتاحة للمراجعة العلمية")
        
    except Exception as e:
        print(f"\n❌ خطأ في أدوات البحث: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
