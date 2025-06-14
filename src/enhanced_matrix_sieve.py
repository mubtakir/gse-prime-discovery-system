#!/usr/bin/env python3
"""
الغربال المصفوفي المحسن
تطوير الفكرة الأصلية مع إصلاح المشاكل وتحسين الأداء
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

def enhanced_matrix_sieve(max_num=200):
    """
    الغربال المصفوفي المحسن مع إصلاح مشكلة حجم المصفوفة
    """
    
    print(f"🔍 الغربال المصفوفي المحسن حتى {max_num}")
    print("="*60)
    
    # الخطوة 1: الحصول على الأعداد الفردية
    odd_numbers = [n for n in range(3, max_num + 1, 2)]
    print(f"📊 الأعداد الفردية: {len(odd_numbers)} عدد")
    
    # الخطوة 2: تحديد حجم المصفوفة المناسب
    # نحتاج جميع الأعداد الأولية حتى √max_num
    sqrt_max = int(np.sqrt(max_num)) + 1
    
    # إيجاد الأعداد الأولية الصغيرة أولاً (للمصفوفة)
    def is_prime_simple(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    # الأعداد الأولية الفردية للمصفوفة
    prime_odds_for_matrix = [n for n in range(3, sqrt_max + 1, 2) if is_prime_simple(n)]
    
    # إضافة بعض الأعداد المركبة الفردية الصغيرة للتأكد
    composite_odds = [9, 15, 21, 25, 27]  # أعداد مركبة مهمة
    matrix_numbers = sorted(list(set(prime_odds_for_matrix + composite_odds)))
    
    print(f"🔢 أعداد المصفوفة: {matrix_numbers}")
    print(f"   حجم المصفوفة: {len(matrix_numbers)} x {len(matrix_numbers)}")
    
    # الخطوة 3: إنشاء مصفوفة الضرب المحسنة
    print(f"\n🔢 إنشاء مصفوفة الضرب المحسنة...")
    
    multiplication_products = set()
    matrix_data = {}
    
    # إنشاء المصفوفة وحفظ البيانات
    for i, num1 in enumerate(matrix_numbers):
        for j, num2 in enumerate(matrix_numbers):
            product = num1 * num2
            if product <= max_num and product % 2 == 1:  # فقط الأعداد الفردية
                multiplication_products.add(product)
                
                # حفظ معلومات إضافية
                if product not in matrix_data:
                    matrix_data[product] = []
                matrix_data[product].append((num1, num2, i, j))
    
    print(f"   نواتج الضرب الفردية: {len(multiplication_products)}")
    print(f"   أمثلة: {sorted(list(multiplication_products))[:20]}")
    
    # الخطوة 4: الحذف الذكي مع تتبع الأسباب
    print(f"\n🗑️ الحذف الذكي من الأعداد الفردية...")
    
    removed_numbers = []
    removal_reasons = {}
    
    for num in odd_numbers:
        if num in multiplication_products:
            removed_numbers.append(num)
            removal_reasons[num] = matrix_data[num]
    
    remaining_numbers = [num for num in odd_numbers if num not in multiplication_products]
    
    # إضافة العدد 2
    prime_candidates = [2] + remaining_numbers
    
    print(f"   الأعداد الأصلية: {len(odd_numbers)}")
    print(f"   المحذوفة: {len(removed_numbers)}")
    print(f"   المتبقية: {len(remaining_numbers)}")
    print(f"   مع إضافة 2: {len(prime_candidates)}")
    
    return {
        'odd_numbers': odd_numbers,
        'matrix_numbers': matrix_numbers,
        'multiplication_products': multiplication_products,
        'matrix_data': matrix_data,
        'removed_numbers': removed_numbers,
        'removal_reasons': removal_reasons,
        'remaining_numbers': remaining_numbers,
        'prime_candidates': prime_candidates
    }

def multi_stage_sieve(max_num=500):
    """
    غربال متعدد المراحل للأعداد الكبيرة
    """
    
    print(f"\n🔄 الغربال متعدد المراحل حتى {max_num}")
    print("="*60)
    
    # المرحلة 1: الغربال الأساسي
    stage1_result = enhanced_matrix_sieve(min(max_num, 200))
    current_candidates = stage1_result['prime_candidates']
    
    print(f"📊 المرحلة 1 انتهت: {len(current_candidates)} مرشح")
    
    if max_num <= 200:
        return stage1_result
    
    # المرحلة 2: تطبيق الغربال على النطاق الأكبر
    print(f"\n🔄 المرحلة 2: النطاق {200}-{max_num}")
    
    # استخدام المرشحين من المرحلة الأولى كأساس للمصفوفة
    confirmed_primes = [p for p in current_candidates if p <= int(np.sqrt(max_num)) + 1]
    
    # الأعداد الفردية في النطاق الجديد
    new_odd_numbers = [n for n in range(201, max_num + 1, 2)]
    
    # إنشاء نواتج ضرب جديدة
    new_products = set()
    for prime in confirmed_primes:
        for odd in new_odd_numbers:
            if prime * odd <= max_num:
                new_products.add(prime * odd)
    
    # حذف النواتج الجديدة
    final_candidates = current_candidates + [n for n in new_odd_numbers if n not in new_products]
    
    print(f"   أعداد جديدة: {len(new_odd_numbers)}")
    print(f"   نواتج جديدة: {len(new_products)}")
    print(f"   مرشحين نهائيين: {len(final_candidates)}")
    
    return {
        'stage1_result': stage1_result,
        'new_odd_numbers': new_odd_numbers,
        'new_products': new_products,
        'final_candidates': sorted(final_candidates),
        'confirmed_primes': confirmed_primes
    }

def extract_matrix_features(number, matrix_result):
    """
    استخراج ميزات مصفوفية لعدد معين
    """
    
    features = {}
    
    # الميزة 1: هل العدد في نواتج الضرب؟
    features['in_products'] = 1 if number in matrix_result['multiplication_products'] else 0
    
    # الميزة 2: عدد طرق تكوين العدد
    if number in matrix_result['matrix_data']:
        features['formation_ways'] = len(matrix_result['matrix_data'][number])
        
        # الميزة 3: أصغر عامل
        factors = [min(pair[0], pair[1]) for pair in matrix_result['matrix_data'][number]]
        features['smallest_factor'] = min(factors)
        
        # الميزة 4: أكبر عامل
        features['largest_factor'] = max([max(pair[0], pair[1]) for pair in matrix_result['matrix_data'][number]])
        
        # الميزة 5: متوسط العوامل
        all_factors = []
        for pair in matrix_result['matrix_data'][number]:
            all_factors.extend([pair[0], pair[1]])
        features['average_factor'] = np.mean(all_factors)
        
    else:
        features['formation_ways'] = 0
        features['smallest_factor'] = number  # العدد نفسه
        features['largest_factor'] = number
        features['average_factor'] = number
    
    # الميزة 6: موقع في قائمة الأعداد الفردية
    if number in matrix_result['odd_numbers']:
        features['odd_position'] = matrix_result['odd_numbers'].index(number)
    else:
        features['odd_position'] = -1
    
    # الميزة 7: المسافة من أقرب عدد في المصفوفة
    if matrix_result['matrix_numbers']:
        distances = [abs(number - m) for m in matrix_result['matrix_numbers']]
        features['distance_to_matrix'] = min(distances)
    else:
        features['distance_to_matrix'] = 0
    
    # الميزة 8: نمط الرقم (آحاد)
    features['last_digit'] = number % 10
    
    # الميزة 9: قابلية القسمة على الأعداد الصغيرة
    small_primes = [3, 5, 7, 11, 13]
    for prime in small_primes:
        features[f'divisible_by_{prime}'] = 1 if number % prime == 0 else 0
    
    return features

def create_enhanced_visualization(enhanced_result, verification_result):
    """
    تصور محسن للنتائج
    """
    
    print(f"\n📈 إنشاء تصور محسن...")
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('الغربال المصفوفي المحسن - تحليل شامل', fontsize=16, fontweight='bold')
        
        # 1. مقارنة النتائج المحسنة
        traditional_primes = verification_result['traditional_primes']
        matrix_primes = verification_result['matrix_primes']
        
        # عرض أول 50 عدد للوضوح
        x_range = range(2, min(max(traditional_primes) + 1, 100))
        traditional_indicators = [1 if x in traditional_primes else 0 for x in x_range]
        matrix_indicators = [1 if x in matrix_primes else 0 for x in x_range]
        
        ax1.plot(x_range, traditional_indicators, 'bo-', label='الطريقة التقليدية', markersize=3)
        ax1.plot(x_range, matrix_indicators, 'ro-', label='الغربال المصفوفي', markersize=3, alpha=0.7)
        ax1.set_title('مقارنة النتائج المحسنة')
        ax1.set_xlabel('العدد')
        ax1.set_ylabel('أولي (1) أم لا (0)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. تحليل الأخطاء
        correct_primes = verification_result['correct_primes']
        missed_primes = verification_result['missed_primes']
        false_primes = verification_result['false_primes']
        
        categories = ['صحيحة', 'مفقودة', 'خاطئة']
        counts = [len(correct_primes), len(missed_primes), len(false_primes)]
        colors = ['green', 'orange', 'red']
        
        bars = ax2.bar(categories, counts, color=colors, alpha=0.7)
        ax2.set_title('تحليل دقة النتائج')
        ax2.set_ylabel('عدد الأعداد الأولية')
        
        # إضافة قيم على الأعمدة
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(True, alpha=0.3)
        
        # 3. توزيع أسباب الحذف
        removal_reasons = enhanced_result['removal_reasons']
        formation_ways = [len(reasons) for reasons in removal_reasons.values()]
        
        if formation_ways:
            ax3.hist(formation_ways, bins=range(1, max(formation_ways) + 2), 
                    alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_title('توزيع طرق تكوين الأعداد المحذوفة')
            ax3.set_xlabel('عدد طرق التكوين')
            ax3.set_ylabel('التكرار')
            ax3.grid(True, alpha=0.3)
        
        # 4. مقاييس الأداء المحسنة
        metrics = ['الدقة\n(Accuracy)', 'الدقة\n(Precision)', 'الاستدعاء\n(Recall)']
        values = [
            verification_result['accuracy'],
            verification_result['precision'],
            verification_result.get('recall', 0)
        ]
        colors = ['lightgreen', 'lightblue', 'lightcoral']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        ax4.set_title('مقاييس الأداء المحسنة')
        ax4.set_ylabel('النسبة المئوية (%)')
        ax4.set_ylim(0, 105)
        
        # إضافة قيم على الأعمدة
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # حفظ الرسم
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'enhanced_matrix_sieve_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   ✅ تم حفظ التصور في: {filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"   ❌ تعذر إنشاء التصور: {e}")

def verify_enhanced_results(enhanced_result, max_num):
    """
    التحقق من النتائج المحسنة
    """
    
    print(f"\n✅ التحقق من النتائج المحسنة...")
    
    # الطريقة التقليدية
    def is_prime_traditional(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    traditional_primes = [n for n in range(2, max_num + 1) if is_prime_traditional(n)]
    
    # النتائج المصفوفية
    if 'final_candidates' in enhanced_result:
        matrix_primes = sorted(enhanced_result['final_candidates'])
    else:
        matrix_primes = sorted(enhanced_result['prime_candidates'])
    
    print(f"   الطريقة التقليدية: {len(traditional_primes)} عدد أولي")
    print(f"   الطريقة المصفوفية: {len(matrix_primes)} عدد أولي")
    
    # مقارنة النتائج
    traditional_set = set(traditional_primes)
    matrix_set = set(matrix_primes)
    
    correct_primes = traditional_set & matrix_set
    missed_primes = traditional_set - matrix_set
    false_primes = matrix_set - traditional_set
    
    print(f"\n📊 نتائج المقارنة المحسنة:")
    print(f"   أعداد أولية صحيحة: {len(correct_primes)}")
    print(f"   أعداد أولية مفقودة: {len(missed_primes)}")
    print(f"   أعداد خاطئة: {len(false_primes)}")
    
    if missed_primes:
        print(f"   المفقودة: {sorted(list(missed_primes))}")
    if false_primes:
        print(f"   الخاطئة: {sorted(list(false_primes))}")
    
    # حساب المقاييس
    accuracy = len(correct_primes) / len(traditional_set) * 100 if traditional_set else 0
    precision = len(correct_primes) / len(matrix_set) * 100 if matrix_set else 0
    recall = len(correct_primes) / len(traditional_set) * 100 if traditional_set else 0
    
    print(f"\n🎯 مقاييس الأداء المحسنة:")
    print(f"   الدقة (Accuracy): {accuracy:.2f}%")
    print(f"   الدقة (Precision): {precision:.2f}%")
    print(f"   الاستدعاء (Recall): {recall:.2f}%")
    
    return {
        'traditional_primes': traditional_primes,
        'matrix_primes': matrix_primes,
        'correct_primes': correct_primes,
        'missed_primes': missed_primes,
        'false_primes': false_primes,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

def main():
    """
    الدالة الرئيسية للغربال المحسن
    """
    
    print("🚀 الغربال المصفوفي المحسن")
    print("تطوير الفكرة الأصلية مع إصلاح المشاكل وتحسين الأداء")
    print("="*80)
    
    try:
        # اختبار الغربال المحسن
        max_num = 200
        enhanced_result = enhanced_matrix_sieve(max_num)
        
        # التحقق من النتائج
        verification = verify_enhanced_results(enhanced_result, max_num)
        
        # إنشاء التصور
        create_enhanced_visualization(enhanced_result, verification)
        
        # اختبار الغربال متعدد المراحل
        print(f"\n" + "="*60)
        multi_stage_result = multi_stage_sieve(300)
        multi_verification = verify_enhanced_results(multi_stage_result, 300)
        
        print(f"\n" + "="*80)
        print(f"🎉 النتائج النهائية:")
        print(f"="*80)
        print(f"📊 الغربال المحسن (حتى {max_num}):")
        print(f"   دقة: {verification['accuracy']:.2f}%")
        print(f"   دقة التنبؤ: {verification['precision']:.2f}%")
        print(f"   استدعاء: {verification['recall']:.2f}%")
        
        print(f"\n📊 الغربال متعدد المراحل (حتى 300):")
        print(f"   دقة: {multi_verification['accuracy']:.2f}%")
        print(f"   دقة التنبؤ: {multi_verification['precision']:.2f}%")
        print(f"   استدعاء: {multi_verification['recall']:.2f}%")
        
        print(f"\n🌟 الفكرة محسنة وجاهزة للدمج مع GSE!")
        
        return enhanced_result, verification, multi_stage_result, multi_verification
        
    except Exception as e:
        print(f"\n❌ خطأ في الغربال المحسن: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    main()
