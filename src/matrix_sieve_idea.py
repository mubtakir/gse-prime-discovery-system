#!/usr/bin/env python3
"""
تطبيق فكرة الغربال المصفوفي للأعداد الأولية
الفكرة: استخدام مصفوفة الضرب المتعامدة لاكتشاف الأعداد الأولية
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

def matrix_sieve_method(max_num=100):
    """
    تطبيق فكرة الغربال المصفوفي
    """
    
    print(f"🔍 تطبيق فكرة الغربال المصفوفي حتى {max_num}")
    print("="*60)
    
    # الخطوة 1: الحصول على الأعداد الفردية
    odd_numbers = [n for n in range(3, max_num + 1, 2)]
    print(f"📊 الأعداد الفردية: {len(odd_numbers)} عدد")
    print(f"   أول 10: {odd_numbers[:10]}")
    print(f"   آخر 10: {odd_numbers[-10:]}")
    
    # الخطوة 2: إنشاء مصفوفة الضرب
    print(f"\n🔢 إنشاء مصفوفة الضرب...")
    
    # نأخذ الجذر التربيعي تقريب<|im_start|> لتحديد حجم المصفوفة المطلوبة
    matrix_size = int(np.sqrt(max_num)) + 1
    matrix_odds = [n for n in range(3, matrix_size * 2, 2) if n <= max_num]
    
    print(f"   حجم المصفوفة: {len(matrix_odds)} x {len(matrix_odds)}")
    print(f"   أعداد المصفوفة: {matrix_odds}")
    
    # إنشاء مصفوفة الضرب
    multiplication_matrix = np.zeros((len(matrix_odds), len(matrix_odds)), dtype=int)
    multiplication_products = set()
    
    for i, num1 in enumerate(matrix_odds):
        for j, num2 in enumerate(matrix_odds):
            product = num1 * num2
            multiplication_matrix[i, j] = product
            if product <= max_num:
                multiplication_products.add(product)
    
    print(f"   نواتج الضرب المختلفة: {len(multiplication_products)}")
    print(f"   أمثلة على النواتج: {sorted(list(multiplication_products))[:15]}")
    
    # الخطوة 3: حذف النواتج من الأعداد الفردية
    print(f"\n🗑️ حذف النواتج من الأعداد الفردية...")
    
    original_count = len(odd_numbers)
    remaining_numbers = [num for num in odd_numbers if num not in multiplication_products]
    
    # إضافة العدد 2 (الوحيد الزوجي الأولي)
    prime_candidates = [2] + remaining_numbers
    
    print(f"   الأعداد الأصلية: {original_count}")
    print(f"   المحذوفة: {original_count - len(remaining_numbers)}")
    print(f"   المتبقية: {len(remaining_numbers)}")
    print(f"   مع إضافة 2: {len(prime_candidates)}")
    
    return {
        'odd_numbers': odd_numbers,
        'matrix_odds': matrix_odds,
        'multiplication_matrix': multiplication_matrix,
        'multiplication_products': multiplication_products,
        'prime_candidates': prime_candidates,
        'remaining_numbers': remaining_numbers
    }

def verify_results(results, max_num):
    """
    التحقق من صحة النتائج مقارنة بالطريقة التقليدية
    """
    
    print(f"\n✅ التحقق من صحة النتائج...")
    
    # الطريقة التقليدية لإيجاد الأعداد الأولية
    def is_prime_traditional(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    traditional_primes = [n for n in range(2, max_num + 1) if is_prime_traditional(n)]
    matrix_primes = sorted(results['prime_candidates'])
    
    print(f"   الطريقة التقليدية: {len(traditional_primes)} عدد أولي")
    print(f"   الطريقة المصفوفية: {len(matrix_primes)} عدد أولي")
    
    # مقارنة النتائج
    traditional_set = set(traditional_primes)
    matrix_set = set(matrix_primes)
    
    correct_primes = traditional_set & matrix_set
    missed_primes = traditional_set - matrix_set
    false_primes = matrix_set - traditional_set
    
    print(f"\n📊 نتائج المقارنة:")
    print(f"   أعداد أولية صحيحة: {len(correct_primes)}")
    print(f"   أعداد أولية مفقودة: {len(missed_primes)}")
    print(f"   أعداد خاطئة: {len(false_primes)}")
    
    if missed_primes:
        print(f"   المفقودة: {sorted(list(missed_primes))[:10]}")
    if false_primes:
        print(f"   الخاطئة: {sorted(list(false_primes))[:10]}")
    
    accuracy = len(correct_primes) / len(traditional_set) * 100
    precision = len(correct_primes) / len(matrix_set) * 100 if matrix_set else 0
    
    print(f"\n🎯 مقاييس الأداء:")
    print(f"   الدقة (Accuracy): {accuracy:.2f}%")
    print(f"   الدقة (Precision): {precision:.2f}%")
    
    return {
        'traditional_primes': traditional_primes,
        'matrix_primes': matrix_primes,
        'correct_primes': correct_primes,
        'missed_primes': missed_primes,
        'false_primes': false_primes,
        'accuracy': accuracy,
        'precision': precision
    }

def visualize_matrix_sieve(results, verification, max_num):
    """
    تصور فكرة الغربال المصفوفي
    """
    
    print(f"\n📈 إنشاء تصور للغربال المصفوفي...")
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('الغربال المصفوفي للأعداد الأولية - فكرة مبتكرة', fontsize=16, fontweight='bold')
        
        # 1. مصفوفة الضرب
        ax1.imshow(results['multiplication_matrix'], cmap='viridis', aspect='auto')
        ax1.set_title('مصفوفة الضرب المتعامدة')
        ax1.set_xlabel('الأعداد الفردية (المحور السيني)')
        ax1.set_ylabel('الأعداد الفردية (المحور الصادي)')
        
        # إضافة قيم المصفوفة
        matrix_size = min(8, len(results['matrix_odds']))  # عرض 8x8 فقط للوضوح
        for i in range(matrix_size):
            for j in range(matrix_size):
                value = results['multiplication_matrix'][i, j]
                if value <= max_num:
                    ax1.text(j, i, str(value), ha='center', va='center', 
                            color='white' if value > 50 else 'black', fontsize=8)
        
        # 2. مقارنة النتائج
        x_range = range(2, min(max_num + 1, 50))  # عرض أول 50 عدد للوضوح
        traditional_indicators = [1 if x in verification['traditional_primes'] else 0 for x in x_range]
        matrix_indicators = [1 if x in verification['matrix_primes'] else 0 for x in x_range]
        
        ax2.plot(x_range, traditional_indicators, 'bo-', label='الطريقة التقليدية', markersize=4)
        ax2.plot(x_range, matrix_indicators, 'ro-', label='الطريقة المصفوفية', markersize=4, alpha=0.7)
        ax2.set_title('مقارنة النتائج')
        ax2.set_xlabel('العدد')
        ax2.set_ylabel('أولي (1) أم لا (0)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. توزيع الأعداد الأولية
        prime_gaps = []
        traditional_primes = verification['traditional_primes']
        for i in range(1, len(traditional_primes)):
            gap = traditional_primes[i] - traditional_primes[i-1]
            prime_gaps.append(gap)
        
        ax3.hist(prime_gaps, bins=range(1, max(prime_gaps) + 2), alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('توزيع الفجوات بين الأعداد الأولية')
        ax3.set_xlabel('حجم الفجوة')
        ax3.set_ylabel('التكرار')
        ax3.grid(True, alpha=0.3)
        
        # 4. مقاييس الأداء
        metrics = ['الدقة\n(Accuracy)', 'الدقة\n(Precision)']
        values = [verification['accuracy'], verification['precision']]
        colors = ['lightgreen', 'lightblue']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        ax4.set_title('مقاييس أداء الطريقة المصفوفية')
        ax4.set_ylabel('النسبة المئوية (%)')
        ax4.set_ylim(0, 105)
        
        # إضافة قيم على الأعمدة
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # حفظ الرسم
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'matrix_sieve_visualization_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   ✅ تم حفظ التصور في: {filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"   ❌ تعذر إنشاء التصور: {e}")

def analyze_efficiency(results, verification, max_num):
    """
    تحليل كفاءة الطريقة المصفوفية
    """
    
    print(f"\n⚡ تحليل كفاءة الطريقة المصفوفية...")
    
    # حساب التعقيد الحاسوبي
    matrix_size = len(results['matrix_odds'])
    matrix_operations = matrix_size ** 2
    traditional_operations = sum(int(np.sqrt(n)) for n in range(2, max_num + 1))
    
    print(f"📊 مقارنة التعقيد الحاسوبي:")
    print(f"   الطريقة المصفوفية: {matrix_operations} عملية ضرب")
    print(f"   الطريقة التقليدية: ~{traditional_operations} عملية قسمة")
    
    # تحليل استخدام الذاكرة
    matrix_memory = matrix_size ** 2 * 4  # 4 bytes per integer
    traditional_memory = max_num * 1  # 1 byte per boolean
    
    print(f"\n💾 مقارنة استخدام الذاكرة:")
    print(f"   الطريقة المصفوفية: {matrix_memory} bytes")
    print(f"   الطريقة التقليدية: {traditional_memory} bytes")
    
    # تحليل الدقة
    print(f"\n🎯 تحليل الدقة:")
    print(f"   أعداد أولية صحيحة: {len(verification['correct_primes'])}")
    print(f"   أعداد مفقودة: {len(verification['missed_primes'])}")
    print(f"   أعداد خاطئة: {len(verification['false_primes'])}")
    
    # اقتراحات التحسين
    print(f"\n💡 اقتراحات التحسين:")
    if verification['missed_primes']:
        print(f"   1. زيادة حجم المصفوفة لتغطية أعداد أكبر")
        print(f"   2. تطبيق الغربال على مراحل متعددة")
    
    if verification['false_primes']:
        print(f"   3. إضافة مرحلة تحقق إضافية")
        print(f"   4. تحسين خوارزمية الحذف")
    
    print(f"   5. تحسين استخدام الذاكرة باستخدام مصفوفات متناثرة")
    print(f"   6. تطبيق التوازي في العمليات")

def suggest_gse_integration(results, verification):
    """
    اقتراح دمج الفكرة مع نموذج GSE
    """
    
    print(f"\n🔗 اقتراح دمج الفكرة مع نموذج GSE...")
    print("="*60)
    
    print(f"💡 طرق الدمج المقترحة:")
    
    print(f"\n1. 🎯 استخدام المصفوفة كمدخل للنموذج:")
    print(f"   - تحويل مصفوفة الضرب إلى ميزات")
    print(f"   - تدريب GSE على أنماط المصفوفة")
    print(f"   - استخدام موقع العدد في المصفوفة كميزة")
    
    print(f"\n2. 🧠 دمج المنطق المصفوفي في GSE:")
    print(f"   - إضافة طبقة 'مصفوفة الضرب' للنموذج")
    print(f"   - استخدام النواتج كميزات سلبية")
    print(f"   - تعزيز التنبؤ بناءً على موقع المصفوفة")
    
    print(f"\n3. 🔄 نموذج هجين:")
    print(f"   - الغربال المصفوفي للتصفية الأولية")
    print(f"   - GSE للتنبؤ بالأعداد المتبقية")
    print(f"   - دمج النتائج للحصول على دقة أعلى")
    
    print(f"\n4. 📊 استخدام البيانات المصفوفية:")
    print(f"   - تحليل أنماط توزيع النواتج")
    print(f"   - استخدام الفجوات في المصفوفة كميزات")
    print(f"   - تدريب GSE على العلاقات المصفوفية")
    
    print(f"\n🚀 الفوائد المتوقعة:")
    print(f"   ✅ تحسين دقة التنبؤ")
    print(f"   ✅ تقليل التعقيد الحاسوبي")
    print(f"   ✅ فهم أعمق لأنماط الأعداد الأولية")
    print(f"   ✅ نموذج أكثر قابلية للتفسير")

def main():
    """
    الدالة الرئيسية لتطبيق فكرة الغربال المصفوفي
    """
    
    print("🧠 تطبيق فكرة الغربال المصفوفي للأعداد الأولية")
    print("فكرة مبتكرة لاكتشاف الأعداد الأولية باستخدام مصفوفة الضرب المتعامدة")
    print("="*80)
    
    try:
        # تطبيق الفكرة
        max_num = 100
        results = matrix_sieve_method(max_num)
        
        # التحقق من النتائج
        verification = verify_results(results, max_num)
        
        # تصور النتائج
        visualize_matrix_sieve(results, verification, max_num)
        
        # تحليل الكفاءة
        analyze_efficiency(results, verification, max_num)
        
        # اقتراح الدمج مع GSE
        suggest_gse_integration(results, verification)
        
        print(f"\n" + "="*80)
        print(f"🎉 تم تطبيق الفكرة بنجاح!")
        print(f"✅ دقة الطريقة: {verification['accuracy']:.2f}%")
        print(f"✅ دقة التنبؤ: {verification['precision']:.2f}%")
        print(f"🔍 الفكرة واعدة ويمكن تطويرها أكثر!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ خطأ في تطبيق الفكرة: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
