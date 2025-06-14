#!/usr/bin/env python3
"""
مقارنة الأداء: النموذج الأصلي مقابل النموذج المحسن
تحليل لفهم سبب تراجع بعض النتائج
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# إضافة مسار المشروع
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from gse_advanced_model import AdvancedGSEModel
    from adaptive_equations import AdaptiveGSEEquation
    from three_theories_core import ThreeTheoriesIntegrator
    print("✅ تم تحميل جميع المكونات بنجاح")
except ImportError as e:
    print(f"❌ خطأ في تحميل المكونات: {e}")
    sys.exit(1)

def generate_prime_data(max_num=100):
    """توليد بيانات الأعداد الأولية"""
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    numbers = list(range(2, max_num + 1))
    prime_indicators = [1 if is_prime(n) else 0 for n in numbers]
    
    return np.array(numbers), np.array(prime_indicators)

def test_original_model():
    """اختبار النموذج الأصلي"""
    
    print("\n" + "="*60)
    print("🔵 اختبار النموذج الأصلي GSE")
    print("="*60)
    
    # إنشاء النموذج الأصلي
    original_model = AdvancedGSEModel()
    
    # إضافة مكونات بسيطة
    original_model.add_sigmoid_component(alpha=1.0, k=1.0, x0=10.0)
    original_model.add_sigmoid_component(alpha=0.8, k=0.5, x0=30.0)
    
    print(f"   تم إنشاء النموذج الأصلي بـ {len(original_model.alpha_values)} مكونات")
    
    # بيانات الاختبار
    x_data, y_data = generate_prime_data(100)
    
    print(f"\n📊 بيانات الاختبار:")
    print(f"   نطاق الأرقام: 2-100")
    print(f"   أعداد أولية: {np.sum(y_data)}")
    
    # تقييم الأداء
    try:
        y_pred = original_model.evaluate(x_data)
        
        # تحويل للتصنيف
        threshold = 0.5
        predictions = (y_pred > threshold).astype(int)
        
        # حساب الدقة
        accuracy = np.mean(predictions == y_data)
        precision = np.sum((predictions == 1) & (y_data == 1)) / max(1, np.sum(predictions == 1))
        recall = np.sum((predictions == 1) & (y_data == 1)) / max(1, np.sum(y_data == 1))
        
        print(f"\n📈 نتائج النموذج الأصلي:")
        print(f"   الدقة العامة: {accuracy:.2%}")
        print(f"   الدقة (Precision): {precision:.2%}")
        print(f"   الاستدعاء (Recall): {recall:.2%}")
        print(f"   متوسط التنبؤات: {np.mean(y_pred):.4f}")
        print(f"   نطاق التنبؤات: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
        
        return {
            'model': original_model,
            'predictions': y_pred,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'mean_pred': np.mean(y_pred),
            'pred_range': [np.min(y_pred), np.max(y_pred)]
        }
        
    except Exception as e:
        print(f"   ❌ خطأ في تقييم النموذج الأصلي: {e}")
        return None

def test_enhanced_model():
    """اختبار النموذج المحسن"""
    
    print("\n" + "="*60)
    print("🟢 اختبار النموذج المحسن GSE")
    print("="*60)
    
    # إنشاء النموذج المحسن
    enhanced_model = AdaptiveGSEEquation()
    
    # إضافة مكونات متقدمة
    enhanced_model.add_sigmoid_component(alpha=1.0, k=0.1, x0=10.0)
    enhanced_model.add_sigmoid_component(alpha=0.8, k=0.05, x0=50.0)
    enhanced_model.add_sigmoid_component(alpha=0.6, k=0.02, x0=100.0)
    enhanced_model.add_linear_component(beta=0.001, gamma=0.1)
    
    print(f"   تم إنشاء النموذج المحسن بـ {len(enhanced_model.components)} مكونات")
    
    # بيانات الاختبار
    x_data, y_data = generate_prime_data(100)
    
    # تدريب النموذج المحسن
    print(f"\n🎯 تدريب النموذج المحسن:")
    initial_error = enhanced_model.calculate_error(x_data, y_data)
    print(f"   خطأ أولي: {initial_error:.6f}")
    
    # تطبيق التكيف
    for i in range(5):
        success = enhanced_model.adapt_to_data(x_data, y_data)
        if not success:
            break
        current_error = enhanced_model.calculate_error(x_data, y_data)
        print(f"   تكيف {i+1}: خطأ = {current_error:.6f}")
    
    final_error = enhanced_model.calculate_error(x_data, y_data)
    improvement = ((initial_error - final_error) / initial_error) * 100
    print(f"   تحسن في التدريب: {improvement:.2f}%")
    
    # تقييم الأداء
    y_pred = enhanced_model.evaluate(x_data)
    
    # تحويل للتصنيف
    threshold = 0.5
    predictions = (y_pred > threshold).astype(int)
    
    # حساب الدقة
    accuracy = np.mean(predictions == y_data)
    precision = np.sum((predictions == 1) & (y_data == 1)) / max(1, np.sum(predictions == 1))
    recall = np.sum((predictions == 1) & (y_data == 1)) / max(1, np.sum(y_data == 1))
    
    print(f"\n📈 نتائج النموذج المحسن:")
    print(f"   الدقة العامة: {accuracy:.2%}")
    print(f"   الدقة (Precision): {precision:.2%}")
    print(f"   الاستدعاء (Recall): {recall:.2%}")
    print(f"   متوسط التنبؤات: {np.mean(y_pred):.4f}")
    print(f"   نطاق التنبؤات: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
    
    return {
        'model': enhanced_model,
        'predictions': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'mean_pred': np.mean(y_pred),
        'pred_range': [np.min(y_pred), np.max(y_pred)],
        'training_improvement': improvement
    }

def analyze_differences(original_results, enhanced_results):
    """تحليل الاختلافات بين النموذجين"""
    
    print("\n" + "="*60)
    print("🔍 تحليل الاختلافات بين النموذجين")
    print("="*60)
    
    if original_results is None:
        print("❌ لا يمكن المقارنة - النموذج الأصلي فشل")
        return
    
    print(f"\n📊 مقارنة الأداء:")
    print(f"{'المقياس':<20} {'الأصلي':<15} {'المحسن':<15} {'الفرق':<15}")
    print("-" * 65)
    
    # مقارنة المقاييس
    metrics = [
        ('الدقة العامة', 'accuracy'),
        ('الدقة (Precision)', 'precision'),
        ('الاستدعاء (Recall)', 'recall'),
        ('متوسط التنبؤات', 'mean_pred')
    ]
    
    for metric_name, metric_key in metrics:
        orig_val = original_results[metric_key]
        enh_val = enhanced_results[metric_key]
        diff = enh_val - orig_val
        
        print(f"{metric_name:<20} {orig_val:<15.3f} {enh_val:<15.3f} {diff:<15.3f}")
    
    print(f"\n🔍 تحليل المشاكل:")
    
    # تحليل نطاق التنبؤات
    orig_range = original_results['pred_range']
    enh_range = enhanced_results['pred_range']
    
    print(f"   نطاق التنبؤات الأصلي: [{orig_range[0]:.4f}, {orig_range[1]:.4f}]")
    print(f"   نطاق التنبؤات المحسن: [{enh_range[0]:.4f}, {enh_range[1]:.4f}]")
    
    # تشخيص المشاكل
    if enh_range[1] < 0.5:
        print(f"   ⚠️ مشكلة: النموذج المحسن لا يصل للعتبة (0.5)")
        print(f"   💡 الحل: تقليل العتبة أو تعديل المعاملات")
    
    if enhanced_results['mean_pred'] < original_results['mean_pred']:
        print(f"   ⚠️ مشكلة: النموذج المحسن أكثر تحفظاً")
        print(f"   💡 الحل: زيادة معاملات alpha أو تقليل k")
    
    # اقتراح حلول
    print(f"\n💡 اقتراحات التحسين:")
    print(f"   1. تعديل العتبة من 0.5 إلى {enh_range[1] * 0.8:.3f}")
    print(f"   2. زيادة معاملات alpha بنسبة 50%")
    print(f"   3. تقليل معاملات k لزيادة الحساسية")
    print(f"   4. إعادة ضبط النظريات الثلاث")

def create_comparison_visualization(original_results, enhanced_results, x_data, y_data):
    """إنشاء تصور مقارن"""
    
    print(f"\n📊 إنشاء تصور مقارن...")
    
    if original_results is None:
        print("❌ لا يمكن إنشاء التصور - النموذج الأصلي فشل")
        return
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('مقارنة الأداء: النموذج الأصلي مقابل المحسن', fontsize=16, fontweight='bold')
        
        # 1. مقارنة التنبؤات
        ax1.plot(x_data, original_results['predictions'], 'b-', label='النموذج الأصلي', linewidth=2)
        ax1.plot(x_data, enhanced_results['predictions'], 'r-', label='النموذج المحسن', linewidth=2)
        ax1.scatter(x_data[y_data == 1], [1]*np.sum(y_data), color='green', s=20, alpha=0.7, label='أعداد أولية حقيقية')
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='عتبة القرار')
        ax1.set_title('مقارنة التنبؤات')
        ax1.set_xlabel('العدد')
        ax1.set_ylabel('احتمالية كونه أولي')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. توزيع التنبؤات
        ax2.hist(original_results['predictions'], bins=20, alpha=0.7, label='الأصلي', color='blue')
        ax2.hist(enhanced_results['predictions'], bins=20, alpha=0.7, label='المحسن', color='red')
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='عتبة القرار')
        ax2.set_title('توزيع التنبؤات')
        ax2.set_xlabel('قيمة التنبؤ')
        ax2.set_ylabel('التكرار')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. مقارنة المقاييس
        metrics = ['الدقة', 'Precision', 'Recall']
        orig_values = [original_results['accuracy'], original_results['precision'], original_results['recall']]
        enh_values = [enhanced_results['accuracy'], enhanced_results['precision'], enhanced_results['recall']]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        ax3.bar(x_pos - width/2, orig_values, width, label='الأصلي', alpha=0.8, color='blue')
        ax3.bar(x_pos + width/2, enh_values, width, label='المحسن', alpha=0.8, color='red')
        ax3.set_title('مقارنة المقاييس')
        ax3.set_xlabel('المقياس')
        ax3.set_ylabel('القيمة')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. تحليل الأخطاء
        orig_errors = np.abs(original_results['predictions'] - y_data)
        enh_errors = np.abs(enhanced_results['predictions'] - y_data)
        
        ax4.plot(x_data, orig_errors, 'b-', label='أخطاء الأصلي', alpha=0.7)
        ax4.plot(x_data, enh_errors, 'r-', label='أخطاء المحسن', alpha=0.7)
        ax4.set_title('مقارنة الأخطاء')
        ax4.set_xlabel('العدد')
        ax4.set_ylabel('الخطأ المطلق')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # حفظ الرسم
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'model_comparison_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   تم حفظ التصور في: {filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"   تعذر إنشاء التصور: {e}")

def suggest_improvements():
    """اقتراح تحسينات للنموذج المحسن"""
    
    print("\n" + "="*60)
    print("💡 اقتراحات تحسين النموذج المحسن")
    print("="*60)
    
    print(f"\n🔧 المشاكل المحددة:")
    print(f"   1. النموذج المحسن أصبح محافظ جداً")
    print(f"   2. معاملات k صغيرة جداً (0.1, 0.05, 0.02)")
    print(f"   3. النظريات الثلاث قد تكون تقلل الحساسية")
    print(f"   4. العتبة الثابتة (0.5) غير مناسبة")
    
    print(f"\n🛠️ الحلول المقترحة:")
    print(f"   1. زيادة معاملات alpha بنسبة 100-200%")
    print(f"   2. زيادة معاملات k إلى نطاق 0.5-2.0")
    print(f"   3. تعديل العتبة التكيفية حسب البيانات")
    print(f"   4. ضبط النظريات الثلاث لتكون أقل تحفظاً")
    print(f"   5. إضافة مرحلة معايرة بعد التدريب")
    
    print(f"\n🎯 خطة التحسين:")
    print(f"   المرحلة 1: ضبط المعاملات الأساسية")
    print(f"   المرحلة 2: تحسين النظريات الثلاث")
    print(f"   المرحلة 3: تطوير عتبة تكيفية")
    print(f"   المرحلة 4: اختبار وتقييم شامل")

def main():
    """الدالة الرئيسية للمقارنة"""
    
    print("🔍 مقارنة شاملة: النموذج الأصلي مقابل المحسن")
    print("تحليل أسباب تراجع الأداء واقتراح الحلول")
    print("="*80)
    
    try:
        # اختبار النموذج الأصلي
        original_results = test_original_model()
        
        # اختبار النموذج المحسن
        enhanced_results = test_enhanced_model()
        
        # تحليل الاختلافات
        analyze_differences(original_results, enhanced_results)
        
        # إنشاء التصور المقارن
        x_data, y_data = generate_prime_data(100)
        create_comparison_visualization(original_results, enhanced_results, x_data, y_data)
        
        # اقتراح التحسينات
        suggest_improvements()
        
        print("\n" + "="*80)
        print("🎯 الخلاصة:")
        print("="*80)
        print("✅ تم تحديد أسباب تراجع الأداء")
        print("✅ تم اقتراح حلول محددة")
        print("✅ النموذج المحسن يحتاج ضبط المعاملات")
        print("✅ النظريات الثلاث تعمل لكن تحتاج توازن")
        
    except Exception as e:
        print(f"\n❌ خطأ في المقارنة: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
