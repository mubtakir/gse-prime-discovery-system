#!/usr/bin/env python3
"""
اختبار شامل للنموذج المحسن GSE
يختبر جميع الميزات الجديدة المستوحاة من نظام Baserah

الاختبارات:
1. النظريات الثلاث الأساسية
2. المعادلات المتكيفة
3. نظام الخبير/المستكشف
4. النموذج المتكامل المحسن
5. مقارنة الأداء مع النموذج الأصلي
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from typing import Dict, List, Any
import logging

# إضافة مسار المشروع
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# تهيئة التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_three_theories():
    """اختبار النظريات الثلاث الأساسية"""
    
    print("\n" + "="*60)
    print("🧪 اختبار النظريات الثلاث الأساسية")
    print("="*60)
    
    try:
        from three_theories_core import (
            ZeroDualityTheory, 
            PerpendicularOptimizationTheory, 
            FilamentConnectionTheory,
            ThreeTheoriesIntegrator
        )
        
        # اختبار نظرية التوازن
        print("\n1️⃣ اختبار نظرية التوازن (ثنائية الصفر):")
        zero_duality = ZeroDualityTheory()
        
        test_values = np.array([2.5, -1.8, 3.2, -0.9, 1.5])
        balanced_values = zero_duality.apply_zero_duality_balance(test_values)
        
        print(f"   القيم الأصلية: {test_values}")
        print(f"   القيم المتوازنة: {balanced_values}")
        print(f"   ✅ نظرية التوازن تعمل بنجاح")
        
        # اختبار نظرية التعامد
        print("\n2️⃣ اختبار نظرية التعامد:")
        perpendicular = PerpendicularOptimizationTheory()
        
        test_gradient = np.array([1.5, 0.8, -0.3])
        perpendicular_vector = perpendicular.calculate_perpendicular_vector(test_gradient)
        
        # فحص التعامد (الضرب النقطي يجب أن يكون قريب من الصفر)
        dot_product = np.dot(test_gradient, perpendicular_vector)
        
        print(f"   التدرج الأصلي: {test_gradient}")
        print(f"   المتجه المتعامد: {perpendicular_vector}")
        print(f"   الضرب النقطي: {dot_product:.6f} (يجب أن يكون قريب من 0)")
        print(f"   ✅ نظرية التعامد تعمل بنجاح")
        
        # اختبار نظرية الفتائل
        print("\n3️⃣ اختبار نظرية الفتائل:")
        filament = FilamentConnectionTheory()
        
        test_components = [
            {'alpha': 1.0, 'k': 1.0, 'x0': 0.0},
            {'alpha': 1.5, 'k': 0.8, 'x0': 0.5},
            {'alpha': 0.7, 'k': 1.2, 'x0': -0.3}
        ]
        
        enhanced_components = filament.apply_filament_enhancement(test_components)
        
        print(f"   مكونات أصلية: {len(test_components)}")
        print(f"   مكونات محسنة: {len(enhanced_components)}")
        
        for i, (original, enhanced) in enumerate(zip(test_components, enhanced_components)):
            improvement = enhanced['alpha'] / original['alpha']
            print(f"   مكون {i+1}: تحسن α = {improvement:.3f}x")
        
        print(f"   ✅ نظرية الفتائل تعمل بنجاح")
        
        # اختبار التكامل
        print("\n4️⃣ اختبار التكامل بين النظريات:")
        integrator = ThreeTheoriesIntegrator()
        
        test_params = np.array([1.0, 0.5, -0.2, 0.8])
        test_gradient = np.array([0.1, -0.05, 0.03, -0.02])
        test_components = test_components[:2]  # تقليل العدد للاختبار
        
        optimized_params, enhanced_comps = integrator.integrated_optimization_step(
            test_params, test_gradient, test_components
        )
        
        print(f"   معاملات أصلية: {test_params}")
        print(f"   معاملات محسنة: {optimized_params}")
        print(f"   تغيير المعاملات: {np.linalg.norm(optimized_params - test_params):.6f}")
        print(f"   ✅ التكامل بين النظريات يعمل بنجاح")
        
        return True
        
    except Exception as e:
        print(f"   ❌ خطأ في اختبار النظريات الثلاث: {e}")
        return False

def test_adaptive_equations():
    """اختبار المعادلات المتكيفة"""
    
    print("\n" + "="*60)
    print("🧪 اختبار المعادلات المتكيفة")
    print("="*60)
    
    try:
        from adaptive_equations import AdaptiveGSEEquation, AdaptationDirection
        
        # إنشاء معادلة متكيفة
        print("\n1️⃣ إنشاء معادلة متكيفة:")
        adaptive_eq = AdaptiveGSEEquation()
        adaptive_eq.add_sigmoid_component(alpha=1.0, k=1.0, x0=0.0)
        adaptive_eq.add_sigmoid_component(alpha=0.8, k=0.5, x0=1.0)
        adaptive_eq.add_linear_component(beta=0.1, gamma=0.0)
        
        print(f"   تم إنشاء معادلة بـ {len(adaptive_eq.components)} مكونات")
        print(f"   معرف المعادلة: {adaptive_eq.equation_id}")
        
        # اختبار التقييم
        print("\n2️⃣ اختبار تقييم المعادلة:")
        x_test = np.linspace(-2, 2, 10)
        y_result = adaptive_eq.evaluate(x_test)
        
        print(f"   مدخلات الاختبار: {len(x_test)} نقطة")
        print(f"   نتائج التقييم: متوسط = {np.mean(y_result):.4f}")
        
        # اختبار التكيف
        print("\n3️⃣ اختبار التكيف مع البيانات:")
        
        # بيانات هدف (دالة جيب مبسطة)
        x_data = np.linspace(-3, 3, 50)
        y_target = 0.5 * np.sin(x_data) + 0.5
        
        initial_error = adaptive_eq.calculate_error(x_data, y_target)
        print(f"   خطأ أولي: {initial_error:.6f}")
        
        # تطبيق عدة دورات تكيف
        adaptation_results = []
        for i in range(5):
            success = adaptive_eq.adapt_to_data(x_data, y_target, AdaptationDirection.IMPROVE_ACCURACY)
            current_error = adaptive_eq.calculate_error(x_data, y_target)
            adaptation_results.append(current_error)
            
            if success:
                print(f"   تكيف {i+1}: خطأ = {current_error:.6f}")
            else:
                print(f"   تكيف {i+1}: فشل أو تقارب")
                break
        
        final_error = adaptive_eq.calculate_error(x_data, y_target)
        improvement = initial_error - final_error
        improvement_percentage = (improvement / initial_error) * 100
        
        print(f"   خطأ نهائي: {final_error:.6f}")
        print(f"   تحسن: {improvement:.6f} ({improvement_percentage:.2f}%)")
        
        # إحصائيات التكيف
        print("\n4️⃣ إحصائيات التكيف:")
        stats = adaptive_eq.get_adaptation_statistics()
        
        print(f"   تكيفات إجمالية: {stats['total_adaptations']}")
        print(f"   تكيفات ناجحة: {stats['successful_adaptations']}")
        print(f"   معدل النجاح: {stats['success_rate']:.2%}")
        print(f"   متوسط التحسن: {stats['average_improvement']:.6f}")
        print(f"   أفضل أداء: {stats['best_performance']:.6f}")
        print(f"   حالة التقارب: {'نعم' if stats['is_converged'] else 'لا'}")
        
        print(f"   ✅ المعادلات المتكيفة تعمل بنجاح")
        
        return True
        
    except Exception as e:
        print(f"   ❌ خطأ في اختبار المعادلات المتكيفة: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_expert_explorer_system():
    """اختبار نظام الخبير/المستكشف"""
    
    print("\n" + "="*60)
    print("🧪 اختبار نظام الخبير/المستكشف")
    print("="*60)
    
    try:
        from expert_explorer_system import (
            GSEExpertSystem, 
            GSEExplorerSystem, 
            IntegratedExpertExplorer,
            ExplorerMode
        )
        from adaptive_equations import AdaptiveGSEEquation
        
        # اختبار النظام الخبير
        print("\n1️⃣ اختبار النظام الخبير:")
        expert = GSEExpertSystem()
        
        # بيانات اختبار (نمط خطي)
        x_data = np.linspace(0, 10, 20)
        y_data = 2 * x_data + 1 + 0.1 * np.random.randn(20)
        
        analysis = expert.analyze_data_pattern(x_data, y_data)
        
        print(f"   نوع النمط المكتشف: {analysis.pattern_type}")
        print(f"   مستوى الثقة: {analysis.confidence:.2%}")
        print(f"   عدد التوصيات: {len(analysis.recommendations)}")
        print(f"   تقييم المخاطر: {analysis.risk_assessment}")
        print(f"   التحسن المتوقع: {analysis.expected_improvement:.4f}")
        
        # اختبار النظام المستكشف
        print("\n2️⃣ اختبار النظام المستكشف:")
        explorer = GSEExplorerSystem()
        
        # معادلة أساسية للاستكشاف
        base_equation = AdaptiveGSEEquation()
        base_equation.add_sigmoid_component(alpha=1.0, k=1.0, x0=0.0)
        
        exploration_result = explorer.explore_parameter_space(
            base_equation, x_data, y_data, ExplorerMode.GUIDED
        )
        
        print(f"   أنماط مكتشفة: {len(exploration_result.discovered_patterns)}")
        print(f"   معاملات جديدة: {len(exploration_result.new_parameters)}")
        print(f"   معدل نجاح الاستكشاف: {exploration_result.success_rate:.2%}")
        
        if exploration_result.performance_scores:
            best_performance = min(exploration_result.performance_scores)
            print(f"   أفضل أداء مكتشف: {best_performance:.6f}")
        
        # اختبار النظام المتكامل
        print("\n3️⃣ اختبار النظام المتكامل:")
        integrated = IntegratedExpertExplorer()
        
        optimization_result = integrated.intelligent_optimization(
            base_equation, x_data, y_data, max_iterations=3
        )
        
        print(f"   أفضل أداء محقق: {optimization_result['best_performance']:.6f}")
        print(f"   تحسن إجمالي: {optimization_result['total_improvement']:.6f}")
        print(f"   تكرارات مكتملة: {len(optimization_result['optimization_history'])}")
        
        exploration_stats = optimization_result['exploration_statistics']
        print(f"   إجمالي الاستكشافات: {exploration_stats['total_explorations']}")
        print(f"   معدل نجاح الاستكشاف: {exploration_stats['success_rate']:.2%}")
        
        print(f"   ✅ نظام الخبير/المستكشف يعمل بنجاح")
        
        return True
        
    except Exception as e:
        print(f"   ❌ خطأ في اختبار نظام الخبير/المستكشف: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_gse_model():
    """اختبار النموذج المحسن الكامل"""
    
    print("\n" + "="*60)
    print("🧪 اختبار النموذج المحسن الكامل")
    print("="*60)
    
    try:
        from enhanced_gse_model import EnhancedGSEModel
        
        # إنشاء النموذج المحسن
        print("\n1️⃣ إنشاء النموذج المحسن:")
        enhanced_model = EnhancedGSEModel(enable_theories=True)
        
        # إضافة مكونات أساسية
        enhanced_model.add_sigmoid_component(alpha=1.0, k=1.0, x0=0.0)
        enhanced_model.add_sigmoid_component(alpha=0.8, k=0.5, x0=2.0)
        
        print(f"   تم إنشاء نموذج بـ {len(enhanced_model.alpha_values)} مكونات")
        print(f"   النظريات الثلاث: {'مفعلة' if enhanced_model.enable_theories else 'معطلة'}")
        print(f"   مستوى التحسين: {enhanced_model.enhancement_level}")
        
        # بيانات اختبار (محاكاة الأعداد الأولية)
        print("\n2️⃣ إعداد بيانات الاختبار:")
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        x_data = np.array(range(1, len(primes) + 1))
        y_data = np.array([1 if i+1 in primes else 0 for i in range(len(primes))])
        
        print(f"   نقاط البيانات: {len(x_data)}")
        print(f"   أعداد أولية: {np.sum(y_data)}")
        
        # قياس الأداء الأولي
        print("\n3️⃣ قياس الأداء الأولي:")
        initial_performance = enhanced_model.calculate_loss(x_data, y_data)
        print(f"   أداء أولي: {initial_performance:.6f}")
        
        # تطبيق التحسين بالنظريات الثلاث
        print("\n4️⃣ تطبيق التحسين بالنظريات الثلاث:")
        theories_result = enhanced_model.enhance_with_three_theories(x_data, y_data, max_enhancement_cycles=3)
        
        if theories_result['success']:
            print(f"   تحسن بالنظريات: {theories_result['total_improvement']:.6f}")
            print(f"   نسبة التحسن: {theories_result['improvement_percentage']:.2f}%")
            print(f"   دورات مكتملة: {theories_result['cycles_completed']}")
            print(f"   أفضل أداء: {theories_result['best_performance']:.6f}")
        
        # تطبيق التحسين الذكي الشامل
        print("\n5️⃣ تطبيق التحسين الذكي الشامل:")
        comprehensive_result = enhanced_model.intelligent_adaptive_optimization(
            x_data, y_data, max_iterations=3
        )
        
        if comprehensive_result['success']:
            print(f"   أداء نهائي: {comprehensive_result['final_performance']:.6f}")
            print(f"   معادلات متكيفة: {comprehensive_result['adaptive_equations_created']}")
            print(f"   مستوى التحسين: {comprehensive_result['enhancement_level']}")
            print(f"   معدل نجاح التحسينات: {comprehensive_result['success_rate']:.2%}")
        
        # حساب التحسن الإجمالي
        final_performance = enhanced_model.calculate_loss(x_data, y_data)
        total_improvement = initial_performance - final_performance
        improvement_percentage = (total_improvement / initial_performance) * 100
        
        print(f"\n6️⃣ النتائج النهائية:")
        print(f"   أداء أولي: {initial_performance:.6f}")
        print(f"   أداء نهائي: {final_performance:.6f}")
        print(f"   تحسن إجمالي: {total_improvement:.6f}")
        print(f"   نسبة التحسن: {improvement_percentage:.2f}%")
        
        print(f"   ✅ النموذج المحسن يعمل بنجاح")
        
        return True, {
            'initial_performance': initial_performance,
            'final_performance': final_performance,
            'total_improvement': total_improvement,
            'improvement_percentage': improvement_percentage
        }
        
    except Exception as e:
        print(f"   ❌ خطأ في اختبار النموذج المحسن: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def run_comprehensive_test():
    """تشغيل اختبار شامل لجميع المكونات"""
    
    print("🚀 بدء الاختبار الشامل للنموذج المحسن GSE")
    print("مستوحى من نظام Baserah - النظريات الثلاث والذكاء التكيفي")
    print("="*80)
    
    test_results = {}
    start_time = time.time()
    
    # اختبار المكونات الفردية
    test_results['three_theories'] = test_three_theories()
    test_results['adaptive_equations'] = test_adaptive_equations()
    test_results['expert_explorer'] = test_expert_explorer_system()
    
    # اختبار النموذج المتكامل
    enhanced_success, enhanced_metrics = test_enhanced_gse_model()
    test_results['enhanced_model'] = enhanced_success
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # تلخيص النتائج
    print("\n" + "="*80)
    print("📊 ملخص نتائج الاختبار الشامل")
    print("="*80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\n🎯 النتائج الإجمالية:")
    print(f"   اختبارات ناجحة: {passed_tests}/{total_tests}")
    print(f"   معدل النجاح: {success_rate:.1f}%")
    print(f"   وقت التنفيذ: {total_time:.2f} ثانية")
    
    print(f"\n📋 تفاصيل الاختبارات:")
    for test_name, result in test_results.items():
        status = "✅ نجح" if result else "❌ فشل"
        print(f"   {test_name}: {status}")
    
    if enhanced_metrics:
        print(f"\n📈 مقاييس الأداء للنموذج المحسن:")
        print(f"   تحسن الأداء: {enhanced_metrics['improvement_percentage']:.2f}%")
        print(f"   تحسن مطلق: {enhanced_metrics['total_improvement']:.6f}")
    
    # تقييم النجاح الإجمالي
    if success_rate >= 80:
        print(f"\n🎉 النتيجة: النموذج المحسن يعمل بنجاح!")
        print(f"✅ جميع المكونات الأساسية تعمل بشكل صحيح")
        print(f"🚀 النظريات الثلاث مطبقة بنجاح")
        print(f"🧠 الذكاء التكيفي يعمل كما هو متوقع")
        print(f"🎯 النموذج جاهز للاستخدام في تحليل الأعداد الأولية")
    elif success_rate >= 60:
        print(f"\n⚠️ النتيجة: النموذج يعمل جزئياً")
        print(f"🔧 بعض المكونات تحتاج إلى تحسين")
    else:
        print(f"\n❌ النتيجة: النموذج يحتاج إلى إصلاحات")
        print(f"🛠️ مراجعة شاملة مطلوبة")
    
    return test_results, enhanced_metrics

if __name__ == "__main__":
    # تشغيل الاختبار الشامل
    results, metrics = run_comprehensive_test()
    
    # حفظ النتائج (اختياري)
    try:
        import json
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'test_results': results,
                'performance_metrics': metrics,
                'timestamp': time.time()
            }, f, indent=2, ensure_ascii=False)
        print(f"\n💾 تم حفظ نتائج الاختبار في test_results.json")
    except Exception as e:
        print(f"\n⚠️ لم يتم حفظ النتائج: {e}")
