"""
الملف الرئيسي المتقدم لنظام GSE
تشغيل شامل للنموذج مع جميع الميزات المتقدمة
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# إضافة المسار الحالي للاستيراد
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gse_advanced_model import AdvancedGSEModel
from number_theory_utils import NumberTheoryUtils, PrimeAnalyzer
from visualization_tools import GSEVisualizer

def main():
    """الدالة الرئيسية لتشغيل النظام المتقدم"""
    
    print("🚀 مرحباً بك في نظام GSE المتقدم!")
    print("=" * 50)
    
    # 1. إعداد البيانات
    print("\n📊 إعداد البيانات...")
    x_train, y_train = NumberTheoryUtils.generate_prime_data(2, 100)
    x_test, y_test = NumberTheoryUtils.generate_prime_data(101, 200)
    
    print(f"   بيانات التدريب: {len(x_train)} نقطة")
    print(f"   بيانات الاختبار: {len(x_test)} نقطة")
    print(f"   عدد الأعداد الأولية في التدريب: {np.sum(y_train)}")
    print(f"   عدد الأعداد الأولية في الاختبار: {np.sum(y_test)}")
    
    # 2. إنشاء النموذج المتقدم
    print("\n🧠 إنشاء النموذج المتقدم...")
    model = AdvancedGSEModel()
    
    # إضافة مكونات سيجمويد متعددة
    model.add_sigmoid(alpha=complex(1.0, 0.1), n=2.0, z=complex(1.2, 0.3), x0=5.0)
    model.add_sigmoid(alpha=complex(0.8, -0.2), n=1.5, z=complex(0.9, -0.1), x0=10.0)
    model.add_sigmoid(alpha=complex(0.6, 0.3), n=3.0, z=complex(1.1, 0.2), x0=15.0)
    
    print(f"   تم إضافة {len(model.sigmoid_components)} مكون سيجمويد")
    
    # 3. التدريب المتقدم
    print("\n🎯 بدء التدريب المتقدم...")
    try:
        result = model.optimize_advanced(
            x_train, y_train, 
            method='differential_evolution',
            max_iter=100,
            verbose=True
        )
        print(f"✅ التدريب مكتمل بنجاح!")
    except Exception as e:
        print(f"❌ خطأ في التدريب: {e}")
        return
    
    # 4. التقييم
    print("\n📈 تقييم الأداء...")
    
    # تقييم على بيانات التدريب
    train_predictions = model.evaluate(x_train)
    train_mse = np.mean((y_train - train_predictions) ** 2)
    
    # تقييم على بيانات الاختبار
    test_predictions = model.evaluate(x_test)
    test_mse = np.mean((y_test - test_predictions) ** 2)
    
    print(f"   خطأ التدريب (MSE): {train_mse:.6f}")
    print(f"   خطأ الاختبار (MSE): {test_mse:.6f}")
    
    # 5. توقع الأعداد الأولية
    print("\n🔍 توقع الأعداد الأولية...")
    predicted_primes, predictions, binary_preds = model.predict_primes((201, 250))
    
    print(f"   الأعداد المتوقعة كأولية في النطاق 201-250:")
    print(f"   {predicted_primes}")
    
    # التحقق من الدقة
    actual_primes = [n for n in range(201, 251) if NumberTheoryUtils.is_prime(n)]
    print(f"   الأعداد الأولية الحقيقية:")
    print(f"   {actual_primes}")
    
    # حساب الدقة
    correct_predictions = len(set(predicted_primes) & set(actual_primes))
    total_actual = len(actual_primes)
    total_predicted = len(predicted_primes)
    
    precision = correct_predictions / total_predicted if total_predicted > 0 else 0
    recall = correct_predictions / total_actual if total_actual > 0 else 0
    
    print(f"   الدقة (Precision): {precision:.4f}")
    print(f"   الاستدعاء (Recall): {recall:.4f}")
    
    # 6. التصور المتقدم
    print("\n🎨 إنشاء التصورات المتقدمة...")
    visualizer = GSEVisualizer(model)
    
    try:
        # رسم توقعات النموذج
        visualizer.plot_model_prediction((2, 150), "أداء نموذج GSE المتقدم")
        
        # رسم مكونات السيجمويد
        visualizer.plot_sigmoid_components((-5, 25))
        
        # تحليل التدريب
        visualizer.plot_training_analysis()
        
        # تحليل شامل للتوقعات
        visualizer.plot_prime_prediction_analysis((2, 200))
        
    except Exception as e:
        print(f"⚠️ خطأ في التصور: {e}")
    
    # 7. تحليل نظرية الأعداد
    print("\n🔬 تحليل نظرية الأعداد...")
    try:
        NumberTheoryUtils.plot_number_theory_functions(100)
        
        # تحليل الفجوات بين الأعداد الأولية
        analyzer = PrimeAnalyzer()
        gap_analysis, gaps = analyzer.analyze_prime_gaps(500)
        
        print(f"   تحليل الفجوات بين الأعداد الأولية:")
        print(f"   متوسط الفجوة: {gap_analysis['average_gap']:.2f}")
        print(f"   أكبر فجوة: {gap_analysis['max_gap']}")
        print(f"   أصغر فجوة: {gap_analysis['min_gap']}")
        
        # الأعداد الأولية التوأم
        twin_primes = analyzer.twin_primes_analysis(200)
        print(f"   عدد الأعداد الأولية التوأم حتى 200: {len(twin_primes)}")
        print(f"   أمثلة: {twin_primes[:5]}")
        
    except Exception as e:
        print(f"⚠️ خطأ في تحليل نظرية الأعداد: {e}")
    
    # 8. ملخص النموذج
    print("\n📋 ملخص النموذج النهائي:")
    summary = model.get_model_summary()
    
    print(f"   عدد مكونات السيجمويد: {summary['num_sigmoid_components']}")
    print(f"   حالة التدريب: {'مدرب' if summary['trained'] else 'غير مدرب'}")
    print(f"   عدد تكرارات التدريب: {summary['training_iterations']}")
    if summary['final_loss']:
        print(f"   الخطأ النهائي: {summary['final_loss']:.8f}")
    
    print("\n" + "=" * 50)
    print("🎉 انتهى تشغيل النظام بنجاح!")
    
    return model, visualizer

def demo_interactive():
    """عرض تفاعلي للنظام"""
    print("\n🎮 العرض التفاعلي")
    print("-" * 30)
    
    while True:
        print("\nاختر العملية:")
        print("1. تشغيل النظام الكامل")
        print("2. اختبار نموذج بسيط")
        print("3. تحليل نظرية الأعداد فقط")
        print("4. إنشاء تصورات")
        print("5. خروج")
        
        choice = input("اختيارك (1-5): ").strip()
        
        if choice == '1':
            main()
        elif choice == '2':
            test_simple_model()
        elif choice == '3':
            analyze_number_theory()
        elif choice == '4':
            create_visualizations()
        elif choice == '5':
            print("👋 وداعاً!")
            break
        else:
            print("❌ اختيار غير صحيح")

def test_simple_model():
    """اختبار نموذج بسيط"""
    print("\n🧪 اختبار نموذج بسيط...")
    
    # إنشاء نموذج بسيط
    model = AdvancedGSEModel()
    model.add_sigmoid(alpha=1.0, n=2.0, z=complex(1.0, 0.0), x0=5.0)
    
    # بيانات بسيطة
    x_data, y_data = NumberTheoryUtils.generate_prime_data(2, 50)
    
    # تدريب سريع
    model.optimize_advanced(x_data, y_data, max_iter=50, verbose=False)
    
    # تقييم
    predictions = model.evaluate(x_data)
    mse = np.mean((y_data - predictions) ** 2)
    
    print(f"✅ النموذج البسيط: MSE = {mse:.6f}")
    
    # رسم بسيط
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'ro', label='حقيقي', markersize=4)
    plt.plot(x_data, predictions, 'b-', label='متوقع', linewidth=2)
    plt.title('اختبار النموذج البسيط')
    plt.xlabel('العدد')
    plt.ylabel('الاحتمالية')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def analyze_number_theory():
    """تحليل نظرية الأعداد فقط"""
    print("\n🔬 تحليل نظرية الأعداد...")
    
    try:
        # رسم الدوال الأساسية
        NumberTheoryUtils.plot_number_theory_functions(150)
        
        # تحليل متقدم
        analyzer = PrimeAnalyzer()
        
        # تحليل الكثافة
        positions, densities = analyzer.prime_density_analysis(1000, 50)
        
        plt.figure(figsize=(12, 6))
        plt.plot(positions, densities, 'b-', linewidth=2)
        plt.title('كثافة الأعداد الأولية')
        plt.xlabel('الموضع')
        plt.ylabel('الكثافة')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"✅ تم تحليل نظرية الأعداد بنجاح")
        
    except Exception as e:
        print(f"❌ خطأ في التحليل: {e}")

def create_visualizations():
    """إنشاء تصورات فقط"""
    print("\n🎨 إنشاء التصورات...")
    
    # إنشاء نموذج وهمي للتصور
    model = AdvancedGSEModel()
    model.add_sigmoid(alpha=complex(1.0, 0.2), n=2.0, z=complex(1.1, 0.3), x0=5.0)
    model.add_sigmoid(alpha=complex(0.8, -0.1), n=1.5, z=complex(0.9, -0.2), x0=10.0)
    
    # إضافة تاريخ وهمي للتدريب
    model.training_history = [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15, 0.12]
    
    visualizer = GSEVisualizer(model)
    
    try:
        # رسم مكونات السيجمويد
        visualizer.plot_sigmoid_components((-10, 20))
        
        # رسم ثلاثي الأبعاد
        visualizer.plot_3d_surface((-3, 3), 30)
        
        print("✅ تم إنشاء التصورات بنجاح")
        
    except Exception as e:
        print(f"❌ خطأ في التصور: {e}")

if __name__ == "__main__":
    try:
        # تشغيل النظام الرئيسي
        main()
        
        # تشغيل العرض التفاعلي (اختياري)
        # demo_interactive()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ تم إيقاف البرنامج بواسطة المستخدم")
    except Exception as e:
        print(f"\n❌ خطأ عام في النظام: {e}")
        import traceback
        traceback.print_exc()
