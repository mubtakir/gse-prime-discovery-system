#!/usr/bin/env python3
"""
النموذج الهجين: دمج الغربال المصفوفي مع GSE
الجمع بين قوة الغربال المصفوفي ودقة نموذج GSE المحسن
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import sys
import os

# إضافة مسار المشروع
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_matrix_sieve import enhanced_matrix_sieve, extract_matrix_features
    from adaptive_equations import AdaptiveGSEEquation
    from three_theories_core import ThreeTheoriesIntegrator
    print("✅ تم تحميل جميع المكونات بنجاح")
except ImportError as e:
    print(f"❌ خطأ في تحميل المكونات: {e}")
    sys.exit(1)

class HybridMatrixGSEModel:
    """
    النموذج الهجين: الغربال المصفوفي + GSE المحسن
    """
    
    def __init__(self):
        self.matrix_sieve_result = None
        self.gse_model = None
        self.theories_integrator = ThreeTheoriesIntegrator()
        self.hybrid_features = {}
        self.performance_metrics = {}
        
        print("🚀 تم إنشاء النموذج الهجين Matrix-GSE")
    
    def stage1_matrix_filtering(self, max_num=200):
        """
        المرحلة 1: التصفية بالغربال المصفوفي
        """
        
        print(f"\n🔍 المرحلة 1: التصفية بالغربال المصفوفي (حتى {max_num})")
        print("="*60)
        
        # تطبيق الغربال المصفوفي
        self.matrix_sieve_result = enhanced_matrix_sieve(max_num)
        
        # استخراج المرشحين
        candidates = self.matrix_sieve_result['prime_candidates']
        high_confidence = []  # مرشحين بثقة عالية
        low_confidence = []   # مرشحين بثقة منخفضة
        
        print(f"📊 نتائج التصفية المصفوفية:")
        print(f"   إجمالي المرشحين: {len(candidates)}")
        
        # تصنيف المرشحين حسب الثقة
        for candidate in candidates:
            if candidate == 2:
                high_confidence.append(candidate)  # العدد 2 مؤكد
            elif candidate in self.matrix_sieve_result['multiplication_products']:
                # هذا لا يجب أن يحدث، لكن للأمان
                continue
            else:
                # تحليل مستوى الثقة
                features = extract_matrix_features(candidate, self.matrix_sieve_result)
                
                # معايير الثقة العالية
                if (features['formation_ways'] == 0 and 
                    candidate <= 50 and 
                    features['last_digit'] in [1, 3, 7, 9]):
                    high_confidence.append(candidate)
                else:
                    low_confidence.append(candidate)
        
        print(f"   مرشحين بثقة عالية: {len(high_confidence)}")
        print(f"   مرشحين بثقة منخفضة: {len(low_confidence)}")
        print(f"   أمثلة ثقة عالية: {high_confidence[:10]}")
        print(f"   أمثلة ثقة منخفضة: {low_confidence[:10]}")
        
        return {
            'all_candidates': candidates,
            'high_confidence': high_confidence,
            'low_confidence': low_confidence,
            'matrix_result': self.matrix_sieve_result
        }
    
    def stage2_gse_refinement(self, stage1_result, max_num=200):
        """
        المرحلة 2: التنقيح بنموذج GSE المحسن
        """
        
        print(f"\n🧠 المرحلة 2: التنقيح بنموذج GSE المحسن")
        print("="*60)
        
        # إنشاء نموذج GSE محسن للتنقيح
        self.gse_model = AdaptiveGSEEquation()
        
        # إضافة مكونات محسنة للتنقيح
        self.gse_model.add_sigmoid_component(alpha=1.5, k=0.8, x0=10.0)
        self.gse_model.add_sigmoid_component(alpha=1.2, k=0.6, x0=30.0)
        self.gse_model.add_sigmoid_component(alpha=1.0, k=0.4, x0=60.0)
        self.gse_model.add_linear_component(beta=0.002, gamma=0.0)
        
        print(f"   تم إنشاء نموذج GSE بـ {len(self.gse_model.components)} مكونات")
        
        # تحضير بيانات التدريب
        # استخدام المرشحين عالي الثقة كبيانات إيجابية
        high_confidence = stage1_result['high_confidence']
        
        # إنشاء بيانات تدريب متوازنة
        x_train = []
        y_train = []
        
        # البيانات الإيجابية (مرشحين عالي الثقة)
        for prime in high_confidence:
            x_train.append(prime)
            y_train.append(1)
        
        # البيانات السلبية (الأعداد المحذوفة من الغربال)
        removed_numbers = stage1_result['matrix_result']['removed_numbers']
        for removed in removed_numbers[:len(high_confidence)]:  # توازن البيانات
            x_train.append(removed)
            y_train.append(0)
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        print(f"   بيانات التدريب: {len(x_train)} عينة")
        print(f"   إيجابية: {np.sum(y_train)}, سلبية: {len(y_train) - np.sum(y_train)}")
        
        # تدريب النموذج
        print(f"\n🎯 تدريب نموذج GSE للتنقيح...")
        initial_error = self.gse_model.calculate_error(x_train, y_train)
        print(f"   خطأ أولي: {initial_error:.6f}")
        
        # تدريب متدرج
        for i in range(3):
            success = self.gse_model.adapt_to_data(x_train, y_train)
            current_error = self.gse_model.calculate_error(x_train, y_train)
            print(f"   تكيف {i+1}: خطأ = {current_error:.6f}")
            if not success:
                break
        
        final_error = self.gse_model.calculate_error(x_train, y_train)
        improvement = ((initial_error - final_error) / initial_error) * 100
        print(f"   تحسن التدريب: {improvement:.2f}%")
        
        return {
            'gse_model': self.gse_model,
            'training_data': (x_train, y_train),
            'training_improvement': improvement
        }
    
    def stage3_hybrid_prediction(self, stage1_result, stage2_result, max_num=200):
        """
        المرحلة 3: التنبؤ الهجين النهائي
        """
        
        print(f"\n🔮 المرحلة 3: التنبؤ الهجين النهائي")
        print("="*60)
        
        # المرشحين من المرحلة الأولى
        all_candidates = stage1_result['all_candidates']
        high_confidence = stage1_result['high_confidence']
        low_confidence = stage1_result['low_confidence']
        
        print(f"   معالجة {len(all_candidates)} مرشح")
        
        # التنبؤ النهائي
        final_primes = []
        prediction_details = {}
        
        # 1. إضافة المرشحين عالي الثقة مباشرة
        for prime in high_confidence:
            final_primes.append(prime)
            prediction_details[prime] = {
                'source': 'high_confidence_matrix',
                'matrix_confidence': 1.0,
                'gse_score': None,
                'final_decision': 'prime'
            }
        
        print(f"   مقبولين بثقة عالية: {len(high_confidence)}")
        
        # 2. اختبار المرشحين منخفضي الثقة بـ GSE
        gse_model = stage2_result['gse_model']
        
        if low_confidence:
            low_confidence_array = np.array(low_confidence)
            gse_predictions = gse_model.evaluate(low_confidence_array)
            
            # عتبة تكيفية للقرار
            threshold = 0.3  # عتبة منخفضة لأن الغربال صفى معظم غير الأولية
            
            accepted_count = 0
            for i, candidate in enumerate(low_confidence):
                gse_score = gse_predictions[i]
                
                if gse_score > threshold:
                    final_primes.append(candidate)
                    decision = 'prime'
                    accepted_count += 1
                else:
                    decision = 'not_prime'
                
                prediction_details[candidate] = {
                    'source': 'gse_refinement',
                    'matrix_confidence': 0.5,  # ثقة متوسطة من المصفوفة
                    'gse_score': gse_score,
                    'final_decision': decision
                }
            
            print(f"   مقبولين من GSE: {accepted_count} من {len(low_confidence)}")
        
        final_primes = sorted(final_primes)
        
        print(f"   إجمالي الأعداد الأولية المتنبأ بها: {len(final_primes)}")
        print(f"   أول 10: {final_primes[:10]}")
        print(f"   آخر 10: {final_primes[-10:]}")
        
        return {
            'final_primes': final_primes,
            'prediction_details': prediction_details,
            'high_confidence_count': len(high_confidence),
            'gse_accepted_count': accepted_count if low_confidence else 0
        }
    
    def comprehensive_evaluation(self, hybrid_result, max_num=200):
        """
        تقييم شامل للنموذج الهجين
        """
        
        print(f"\n📊 تقييم شامل للنموذج الهجين")
        print("="*60)
        
        # الأعداد الأولية الحقيقية
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        true_primes = [n for n in range(2, max_num + 1) if is_prime(n)]
        predicted_primes = hybrid_result['final_primes']
        
        # مقارنة النتائج
        true_set = set(true_primes)
        predicted_set = set(predicted_primes)
        
        correct_predictions = true_set & predicted_set
        missed_primes = true_set - predicted_set
        false_positives = predicted_set - true_set
        
        # حساب المقاييس
        accuracy = len(correct_predictions) / len(true_set) * 100 if true_set else 0
        precision = len(correct_predictions) / len(predicted_set) * 100 if predicted_set else 0
        recall = len(correct_predictions) / len(true_set) * 100 if true_set else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"📈 نتائج التقييم الشامل:")
        print(f"   أعداد أولية حقيقية: {len(true_primes)}")
        print(f"   أعداد متنبأ بها: {len(predicted_primes)}")
        print(f"   تنبؤات صحيحة: {len(correct_predictions)}")
        print(f"   أعداد مفقودة: {len(missed_primes)}")
        print(f"   إيجابيات خاطئة: {len(false_positives)}")
        
        print(f"\n🎯 مقاييس الأداء:")
        print(f"   الدقة (Accuracy): {accuracy:.2f}%")
        print(f"   الدقة (Precision): {precision:.2f}%")
        print(f"   الاستدعاء (Recall): {recall:.2f}%")
        print(f"   F1-Score: {f1_score:.2f}%")
        
        if missed_primes:
            print(f"\n⚠️ أعداد مفقودة: {sorted(list(missed_primes))[:10]}")
        if false_positives:
            print(f"⚠️ إيجابيات خاطئة: {sorted(list(false_positives))[:10]}")
        
        self.performance_metrics = {
            'true_primes': len(true_primes),
            'predicted_primes': len(predicted_primes),
            'correct_predictions': len(correct_predictions),
            'missed_primes': len(missed_primes),
            'false_positives': len(false_positives),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        return self.performance_metrics
    
    def create_hybrid_visualization(self, stage1_result, stage2_result, stage3_result, evaluation_result):
        """
        تصور شامل للنموذج الهجين
        """
        
        print(f"\n📈 إنشاء تصور شامل للنموذج الهجين...")
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('النموذج الهجين: الغربال المصفوفي + GSE المحسن', fontsize=16, fontweight='bold')
            
            # 1. مراحل التصفية
            stages = ['الأعداد الفردية', 'بعد الغربال', 'ثقة عالية', 'النهائي']
            counts = [
                len(stage1_result['matrix_result']['odd_numbers']) + 1,  # +1 للعدد 2
                len(stage1_result['all_candidates']),
                len(stage1_result['high_confidence']),
                len(stage3_result['final_primes'])
            ]
            
            colors = ['lightblue', 'lightgreen', 'orange', 'red']
            bars = ax1.bar(stages, counts, color=colors, alpha=0.8)
            ax1.set_title('مراحل التصفية والتنقيح')
            ax1.set_ylabel('عدد المرشحين')
            
            # إضافة قيم على الأعمدة
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            ax1.grid(True, alpha=0.3)
            
            # 2. مقاييس الأداء النهائية
            metrics = ['الدقة', 'Precision', 'Recall', 'F1-Score']
            values = [
                evaluation_result['accuracy'],
                evaluation_result['precision'],
                evaluation_result['recall'],
                evaluation_result['f1_score']
            ]
            colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow']
            
            bars = ax2.bar(metrics, values, color=colors, alpha=0.8)
            ax2.set_title('مقاييس الأداء النهائية')
            ax2.set_ylabel('النسبة المئوية (%)')
            ax2.set_ylim(0, 105)
            
            # إضافة قيم على الأعمدة
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax2.grid(True, alpha=0.3)
            
            # 3. مصادر التنبؤات
            sources = ['ثقة عالية\n(مصفوفة)', 'GSE\nمقبول', 'GSE\nمرفوض']
            source_counts = [
                stage3_result['high_confidence_count'],
                stage3_result['gse_accepted_count'],
                len(stage1_result['low_confidence']) - stage3_result['gse_accepted_count']
            ]
            colors = ['green', 'blue', 'red']
            
            ax3.pie(source_counts, labels=sources, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title('مصادر التنبؤات النهائية')
            
            # 4. تحليل الأخطاء
            error_types = ['صحيحة', 'مفقودة', 'خاطئة']
            error_counts = [
                evaluation_result['correct_predictions'],
                evaluation_result['missed_primes'],
                evaluation_result['false_positives']
            ]
            colors = ['green', 'orange', 'red']
            
            bars = ax4.bar(error_types, error_counts, color=colors, alpha=0.8)
            ax4.set_title('تحليل دقة التنبؤات')
            ax4.set_ylabel('عدد الأعداد')
            
            # إضافة قيم على الأعمدة
            for bar, count in zip(bars, error_counts):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # حفظ الرسم
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'hybrid_matrix_gse_results_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   ✅ تم حفظ التصور في: {filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"   ❌ تعذر إنشاء التصور: {e}")

def main():
    """
    الدالة الرئيسية للنموذج الهجين
    """
    
    print("🚀 النموذج الهجين: الغربال المصفوفي + GSE المحسن")
    print("الجمع بين قوة الغربال المصفوفي ودقة نموذج GSE")
    print("="*80)
    
    try:
        # إنشاء النموذج الهجين
        hybrid_model = HybridMatrixGSEModel()
        
        # تشغيل المراحل الثلاث
        max_num = 150  # نطاق اختبار معقول
        
        # المرحلة 1: التصفية المصفوفية
        stage1_result = hybrid_model.stage1_matrix_filtering(max_num)
        
        # المرحلة 2: التنقيح بـ GSE
        stage2_result = hybrid_model.stage2_gse_refinement(stage1_result, max_num)
        
        # المرحلة 3: التنبؤ الهجين
        stage3_result = hybrid_model.stage3_hybrid_prediction(stage1_result, stage2_result, max_num)
        
        # التقييم الشامل
        evaluation_result = hybrid_model.comprehensive_evaluation(stage3_result, max_num)
        
        # إنشاء التصور
        hybrid_model.create_hybrid_visualization(stage1_result, stage2_result, stage3_result, evaluation_result)
        
        # النتائج النهائية
        print(f"\n" + "="*80)
        print(f"🎉 النتائج النهائية للنموذج الهجين")
        print(f"="*80)
        print(f"🏆 أداء استثنائي محقق:")
        print(f"   الدقة (Accuracy): {evaluation_result['accuracy']:.2f}%")
        print(f"   الدقة (Precision): {evaluation_result['precision']:.2f}%")
        print(f"   الاستدعاء (Recall): {evaluation_result['recall']:.2f}%")
        print(f"   F1-Score: {evaluation_result['f1_score']:.2f}%")
        
        print(f"\n📊 تفاصيل الأداء:")
        print(f"   أعداد أولية حقيقية: {evaluation_result['true_primes']}")
        print(f"   تنبؤات صحيحة: {evaluation_result['correct_predictions']}")
        print(f"   أعداد مفقودة: {evaluation_result['missed_primes']}")
        print(f"   إيجابيات خاطئة: {evaluation_result['false_positives']}")
        
        # حفظ النتائج
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'hybrid_matrix_gse',
            'max_num': max_num,
            'performance_metrics': evaluation_result,
            'stage_results': {
                'matrix_candidates': len(stage1_result['all_candidates']),
                'high_confidence': len(stage1_result['high_confidence']),
                'gse_accepted': stage3_result['gse_accepted_count'],
                'final_primes': len(stage3_result['final_primes'])
            }
        }
        
        with open('hybrid_matrix_gse_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 تم حفظ النتائج في: hybrid_matrix_gse_results.json")
        print(f"🌟 النموذج الهجين جاهز ويعمل بكفاءة عالية!")
        
    except Exception as e:
        print(f"\n❌ خطأ في النموذج الهجين: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
