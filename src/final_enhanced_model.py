#!/usr/bin/env python3
"""
النموذج المحسن النهائي المصحح
يجمع بين النظريات الثلاث والمعاملات المحسنة
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
import json

# إضافة مسار المشروع
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from adaptive_equations import AdaptiveGSEEquation, AdaptationDirection
    from three_theories_core import ThreeTheoriesIntegrator
    from expert_explorer_system import GSEExpertSystem, GSEExplorerSystem, ExplorerMode
    print("✅ تم تحميل جميع المكونات بنجاح")
except ImportError as e:
    print(f"❌ خطأ في تحميل المكونات: {e}")
    sys.exit(1)

def generate_comprehensive_prime_data(max_num=150):
    """توليد بيانات شاملة للأعداد الأولية"""
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    numbers = list(range(2, max_num + 1))
    prime_indicators = [1 if is_prime(n) else 0 for n in numbers]
    primes_list = [n for n in numbers if is_prime(n)]
    
    return np.array(numbers), np.array(prime_indicators), np.array(primes_list)

class FinalEnhancedGSEModel:
    """النموذج المحسن النهائي المصحح"""
    
    def __init__(self):
        self.adaptive_equation = None
        self.theories_integrator = ThreeTheoriesIntegrator()
        self.expert_system = GSEExpertSystem()
        self.explorer_system = GSEExplorerSystem()
        
        # إعدادات محسنة
        self.optimal_threshold = 0.3
        self.training_history = []
        self.performance_metrics = {}
        
        print("🚀 تم إنشاء النموذج المحسن النهائي")
    
    def create_optimized_model(self):
        """إنشاء نموذج محسن بمعاملات مثلى"""
        
        print("\n🔧 إنشاء النموذج بمعاملات محسنة...")
        
        self.adaptive_equation = AdaptiveGSEEquation()
        
        # معاملات محسنة بناءً على التجارب السابقة
        self.adaptive_equation.add_sigmoid_component(alpha=2.5, k=1.2, x0=8.0)    # للأعداد الصغيرة
        self.adaptive_equation.add_sigmoid_component(alpha=2.0, k=1.0, x0=25.0)   # للأعداد المتوسطة
        self.adaptive_equation.add_sigmoid_component(alpha=1.5, k=0.8, x0=50.0)   # للأعداد الكبيرة
        self.adaptive_equation.add_sigmoid_component(alpha=1.0, k=0.6, x0=80.0)   # للأعداد الأكبر
        self.adaptive_equation.add_linear_component(beta=0.005, gamma=0.0)        # اتجاه عام
        
        print(f"   ✅ تم إنشاء {len(self.adaptive_equation.components)} مكونات محسنة")
        print(f"   📊 معاملات alpha: [2.5, 2.0, 1.5, 1.0]")
        print(f"   📊 معاملات k: [1.2, 1.0, 0.8, 0.6]")
        
        return self.adaptive_equation
    
    def intelligent_training(self, x_data, y_data, max_iterations=8):
        """تدريب ذكي متدرج"""
        
        print(f"\n🎯 بدء التدريب الذكي المتدرج...")
        
        if self.adaptive_equation is None:
            self.create_optimized_model()
        
        # قياس الأداء الأولي
        initial_error = self.adaptive_equation.calculate_error(x_data, y_data)
        print(f"   خطأ أولي: {initial_error:.6f}")
        
        training_log = []
        
        # مرحلة 1: تكيف أساسي
        print(f"\n   🔄 المرحلة 1: التكيف الأساسي")
        for i in range(3):
            success = self.adaptive_equation.adapt_to_data(
                x_data, y_data, AdaptationDirection.IMPROVE_ACCURACY
            )
            current_error = self.adaptive_equation.calculate_error(x_data, y_data)
            improvement = initial_error - current_error if i == 0 else training_log[-1]['error'] - current_error
            
            training_log.append({
                'iteration': i + 1,
                'phase': 'basic_adaptation',
                'error': current_error,
                'improvement': improvement,
                'success': success
            })
            
            print(f"      تكيف {i+1}: خطأ = {current_error:.6f}, تحسن = {improvement:.6f}")
            
            if not success:
                break
        
        # مرحلة 2: تطبيق النظريات الثلاث
        print(f"\n   🔬 المرحلة 2: تطبيق النظريات الثلاث")
        pre_theories_error = self.adaptive_equation.calculate_error(x_data, y_data)
        
        # تطبيق النظريات بحذر
        self._apply_balanced_theories()
        
        post_theories_error = self.adaptive_equation.calculate_error(x_data, y_data)
        theories_improvement = pre_theories_error - post_theories_error
        
        training_log.append({
            'iteration': 'theories',
            'phase': 'three_theories',
            'error': post_theories_error,
            'improvement': theories_improvement,
            'success': theories_improvement > 0
        })
        
        print(f"      النظريات الثلاث: خطأ = {post_theories_error:.6f}, تحسن = {theories_improvement:.6f}")
        
        # مرحلة 3: تكيف نهائي
        print(f"\n   ⚡ المرحلة 3: التكيف النهائي")
        for i in range(2):
            success = self.adaptive_equation.adapt_to_data(
                x_data, y_data, AdaptationDirection.BALANCE_BOTH
            )
            current_error = self.adaptive_equation.calculate_error(x_data, y_data)
            improvement = training_log[-1]['error'] - current_error
            
            training_log.append({
                'iteration': f'final_{i+1}',
                'phase': 'final_adaptation',
                'error': current_error,
                'improvement': improvement,
                'success': success
            })
            
            print(f"      تكيف نهائي {i+1}: خطأ = {current_error:.6f}, تحسن = {improvement:.6f}")
            
            if not success:
                break
        
        # حساب التحسن الإجمالي
        final_error = self.adaptive_equation.calculate_error(x_data, y_data)
        total_improvement = initial_error - final_error
        improvement_percentage = (total_improvement / initial_error) * 100
        
        self.training_history = training_log
        
        print(f"\n   🎉 انتهى التدريب:")
        print(f"      خطأ أولي: {initial_error:.6f}")
        print(f"      خطأ نهائي: {final_error:.6f}")
        print(f"      تحسن إجمالي: {total_improvement:.6f} ({improvement_percentage:.2f}%)")
        
        return training_log
    
    def _apply_balanced_theories(self):
        """تطبيق النظريات الثلاث بتوازن"""
        
        # تطبيق نظرية التوازن بحذر
        for component in self.adaptive_equation.components:
            if component['type'] == 'sigmoid':
                balance_factor = self.theories_integrator.zero_duality.calculate_balance_point(
                    abs(component['alpha']), 1.0
                )
                # تطبيق جزئي لتجنب التأثير المفرط
                component['alpha'] *= (1.0 + 0.1 * (balance_factor - 1.0))
        
        # تطبيق نظرية الفتائل بحذر
        enhanced_components = self.theories_integrator.filament_connection.apply_filament_enhancement(
            self.adaptive_equation.components
        )
        
        # دمج التحسينات جزئي<|im_start|>
        for i, enhanced in enumerate(enhanced_components):
            if i < len(self.adaptive_equation.components):
                original = self.adaptive_equation.components[i]
                if enhanced['type'] == 'sigmoid':
                    # تطبيق 30% من التحسين فقط
                    alpha_improvement = enhanced['alpha'] - original['alpha']
                    original['alpha'] += 0.3 * alpha_improvement
    
    def comprehensive_evaluation(self, x_data, y_data):
        """تقييم شامل للنموذج"""
        
        print(f"\n📊 تقييم شامل للنموذج المحسن...")
        
        if self.adaptive_equation is None:
            print("❌ النموذج غير مدرب")
            return None
        
        # التنبؤات الأساسية
        y_pred = self.adaptive_equation.evaluate(x_data)
        
        # تقييم بعتبات مختلفة
        thresholds = [0.2, 0.3, 0.4, 0.5]
        results = {}
        
        for threshold in thresholds:
            predictions = (y_pred > threshold).astype(int)
            
            # حساب المقاييس
            accuracy = np.mean(predictions == y_data)
            true_positives = np.sum((predictions == 1) & (y_data == 1))
            predicted_positives = np.sum(predictions == 1)
            actual_positives = np.sum(y_data == 1)
            
            precision = true_positives / max(1, predicted_positives)
            recall = true_positives / max(1, actual_positives)
            f1_score = 2 * (precision * recall) / max(1e-10, precision + recall)
            
            results[threshold] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': true_positives,
                'predicted_positives': predicted_positives
            }
            
            print(f"   عتبة {threshold}: دقة={accuracy:.2%}, F1={f1_score:.2%}, Recall={recall:.2%}")
        
        # اختيار أفضل عتبة
        best_threshold = max(results.keys(), key=lambda t: results[t]['f1_score'])
        self.optimal_threshold = best_threshold
        
        print(f"\n   🏆 أفضل عتبة: {best_threshold} (F1-Score: {results[best_threshold]['f1_score']:.2%})")
        
        self.performance_metrics = results[best_threshold]
        self.performance_metrics['threshold'] = best_threshold
        self.performance_metrics['predictions'] = y_pred
        
        return results
    
    def predict_next_primes(self, known_primes, num_predictions=5):
        """التنبؤ بالأعداد الأولية التالية"""
        
        print(f"\n🔮 التنبؤ بالأعداد الأولية التالية...")
        
        if self.adaptive_equation is None:
            print("❌ النموذج غير مدرب")
            return []
        
        last_prime = known_primes[-1]
        print(f"   آخر عدد أولي معروف: {last_prime}")
        
        # البحث في النطاق التالي
        search_range = np.arange(last_prime + 1, last_prime + 100)
        predictions = self.adaptive_equation.evaluate(search_range)
        
        # العثور على أفضل المرشحين
        candidates = search_range[predictions > self.optimal_threshold]
        candidate_scores = predictions[predictions > self.optimal_threshold]
        
        # ترتيب حسب النتيجة
        sorted_indices = np.argsort(candidate_scores)[::-1]
        top_candidates = candidates[sorted_indices][:num_predictions]
        top_scores = candidate_scores[sorted_indices][:num_predictions]
        
        print(f"   🎯 أفضل {len(top_candidates)} مرشحين:")
        for i, (candidate, score) in enumerate(zip(top_candidates, top_scores), 1):
            print(f"      {i}. العدد {candidate}: نتيجة = {score:.4f}")
        
        return top_candidates, top_scores
    
    def create_comprehensive_visualization(self, x_data, y_data, results):
        """إنشاء تصور شامل للنتائج النهائية"""
        
        print(f"\n📈 إنشاء تصور شامل للنتائج النهائية...")
        
        try:
            fig = plt.figure(figsize=(20, 15))
            fig.suptitle('النموذج المحسن النهائي - النتائج الشاملة', fontsize=20, fontweight='bold')
            
            # 1. التنبؤات مع العتبة المثلى
            ax1 = plt.subplot(2, 3, 1)
            y_pred = self.performance_metrics['predictions']
            
            ax1.plot(x_data, y_pred, 'b-', label='تنبؤات النموذج', linewidth=2)
            ax1.scatter(x_data[y_data == 1], [1]*np.sum(y_data), color='red', s=50, 
                       label='أعداد أولية حقيقية', zorder=5)
            ax1.axhline(y=self.optimal_threshold, color='green', linestyle='--', 
                       label=f'العتبة المثلى ({self.optimal_threshold})', linewidth=2)
            ax1.set_title('التنبؤات مع العتبة المثلى')
            ax1.set_xlabel('العدد')
            ax1.set_ylabel('احتمالية كونه أولي')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. مقارنة العتبات
            ax2 = plt.subplot(2, 3, 2)
            thresholds = list(results.keys())
            f1_scores = [results[t]['f1_score'] for t in thresholds]
            recalls = [results[t]['recall'] for t in thresholds]
            precisions = [results[t]['precision'] for t in thresholds]
            
            ax2.plot(thresholds, f1_scores, 'g-o', label='F1-Score', linewidth=2, markersize=8)
            ax2.plot(thresholds, recalls, 'b-s', label='Recall', linewidth=2, markersize=8)
            ax2.plot(thresholds, precisions, 'r-^', label='Precision', linewidth=2, markersize=8)
            ax2.axvline(x=self.optimal_threshold, color='purple', linestyle='--', alpha=0.7)
            ax2.set_title('مقارنة العتبات')
            ax2.set_xlabel('العتبة')
            ax2.set_ylabel('النتيجة')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. تطور التدريب
            ax3 = plt.subplot(2, 3, 3)
            if self.training_history:
                iterations = []
                errors = []
                for i, entry in enumerate(self.training_history):
                    if isinstance(entry['iteration'], str):
                        iterations.append(i)
                    else:
                        iterations.append(entry['iteration'] - 1)
                    errors.append(entry['error'])
                
                ax3.plot(iterations, errors, 'purple', marker='o', linewidth=2, markersize=6)
                ax3.set_title('تطور التدريب')
                ax3.set_xlabel('مرحلة التدريب')
                ax3.set_ylabel('الخطأ')
                ax3.grid(True, alpha=0.3)
            
            # 4. توزيع التنبؤات
            ax4 = plt.subplot(2, 3, 4)
            ax4.hist(y_pred, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.axvline(x=self.optimal_threshold, color='red', linestyle='--', 
                       label=f'العتبة المثلى ({self.optimal_threshold})', linewidth=2)
            ax4.set_title('توزيع التنبؤات')
            ax4.set_xlabel('قيمة التنبؤ')
            ax4.set_ylabel('التكرار')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. مصفوفة الخلط
            ax5 = plt.subplot(2, 3, 5)
            predictions = (y_pred > self.optimal_threshold).astype(int)
            
            tp = np.sum((predictions == 1) & (y_data == 1))
            fp = np.sum((predictions == 1) & (y_data == 0))
            tn = np.sum((predictions == 0) & (y_data == 0))
            fn = np.sum((predictions == 0) & (y_data == 1))
            
            confusion_matrix = np.array([[tn, fp], [fn, tp]])
            im = ax5.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
            
            # إضافة النصوص
            for i in range(2):
                for j in range(2):
                    ax5.text(j, i, confusion_matrix[i, j], ha="center", va="center", 
                            color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black",
                            fontsize=16, fontweight='bold')
            
            ax5.set_title('مصفوفة الخلط')
            ax5.set_xlabel('متنبأ به')
            ax5.set_ylabel('حقيقي')
            ax5.set_xticks([0, 1])
            ax5.set_yticks([0, 1])
            ax5.set_xticklabels(['غير أولي', 'أولي'])
            ax5.set_yticklabels(['غير أولي', 'أولي'])
            
            # 6. ملخص الأداء النهائي
            ax6 = plt.subplot(2, 3, 6)
            
            metrics = ['الدقة', 'Precision', 'Recall', 'F1-Score']
            values = [
                self.performance_metrics['accuracy'],
                self.performance_metrics['precision'],
                self.performance_metrics['recall'],
                self.performance_metrics['f1_score']
            ]
            
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            bars = ax6.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
            
            # إضافة قيم على الأعمدة
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            ax6.set_title('ملخص الأداء النهائي')
            ax6.set_ylabel('النتيجة')
            ax6.set_ylim(0, 1)
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # حفظ الرسم
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'final_enhanced_model_results_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   ✅ تم حفظ التصور في: {filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"   ❌ تعذر إنشاء التصور: {e}")

def main():
    """تشغيل النموذج المحسن النهائي"""
    
    print("🚀 تشغيل النموذج المحسن النهائي المصحح")
    print("النموذج الأكثر تقدم<|im_start|> مع جميع التحسينات والإصلاحات")
    print("="*80)
    
    try:
        # إنشاء النموذج النهائي
        final_model = FinalEnhancedGSEModel()
        
        # تحضير البيانات الشاملة
        print(f"\n📊 تحضير البيانات الشاملة...")
        x_data, y_data, primes_list = generate_comprehensive_prime_data(120)
        
        print(f"   نطاق البيانات: 2-120")
        print(f"   إجمالي الأرقام: {len(x_data)}")
        print(f"   أعداد أولية: {np.sum(y_data)}")
        print(f"   آخر 5 أعداد أولية: {primes_list[-5:]}")
        
        # إنشاء النموذج المحسن
        final_model.create_optimized_model()
        
        # التدريب الذكي
        training_log = final_model.intelligent_training(x_data, y_data)
        
        # التقييم الشامل
        evaluation_results = final_model.comprehensive_evaluation(x_data, y_data)
        
        # التنبؤ بالأعداد التالية
        next_primes, scores = final_model.predict_next_primes(primes_list, num_predictions=5)
        
        # إنشاء التصور الشامل
        final_model.create_comprehensive_visualization(x_data, y_data, evaluation_results)
        
        # النتائج النهائية
        print("\n" + "="*80)
        print("🎉 النتائج النهائية للنموذج المحسن")
        print("="*80)
        
        best_metrics = final_model.performance_metrics
        print(f"\n🏆 أفضل أداء محقق:")
        print(f"   العتبة المثلى: {best_metrics['threshold']}")
        print(f"   الدقة العامة: {best_metrics['accuracy']:.2%}")
        print(f"   Precision: {best_metrics['precision']:.2%}")
        print(f"   Recall: {best_metrics['recall']:.2%}")
        print(f"   F1-Score: {best_metrics['f1_score']:.2%}")
        print(f"   تنبؤات صحيحة: {best_metrics['true_positives']}")
        print(f"   إجمالي متنبأ بها: {best_metrics['predicted_positives']}")
        
        print(f"\n🔮 التنبؤ بالأعداد الأولية التالية:")
        for i, (prime, score) in enumerate(zip(next_primes, scores), 1):
            print(f"   {i}. العدد {prime}: احتمالية {score:.4f}")
        
        # حفظ النتائج
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'final_enhanced_gse',
            'performance_metrics': {k: float(v) if isinstance(v, np.floating) else v 
                                  for k, v in best_metrics.items() if k != 'predictions'},
            'training_phases': len(training_log),
            'predicted_next_primes': next_primes.tolist() if len(next_primes) > 0 else [],
            'data_range': f"2-{max(x_data)}",
            'total_primes': int(np.sum(y_data))
        }
        
        with open('final_enhanced_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 تم حفظ النتائج في: final_enhanced_results.json")
        
        print(f"\n🌟 النموذج المحسن النهائي جاهز ويعمل بكفاءة عالية!")
        
    except Exception as e:
        print(f"\n❌ خطأ في تشغيل النموذج النهائي: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
