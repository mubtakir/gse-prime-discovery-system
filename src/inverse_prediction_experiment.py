"""
تجربة التنبؤ العكسي - اكتشاف الأعداد الأولية مباشرة
Inverse Prediction Experiment - Direct Prime Discovery
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# إضافة مسار المصدر
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gse_advanced_model import AdvancedGSEModel
from optimizer_advanced import GSEOptimizer
from target_functions import TargetFunctions
from number_theory_utils import NumberTheoryUtils

class InversePredictionExperiment:
    """تجربة التنبؤ العكسي لاكتشاف الأعداد الأولية"""
    
    def __init__(self):
        self.forward_model = None  # النموذج الأمامي: x → π(x)
        self.inverse_model = None  # النموذج العكسي: π(x) → x
        self.results = {}
        self.experiment_log = []
    
    def log_message(self, message):
        """تسجيل رسالة مع الوقت"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.experiment_log.append(log_entry)
    
    def train_forward_model(self, x_range=(2, 200)):
        """تدريب النموذج الأمامي: x → π(x)"""
        
        self.log_message("🔄 تدريب النموذج الأمامي x → π(x)")
        
        # إعداد البيانات
        x_data, pi_data = TargetFunctions.prime_counting_function(x_range)
        
        # إنشاء النموذج
        self.forward_model = AdvancedGSEModel()
        self.forward_model.add_sigmoid(alpha=complex(1.0, 0.0), n=2.0, z=complex(1.0, 0.0), x0=10.0)
        self.forward_model.add_sigmoid(alpha=complex(0.8, 0.0), n=1.5, z=complex(0.9, 0.0), x0=50.0)
        
        # التدريب
        optimizer = GSEOptimizer(self.forward_model)
        result = optimizer.optimize_differential_evolution(x_data, pi_data, max_iter=300, verbose=False)
        
        # التقييم
        y_pred = self.forward_model.evaluate(x_data)
        mse = np.mean((pi_data - y_pred) ** 2)
        r2 = 1 - (np.sum((pi_data - y_pred) ** 2) / np.sum((pi_data - np.mean(pi_data)) ** 2))
        
        self.log_message(f"✅ النموذج الأمامي: R² = {r2:.4f}, MSE = {mse:.6f}")
        
        return {
            'model': self.forward_model,
            'x_data': x_data,
            'y_true': pi_data,
            'y_pred': y_pred,
            'r2': r2,
            'mse': mse
        }
    
    def train_inverse_model(self, x_range=(2, 200)):
        """تدريب النموذج العكسي: π(x) → x"""
        
        self.log_message("🔄 تدريب النموذج العكسي π(x) → x")
        
        # إعداد البيانات العكسية
        x_data, pi_data = TargetFunctions.prime_counting_function(x_range)
        
        # في النموذج العكسي: المدخل هو π(x) والمخرج هو x
        inverse_x_data = pi_data  # المدخل: قيم π(x)
        inverse_y_data = x_data   # المخرج المطلوب: قيم x
        
        # إنشاء النموذج العكسي
        self.inverse_model = AdvancedGSEModel()
        self.inverse_model.add_sigmoid(alpha=complex(10.0, 0.0), n=1.2, z=complex(0.5, 0.0), x0=5.0)
        self.inverse_model.add_sigmoid(alpha=complex(8.0, 0.0), n=1.8, z=complex(0.3, 0.0), x0=15.0)
        self.inverse_model.add_sigmoid(alpha=complex(5.0, 0.0), n=1.5, z=complex(0.4, 0.0), x0=10.0)
        
        # التدريب
        optimizer = GSEOptimizer(self.inverse_model)
        result = optimizer.optimize_differential_evolution(inverse_x_data, inverse_y_data, max_iter=400, verbose=False)
        
        # التقييم
        x_pred = self.inverse_model.evaluate(inverse_x_data)
        mse = np.mean((inverse_y_data - x_pred) ** 2)
        r2 = 1 - (np.sum((inverse_y_data - x_pred) ** 2) / np.sum((inverse_y_data - np.mean(inverse_y_data)) ** 2))
        
        self.log_message(f"✅ النموذج العكسي: R² = {r2:.4f}, MSE = {mse:.6f}")
        
        return {
            'model': self.inverse_model,
            'x_data': inverse_x_data,  # قيم π(x)
            'y_true': inverse_y_data,  # قيم x الحقيقية
            'y_pred': x_pred,          # قيم x المتوقعة
            'r2': r2,
            'mse': mse
        }
    
    def prime_discovery_test(self, target_pi_values):
        """اختبار اكتشاف الأعداد الأولية باستخدام النموذج العكسي"""
        
        self.log_message("🔍 اختبار اكتشاف الأعداد الأولية...")
        
        if self.inverse_model is None:
            self.log_message("❌ النموذج العكسي غير مدرب!")
            return None
        
        discovered_positions = []
        actual_primes = []
        
        for pi_val in target_pi_values:
            # استخدام النموذج العكسي للعثور على x حيث π(x) = pi_val
            predicted_x = self.inverse_model.evaluate(np.array([pi_val]))[0]
            discovered_positions.append(predicted_x)
            
            # العثور على أقرب عدد صحيح
            nearest_int = int(round(predicted_x))
            
            # فحص ما إذا كان عدد أولي
            if NumberTheoryUtils.is_prime(nearest_int):
                actual_primes.append(nearest_int)
            
            self.log_message(f"   π({pi_val}) → x ≈ {predicted_x:.2f} → {nearest_int} {'✓' if NumberTheoryUtils.is_prime(nearest_int) else '✗'}")
        
        # حساب معدل النجاح
        success_rate = len(actual_primes) / len(target_pi_values) if target_pi_values else 0
        
        self.log_message(f"📊 معدل اكتشاف الأعداد الأولية: {success_rate:.2%}")
        
        return {
            'target_pi_values': target_pi_values,
            'predicted_positions': discovered_positions,
            'discovered_primes': actual_primes,
            'success_rate': success_rate
        }
    
    def nth_prime_prediction(self, n_values):
        """التنبؤ بالعدد الأولي رقم n"""
        
        self.log_message("🎯 التنبؤ بالأعداد الأولية حسب الترتيب...")
        
        if self.inverse_model is None:
            self.log_message("❌ النموذج العكسي غير مدرب!")
            return None
        
        predictions = []
        actual_primes = NumberTheoryUtils.generate_primes(1000)  # قائمة الأعداد الأولية الحقيقية
        
        for n in n_values:
            if n <= len(actual_primes):
                # استخدام النموذج العكسي: π(x) = n → x
                predicted_x = self.inverse_model.evaluate(np.array([n]))[0]
                predicted_prime = int(round(predicted_x))
                actual_prime = actual_primes[n-1]  # العدد الأولي رقم n الحقيقي
                
                error = abs(predicted_prime - actual_prime)
                relative_error = error / actual_prime * 100
                
                predictions.append({
                    'n': n,
                    'predicted': predicted_prime,
                    'actual': actual_prime,
                    'error': error,
                    'relative_error': relative_error
                })
                
                self.log_message(f"   العدد الأولي #{n}: متوقع={predicted_prime}, حقيقي={actual_prime}, خطأ={error} ({relative_error:.1f}%)")
        
        # حساب متوسط الخطأ
        avg_error = np.mean([p['error'] for p in predictions])
        avg_relative_error = np.mean([p['relative_error'] for p in predictions])
        
        self.log_message(f"📊 متوسط الخطأ: {avg_error:.2f} ({avg_relative_error:.1f}%)")
        
        return predictions
    
    def bidirectional_consistency_test(self, test_range=(50, 150)):
        """اختبار الاتساق بين النموذجين الأمامي والعكسي"""
        
        self.log_message("🔄 اختبار الاتساق ثنائي الاتجاه...")
        
        if self.forward_model is None or self.inverse_model is None:
            self.log_message("❌ أحد النماذج غير مدرب!")
            return None
        
        test_x_values = np.linspace(test_range[0], test_range[1], 20)
        consistency_errors = []
        
        for x in test_x_values:
            # الاتجاه الأمامي: x → π(x)
            pi_predicted = self.forward_model.evaluate(np.array([x]))[0]
            
            # الاتجاه العكسي: π(x) → x
            x_reconstructed = self.inverse_model.evaluate(np.array([pi_predicted]))[0]
            
            # حساب خطأ الإعادة البناء
            reconstruction_error = abs(x - x_reconstructed)
            consistency_errors.append(reconstruction_error)
            
            self.log_message(f"   x={x:.1f} → π(x)={pi_predicted:.2f} → x'={x_reconstructed:.2f} (خطأ={reconstruction_error:.3f})")
        
        avg_consistency_error = np.mean(consistency_errors)
        self.log_message(f"📊 متوسط خطأ الاتساق: {avg_consistency_error:.4f}")
        
        return {
            'test_x_values': test_x_values,
            'consistency_errors': consistency_errors,
            'avg_error': avg_consistency_error
        }
    
    def run_complete_inverse_experiment(self):
        """تشغيل التجربة الكاملة للتنبؤ العكسي"""
        
        self.log_message("🚀 بدء التجربة الكاملة للتنبؤ العكسي")
        self.log_message("=" * 60)
        
        # 1. تدريب النموذج الأمامي
        forward_results = self.train_forward_model()
        
        # 2. تدريب النموذج العكسي
        inverse_results = self.train_inverse_model()
        
        # 3. اختبار اكتشاف الأعداد الأولية
        target_pi_values = [5, 10, 15, 20, 25]  # نريد العثور على x حيث π(x) = هذه القيم
        discovery_results = self.prime_discovery_test(target_pi_values)
        
        # 4. التنبؤ بالأعداد الأولية حسب الترتيب
        n_values = [10, 20, 30, 40, 50]  # العدد الأولي رقم n
        nth_prime_results = self.nth_prime_prediction(n_values)
        
        # 5. اختبار الاتساق
        consistency_results = self.bidirectional_consistency_test()
        
        # حفظ النتائج
        self.results = {
            'forward_model': forward_results,
            'inverse_model': inverse_results,
            'prime_discovery': discovery_results,
            'nth_prime_prediction': nth_prime_results,
            'consistency_test': consistency_results
        }
        
        self.log_message("\n" + "=" * 60)
        self.log_message("🎉 انتهت التجربة الكاملة للتنبؤ العكسي!")
        
        return self.results
    
    def visualize_inverse_results(self):
        """تصور نتائج التجربة العكسية"""
        
        if not self.results:
            self.log_message("⚠️ لا توجد نتائج للتصور")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('نتائج تجربة التنبؤ العكسي', fontsize=16, fontweight='bold')
        
        # 1. النموذج الأمامي
        if 'forward_model' in self.results:
            forward = self.results['forward_model']
            axes[0, 0].plot(forward['x_data'], forward['y_true'], 'ro-', markersize=3, label='حقيقي', alpha=0.7)
            axes[0, 0].plot(forward['x_data'], forward['y_pred'], 'b-', linewidth=2, label='متوقع', alpha=0.8)
            axes[0, 0].set_title(f"النموذج الأمامي x → π(x) - R²={forward['r2']:.4f}")
            axes[0, 0].set_xlabel('x')
            axes[0, 0].set_ylabel('π(x)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. النموذج العكسي
        if 'inverse_model' in self.results:
            inverse = self.results['inverse_model']
            axes[0, 1].plot(inverse['x_data'], inverse['y_true'], 'go-', markersize=3, label='حقيقي', alpha=0.7)
            axes[0, 1].plot(inverse['x_data'], inverse['y_pred'], 'm-', linewidth=2, label='متوقع', alpha=0.8)
            axes[0, 1].set_title(f"النموذج العكسي π(x) → x - R²={inverse['r2']:.4f}")
            axes[0, 1].set_xlabel('π(x)')
            axes[0, 1].set_ylabel('x')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. اختبار الاتساق
        if 'consistency_test' in self.results:
            consistency = self.results['consistency_test']
            axes[1, 0].plot(consistency['test_x_values'], consistency['consistency_errors'], 'co-', linewidth=2, markersize=4)
            axes[1, 0].set_title(f"خطأ الاتساق - متوسط={consistency['avg_error']:.4f}")
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('خطأ إعادة البناء')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. دقة التنبؤ بالأعداد الأولية
        if 'nth_prime_prediction' in self.results:
            nth_pred = self.results['nth_prime_prediction']
            n_vals = [p['n'] for p in nth_pred]
            errors = [p['relative_error'] for p in nth_pred]
            axes[1, 1].bar(n_vals, errors, alpha=0.7, color='orange')
            axes[1, 1].set_title('الخطأ النسبي في التنبؤ بالأعداد الأولية')
            axes[1, 1].set_xlabel('ترتيب العدد الأولي')
            axes[1, 1].set_ylabel('الخطأ النسبي (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        self.log_message("📊 تم عرض نتائج التجربة العكسية")

def main():
    """تشغيل تجربة التنبؤ العكسي"""
    
    experiment = InversePredictionExperiment()
    
    # تشغيل التجربة الكاملة
    results = experiment.run_complete_inverse_experiment()
    
    # تصور النتائج
    experiment.visualize_inverse_results()
    
    return experiment, results

if __name__ == "__main__":
    experiment, results = main()
