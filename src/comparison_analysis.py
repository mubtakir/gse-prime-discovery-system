"""
مقارنة نموذج GSE مع دوال السيجمويد المركبة التقليدية
Comparison of GSE Model with Traditional Complex Sigmoid Functions
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

class ComparisonAnalysis:
    """مقارنة نموذج GSE مع النماذج التقليدية"""
    
    def __init__(self):
        self.results = {}
        self.experiment_log = []
    
    def log_message(self, message):
        """تسجيل رسالة مع الوقت"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.experiment_log.append(log_entry)
    
    def complex_generalized_sigmoid(self, x, z, k=1, x0=0):
        """دالة السيجمويد المعممة المركبة (من البرنامج المعروض)"""
        
        a = z.real
        b = z.imag
        
        term = x - x0
        safe_term = np.where(term == 0, 1e-9, term)
        
        # تبسيط للتعامل مع الأس المركب
        complex_log_term = np.log(safe_term.astype(np.complex128))
        complex_power_term = np.exp(z * complex_log_term)
        
        exponent = -k * complex_power_term
        result = 1 / (1 + np.exp(exponent))
        
        return result
    
    def single_complex_sigmoid_model(self, x_data, y_data, z_values):
        """نموذج سيجمويد مركب واحد (كما في البرامج المعروضة)"""
        
        self.log_message("🔄 اختبار نموذج السيجمويد المركب الواحد...")
        
        best_r2 = -np.inf
        best_z = None
        best_predictions = None
        
        for z in z_values:
            try:
                # التنبؤ باستخدام السيجمويد المركب
                y_pred = self.complex_generalized_sigmoid(x_data, z, k=1, x0=np.mean(x_data))
                y_pred_real = y_pred.real
                
                # حساب R²
                ss_res = np.sum((y_data - y_pred_real) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else -np.inf
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_z = z
                    best_predictions = y_pred_real
                    
            except Exception as e:
                continue
        
        return {
            'best_r2': best_r2,
            'best_z': best_z,
            'predictions': best_predictions,
            'method': 'Single Complex Sigmoid'
        }
    
    def riemann_zeros_sigmoid_model(self, x_data, y_data):
        """نموذج سيجمويد باستخدام أصفار ريمان (كما في البرنامج الثاني)"""
        
        self.log_message("🌟 اختبار نموذج أصفار ريمان...")
        
        # أصفار ريمان المعروفة
        riemann_zeros = [
            -2 + 0j,  # أصفار بديهية
            -4 + 0j,
            -6 + 0j,
            0.5 + 14.134725j,  # أصفار غير بديهية
            0.5 + 21.022039j,
            0.5 + 25.010857j
        ]
        
        return self.single_complex_sigmoid_model(x_data, y_data, riemann_zeros)
    
    def gse_ensemble_model(self, x_data, y_data):
        """نموذج GSE المتقدم (نموذجنا)"""
        
        self.log_message("🚀 اختبار نموذج GSE المتقدم...")
        
        # إنشاء نموذج GSE
        gse_model = AdvancedGSEModel()
        gse_model.add_sigmoid(alpha=complex(1.0, 0.1), n=2.0, z=complex(1.0, 0.0), x0=10.0)
        gse_model.add_sigmoid(alpha=complex(0.8, -0.1), n=1.8, z=complex(0.9, 0.1), x0=50.0)
        gse_model.add_sigmoid(alpha=complex(0.6, 0.2), n=1.5, z=complex(1.1, -0.1), x0=100.0)
        
        # التدريب
        optimizer = GSEOptimizer(gse_model)
        start_time = time.time()
        result = optimizer.optimize_differential_evolution(x_data, y_data, max_iter=200, verbose=False)
        training_time = time.time() - start_time
        
        # التقييم
        y_pred = gse_model.evaluate(x_data)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else -np.inf
        
        return {
            'best_r2': r2,
            'predictions': y_pred,
            'training_time': training_time,
            'model': gse_model,
            'method': 'GSE Ensemble'
        }
    
    def comprehensive_comparison(self, max_n=300):
        """مقارنة شاملة بين النماذج المختلفة"""
        
        self.log_message("🔬 بدء المقارنة الشاملة...")
        self.log_message("=" * 60)
        
        # إعداد البيانات
        x_data, y_data = TargetFunctions.prime_counting_function((2, max_n))
        
        self.log_message(f"📊 البيانات: {len(x_data)} نقطة من 2 إلى {max_n}")
        
        # 1. اختبار السيجمويد المركب التقليدي
        traditional_z_values = [
            1 + 0j, 2 + 0j, 3 + 0j,  # حقيقية
            1 + 1j, 2 + 1j, 3 + 2j,  # مركبة بسيطة
            0.5 + 5j, 1 + 10j, 2 + 15j  # تذبذب عالي
        ]
        
        traditional_result = self.single_complex_sigmoid_model(x_data, y_data, traditional_z_values)
        
        # 2. اختبار نموذج أصفار ريمان
        riemann_result = self.riemann_zeros_sigmoid_model(x_data, y_data)
        
        # 3. اختبار نموذج GSE
        gse_result = self.gse_ensemble_model(x_data, y_data)
        
        # حفظ النتائج
        self.results = {
            'data': {'x': x_data, 'y': y_data},
            'traditional_sigmoid': traditional_result,
            'riemann_zeros': riemann_result,
            'gse_ensemble': gse_result
        }
        
        # طباعة النتائج
        self.print_comparison_results()
        
        return self.results
    
    def print_comparison_results(self):
        """طباعة نتائج المقارنة"""
        
        print("\n" + "="*80)
        print("🏆 نتائج المقارنة الشاملة")
        print("="*80)
        
        methods = ['traditional_sigmoid', 'riemann_zeros', 'gse_ensemble']
        method_names = ['السيجمويد التقليدي', 'أصفار ريمان', 'GSE المتقدم']
        
        results_table = []
        
        for method, name in zip(methods, method_names):
            if method in self.results:
                result = self.results[method]
                r2 = result.get('best_r2', 0)
                
                if method == 'gse_ensemble':
                    training_time = result.get('training_time', 0)
                    results_table.append({
                        'method': name,
                        'r2': r2,
                        'percentage': r2 * 100,
                        'training_time': training_time,
                        'parameters': f"{len(result.get('model', {}).sigmoid_components)} مكونات" if 'model' in result else 'N/A'
                    })
                else:
                    best_z = result.get('best_z', 'N/A')
                    results_table.append({
                        'method': name,
                        'r2': r2,
                        'percentage': r2 * 100,
                        'training_time': 'N/A',
                        'parameters': f"z = {best_z}"
                    })
        
        # ترتيب حسب الأداء
        results_table.sort(key=lambda x: x['r2'], reverse=True)
        
        print(f"{'الترتيب':<5} {'الطريقة':<20} {'R²':<10} {'النسبة المئوية':<15} {'وقت التدريب':<15} {'المعاملات':<30}")
        print("-" * 95)
        
        for i, result in enumerate(results_table, 1):
            print(f"{i:<5} {result['method']:<20} {result['r2']:<10.4f} {result['percentage']:<15.2f}% {str(result['training_time']):<15} {result['parameters']:<30}")
        
        # تحليل النتائج
        print(f"\n🎯 التحليل:")
        best_method = results_table[0]
        print(f"   🥇 الأفضل: {best_method['method']} بدقة {best_method['percentage']:.2f}%")
        
        if len(results_table) > 1:
            second_best = results_table[1]
            improvement = best_method['percentage'] - second_best['percentage']
            print(f"   📈 التحسن: {improvement:.2f}% عن الطريقة الثانية")
        
        print("="*80)
    
    def visualize_comparison(self):
        """تصور المقارنة بين النماذج"""
        
        if not self.results:
            self.log_message("⚠️ لا توجد نتائج للتصور")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('مقارنة شاملة: GSE مقابل النماذج التقليدية', fontsize=16, fontweight='bold')
        
        x_data = self.results['data']['x']
        y_true = self.results['data']['y']
        
        # 1. مقارنة الأداء
        methods = ['traditional_sigmoid', 'riemann_zeros', 'gse_ensemble']
        method_names = ['السيجمويد التقليدي', 'أصفار ريمان', 'GSE المتقدم']
        colors = ['red', 'blue', 'green']
        
        axes[0, 0].plot(x_data, y_true, 'ko-', markersize=2, label='البيانات الحقيقية', alpha=0.7)
        
        for method, name, color in zip(methods, method_names, colors):
            if method in self.results and 'predictions' in self.results[method]:
                y_pred = self.results[method]['predictions']
                r2 = self.results[method]['best_r2']
                axes[0, 0].plot(x_data, y_pred, color=color, linewidth=2, 
                              label=f'{name} (R²={r2:.4f})', alpha=0.8)
        
        axes[0, 0].set_title('مقارنة التنبؤات')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('π(x)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. مقارنة الدقة
        r2_values = []
        method_labels = []
        
        for method, name in zip(methods, method_names):
            if method in self.results:
                r2_values.append(self.results[method]['best_r2'] * 100)
                method_labels.append(name)
        
        bars = axes[0, 1].bar(method_labels, r2_values, color=colors[:len(r2_values)], alpha=0.7)
        axes[0, 1].set_title('مقارنة الدقة (R² %)')
        axes[0, 1].set_ylabel('R² (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # إضافة قيم على الأعمدة
        for bar, value in zip(bars, r2_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. تحليل الأخطاء (GSE فقط)
        if 'gse_ensemble' in self.results:
            gse_pred = self.results['gse_ensemble']['predictions']
            residuals = y_true - gse_pred
            
            axes[1, 0].scatter(x_data, residuals, alpha=0.6, color='green', s=20)
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[1, 0].set_title('تحليل أخطاء نموذج GSE')
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('الخطأ (حقيقي - متوقع)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. توزيع الأخطاء
        if 'gse_ensemble' in self.results:
            gse_pred = self.results['gse_ensemble']['predictions']
            residuals = y_true - gse_pred
            
            axes[1, 1].hist(residuals, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[1, 1].set_title('توزيع أخطاء نموذج GSE')
            axes[1, 1].set_xlabel('الخطأ')
            axes[1, 1].set_ylabel('التكرار')
            axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        self.log_message("📊 تم عرض المقارنة البصرية")

def main():
    """تشغيل المقارنة الشاملة"""
    
    comparison = ComparisonAnalysis()
    
    # تشغيل المقارنة
    results = comparison.comprehensive_comparison(max_n=200)
    
    # تصور النتائج
    comparison.visualize_comparison()
    
    return comparison, results

if __name__ == "__main__":
    comparison, results = main()
