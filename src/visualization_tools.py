"""
أدوات التصور المتقدمة لنموذج GSE
رسم النتائج والتحليلات البصرية
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# محاولة استيراد seaborn (اختياري)
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# إعداد الخط العربي
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Tahoma', 'DejaVu Sans']

class GSEVisualizer:
    """أدوات التصور المتقدمة لنموذج GSE"""
    
    def __init__(self, model=None):
        self.model = model
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    def plot_model_prediction(self, x_range=(2, 150), title="توقعات نموذج GSE"):
        """رسم توقعات النموذج مقابل البيانات الحقيقية"""
        if not self.model:
            print("⚠️ لا يوجد نموذج محمل")
            return
        
        from .number_theory_utils import NumberTheoryUtils
        
        # إعداد البيانات
        x_vals = np.arange(x_range[0], x_range[1] + 1)
        y_true = np.array([1 if NumberTheoryUtils.is_prime(int(x)) else 0 for x in x_vals])
        y_pred = self.model.evaluate(x_vals)
        
        # الرسم
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # الرسم الأول: المقارنة المباشرة
        ax1.plot(x_vals, y_true, 'ro', markersize=3, label='الأعداد الأولية الحقيقية', alpha=0.7)
        ax1.plot(x_vals, y_pred, 'b-', linewidth=2, label='توقعات GSE', alpha=0.8)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='عتبة التصنيف')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_xlabel('العدد (x)')
        ax1.set_ylabel('الاحتمالية')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # الرسم الثاني: الخطأ
        error = np.abs(y_true - y_pred)
        ax2.plot(x_vals, error, 'r-', linewidth=1, alpha=0.7)
        ax2.fill_between(x_vals, error, alpha=0.3, color='red')
        ax2.set_title('الخطأ المطلق بين التوقع والحقيقة')
        ax2.set_xlabel('العدد (x)')
        ax2.set_ylabel('الخطأ المطلق')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # حساب الإحصائيات
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        print(f"📊 إحصائيات الأداء:")
        print(f"   متوسط الخطأ التربيعي (MSE): {mse:.6f}")
        print(f"   متوسط الخطأ المطلق (MAE): {mae:.6f}")
    
    def plot_sigmoid_components(self, x_range=(-10, 10), num_points=1000):
        """رسم مكونات السيجمويد المختلفة"""
        if not self.model or not self.model.sigmoid_components:
            print("⚠️ لا توجد مكونات سيجمويد للعرض")
            return
        
        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('تحليل مكونات السيجمويد في النموذج', fontsize=16, fontweight='bold')
        
        # الرسم الأول: جميع المكونات
        ax1 = axes[0, 0]
        total_prediction = np.zeros_like(x_vals)
        
        for i, comp in enumerate(self.model.sigmoid_components):
            component_output = self.model.complex_sigmoid(
                x_vals, comp['alpha'], comp['n'], comp['z'], comp['x0']
            ).real
            
            color = self.colors[i % len(self.colors)]
            ax1.plot(x_vals, component_output, color=color, linewidth=2, 
                    label=f'مكون {i+1}: α={comp["alpha"]:.2f}', alpha=0.8)
            total_prediction += component_output
        
        ax1.plot(x_vals, total_prediction, 'k-', linewidth=3, label='المجموع الكلي', alpha=0.9)
        ax1.set_title('مكونات السيجمويد الفردية')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # الرسم الثاني: الجزء الحقيقي والتخيلي
        ax2 = axes[0, 1]
        for i, comp in enumerate(self.model.sigmoid_components):
            component_output = self.model.complex_sigmoid(
                x_vals, comp['alpha'], comp['n'], comp['z'], comp['x0']
            )
            
            color = self.colors[i % len(self.colors)]
            ax2.plot(x_vals, component_output.real, color=color, linewidth=2, 
                    label=f'حقيقي {i+1}', linestyle='-')
            ax2.plot(x_vals, component_output.imag, color=color, linewidth=1, 
                    label=f'تخيلي {i+1}', linestyle='--', alpha=0.7)
        
        ax2.set_title('الأجزاء الحقيقية والتخيلية')
        ax2.set_xlabel('x')
        ax2.set_ylabel('القيمة')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # الرسم الثالث: تأثير المعاملات
        ax3 = axes[1, 0]
        if len(self.model.sigmoid_components) > 0:
            comp = self.model.sigmoid_components[0]
            
            # تأثير معامل n
            n_values = [0.5, 1.0, 2.0, 3.0]
            for n_val in n_values:
                temp_output = self.model.complex_sigmoid(
                    x_vals, comp['alpha'], n_val, comp['z'], comp['x0']
                ).real
                ax3.plot(x_vals, temp_output, linewidth=2, label=f'n = {n_val}')
            
            ax3.set_title('تأثير معامل الأس n')
            ax3.set_xlabel('x')
            ax3.set_ylabel('f(x)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # الرسم الرابع: تأثير الإزاحة x0
        ax4 = axes[1, 1]
        if len(self.model.sigmoid_components) > 0:
            comp = self.model.sigmoid_components[0]
            
            # تأثير معامل x0
            x0_values = [-5, 0, 5, 10]
            for x0_val in x0_values:
                temp_output = self.model.complex_sigmoid(
                    x_vals, comp['alpha'], comp['n'], comp['z'], x0_val
                ).real
                ax4.plot(x_vals, temp_output, linewidth=2, label=f'x₀ = {x0_val}')
            
            ax4.set_title('تأثير معامل الإزاحة x₀')
            ax4.set_xlabel('x')
            ax4.set_ylabel('f(x)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_3d_surface(self, param_range=(-5, 5), resolution=50):
        """رسم سطح ثلاثي الأبعاد للدالة"""
        if not self.model:
            print("⚠️ لا يوجد نموذج محمل")
            return
        
        # إنشاء شبكة ثلاثية الأبعاد
        x = np.linspace(param_range[0], param_range[1], resolution)
        y = np.linspace(param_range[0], param_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # حساب القيم للسطح (استخدام أول مكون سيجمويد)
        if self.model.sigmoid_components:
            comp = self.model.sigmoid_components[0]
            Z = np.zeros_like(X)
            
            for i in range(resolution):
                for j in range(resolution):
                    # تقييم الدالة عند النقطة (X[i,j], Y[i,j])
                    complex_input = X[i, j] + 1j * Y[i, j]
                    result = self.model.complex_sigmoid(
                        complex_input, comp['alpha'], comp['n'], comp['z'], comp['x0']
                    )
                    Z[i, j] = abs(result)  # استخدام القيمة المطلقة
            
            # الرسم ثلاثي الأبعاد
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap='viridis', alpha=0.5)
            
            ax.set_title('السطح ثلاثي الأبعاد لدالة السيجمويد المركبة')
            ax.set_xlabel('الجزء الحقيقي')
            ax.set_ylabel('الجزء التخيلي')
            ax.set_zlabel('|f(z)|')
            
            plt.colorbar(surface)
            plt.show()
    
    def plot_training_analysis(self):
        """تحليل بصري لعملية التدريب"""
        if not self.model or not self.model.training_history:
            print("⚠️ لا يوجد تاريخ تدريب للعرض")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('تحليل عملية التدريب', fontsize=16, fontweight='bold')
        
        history = self.model.training_history
        
        # الرسم الأول: تطور الخطأ
        ax1 = axes[0, 0]
        ax1.plot(history, 'b-', linewidth=2)
        ax1.set_title('تطور دالة الخطأ')
        ax1.set_xlabel('التكرار')
        ax1.set_ylabel('قيمة الخطأ')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # الرسم الثاني: معدل التحسن
        ax2 = axes[0, 1]
        if len(history) > 1:
            improvement_rate = [history[i-1] - history[i] for i in range(1, len(history))]
            ax2.plot(improvement_rate, 'g-', linewidth=2)
            ax2.set_title('معدل التحسن')
            ax2.set_xlabel('التكرار')
            ax2.set_ylabel('التحسن في الخطأ')
            ax2.grid(True, alpha=0.3)
        
        # الرسم الثالث: توزيع قيم الخطأ
        ax3 = axes[1, 0]
        ax3.hist(history, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax3.set_title('توزيع قيم الخطأ')
        ax3.set_xlabel('قيمة الخطأ')
        ax3.set_ylabel('التكرار')
        ax3.grid(True, alpha=0.3)
        
        # الرسم الرابع: الخطأ المتحرك
        ax4 = axes[1, 1]
        if len(history) > 10:
            window_size = min(50, len(history) // 10)
            moving_avg = np.convolve(history, np.ones(window_size)/window_size, mode='valid')
            ax4.plot(history, 'lightblue', alpha=0.5, label='الخطأ الأصلي')
            ax4.plot(range(window_size-1, len(history)), moving_avg, 'red', linewidth=2, label=f'المتوسط المتحرك ({window_size})')
            ax4.set_title('المتوسط المتحرك للخطأ')
            ax4.set_xlabel('التكرار')
            ax4.set_ylabel('قيمة الخطأ')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_prime_prediction_analysis(self, x_range=(2, 200)):
        """تحليل مفصل لتوقع الأعداد الأولية"""
        if not self.model:
            print("⚠️ لا يوجد نموذج محمل")
            return
        
        from .number_theory_utils import NumberTheoryUtils
        
        # إعداد البيانات
        x_vals = np.arange(x_range[0], x_range[1] + 1)
        y_true = np.array([1 if NumberTheoryUtils.is_prime(int(x)) else 0 for x in x_vals])
        y_pred = self.model.evaluate(x_vals)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # حساب مصفوفة الخلط
        true_positives = np.sum((y_true == 1) & (y_pred_binary == 1))
        false_positives = np.sum((y_true == 0) & (y_pred_binary == 1))
        true_negatives = np.sum((y_true == 0) & (y_pred_binary == 0))
        false_negatives = np.sum((y_true == 1) & (y_pred_binary == 0))
        
        # حساب المقاييس
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(y_true)
        
        # الرسم
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('تحليل شامل لتوقع الأعداد الأولية', fontsize=16, fontweight='bold')
        
        # الرسم الأول: التوقعات مع التصنيف
        ax1 = axes[0, 0]
        
        # تصنيف النقاط
        correct_primes = x_vals[(y_true == 1) & (y_pred_binary == 1)]
        missed_primes = x_vals[(y_true == 1) & (y_pred_binary == 0)]
        false_primes = x_vals[(y_true == 0) & (y_pred_binary == 1)]
        correct_composites = x_vals[(y_true == 0) & (y_pred_binary == 0)]
        
        ax1.scatter(correct_primes, [1]*len(correct_primes), color='green', s=30, label=f'أولي صحيح ({len(correct_primes)})', alpha=0.8)
        ax1.scatter(missed_primes, [1]*len(missed_primes), color='red', s=30, label=f'أولي مفقود ({len(missed_primes)})', alpha=0.8)
        ax1.scatter(false_primes, [0]*len(false_primes), color='orange', s=30, label=f'أولي خاطئ ({len(false_primes)})', alpha=0.8)
        
        ax1.plot(x_vals, y_pred, 'b-', linewidth=1, alpha=0.6, label='توقع GSE')
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        ax1.set_title('تصنيف التوقعات')
        ax1.set_xlabel('العدد')
        ax1.set_ylabel('التصنيف')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # الرسم الثاني: مصفوفة الخلط
        ax2 = axes[0, 1]
        confusion_matrix = np.array([[true_negatives, false_positives],
                                   [false_negatives, true_positives]])
        
        im = ax2.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
        ax2.set_title('مصفوفة الخلط')
        
        # إضافة النصوص
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, confusion_matrix[i, j], ha="center", va="center", fontsize=14, fontweight='bold')
        
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['مركب', 'أولي'])
        ax2.set_yticklabels(['مركب', 'أولي'])
        ax2.set_xlabel('التوقع')
        ax2.set_ylabel('الحقيقة')
        
        # الرسم الثالث: المقاييس
        ax3 = axes[1, 0]
        metrics = ['الدقة', 'الاستدعاء', 'F1-Score', 'الصحة']
        values = [precision, recall, f1_score, accuracy]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow']
        
        bars = ax3.bar(metrics, values, color=colors, edgecolor='black', alpha=0.8)
        ax3.set_title('مقاييس الأداء')
        ax3.set_ylabel('القيمة')
        ax3.set_ylim(0, 1)
        
        # إضافة القيم على الأعمدة
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.grid(True, alpha=0.3, axis='y')
        
        # الرسم الرابع: توزيع الثقة
        ax4 = axes[1, 1]
        
        # توزيع الثقة للأعداد الأولية والمركبة
        prime_confidences = y_pred[y_true == 1]
        composite_confidences = y_pred[y_true == 0]
        
        ax4.hist(prime_confidences, bins=20, alpha=0.6, label='أعداد أولية', color='green', density=True)
        ax4.hist(composite_confidences, bins=20, alpha=0.6, label='أعداد مركبة', color='red', density=True)
        ax4.axvline(x=0.5, color='black', linestyle='--', label='عتبة التصنيف')
        
        ax4.set_title('توزيع الثقة في التوقعات')
        ax4.set_xlabel('مستوى الثقة')
        ax4.set_ylabel('الكثافة')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # طباعة الإحصائيات
        print(f"\n📊 تقرير الأداء التفصيلي:")
        print(f"   الدقة (Precision): {precision:.4f}")
        print(f"   الاستدعاء (Recall): {recall:.4f}")
        print(f"   F1-Score: {f1_score:.4f}")
        print(f"   الصحة (Accuracy): {accuracy:.4f}")
        print(f"\n🔍 تفاصيل التصنيف:")
        print(f"   أعداد أولية صحيحة: {true_positives}")
        print(f"   أعداد أولية مفقودة: {false_negatives}")
        print(f"   أعداد أولية خاطئة: {false_positives}")
        print(f"   أعداد مركبة صحيحة: {true_negatives}")
    
    def create_interactive_plot(self, x_range=(2, 100)):
        """إنشاء رسم تفاعلي (يتطلب jupyter notebook)"""
        try:
            from ipywidgets import interact, FloatSlider
            import ipywidgets as widgets
        except ImportError:
            print("⚠️ المكتبات التفاعلية غير متوفرة. استخدم: pip install ipywidgets")
            return
        
        def update_plot(threshold=0.5):
            if not self.model:
                return
            
            from .number_theory_utils import NumberTheoryUtils
            
            x_vals = np.arange(x_range[0], x_range[1] + 1)
            y_true = np.array([1 if NumberTheoryUtils.is_prime(int(x)) else 0 for x in x_vals])
            y_pred = self.model.evaluate(x_vals)
            y_pred_binary = (y_pred > threshold).astype(int)
            
            plt.figure(figsize=(12, 6))
            plt.plot(x_vals, y_true, 'ro', markersize=3, label='حقيقي', alpha=0.7)
            plt.plot(x_vals, y_pred, 'b-', linewidth=2, label='توقع', alpha=0.8)
            plt.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5, label=f'عتبة = {threshold}')
            
            plt.title(f'تأثير تغيير العتبة على التصنيف')
            plt.xlabel('العدد')
            plt.ylabel('الاحتمالية')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # حساب الدقة
            accuracy = np.mean(y_true == y_pred_binary)
            print(f"الدقة عند العتبة {threshold}: {accuracy:.4f}")
        
        # إنشاء التفاعل
        interact(update_plot, threshold=FloatSlider(min=0.1, max=0.9, step=0.05, value=0.5))
