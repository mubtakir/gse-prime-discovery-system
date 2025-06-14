"""
محرك التحسين المتقدم لنموذج GSE
Advanced Optimization Engine for GSE Model
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.optimize import dual_annealing
import warnings
warnings.filterwarnings('ignore')

class GSEOptimizer:
    """محرك التحسين المتقدم لنموذج GSE"""
    
    def __init__(self, model):
        self.model = model
        self.optimization_history = []
        self.best_loss = float('inf')
        self.best_params = None
        
    def loss_function(self, param_vector, x_data, y_true, regularization=0.001):
        """
        دالة الخطأ المتقدمة مع التنظيم
        """
        try:
            # تحديث معاملات النموذج
            self.model.set_params_from_vector(param_vector)
            
            # تقييم النموذج
            y_pred = self.model.evaluate(x_data)
            
            # حساب متوسط الخطأ التربيعي
            mse = np.mean((y_true - y_pred) ** 2)
            
            # إضافة تنظيم L2 للمعاملات
            l2_penalty = regularization * np.sum(param_vector ** 2)
            
            # إضافة تنظيم للتعقيد (عدد المكونات)
            complexity_penalty = 0.01 * len(self.model.sigmoid_components)
            
            total_loss = mse + l2_penalty + complexity_penalty
            
            # حفظ التاريخ
            self.optimization_history.append(total_loss)
            
            # تحديث أفضل نتيجة
            if total_loss < self.best_loss:
                self.best_loss = total_loss
                self.best_params = param_vector.copy()
            
            return total_loss
            
        except Exception as e:
            # في حالة الخطأ، إرجاع قيمة كبيرة
            return 1e10
    
    def get_parameter_bounds(self):
        """الحصول على حدود المعاملات للتحسين"""
        bounds = []
        
        for component in self.model.sigmoid_components:
            # حدود لكل مكون سيجمويد
            bounds.extend([
                (-10, 10),    # alpha_real
                (-10, 10),    # alpha_imag  
                (0.1, 5.0),   # n (الأس)
                (-5, 5),      # z_real
                (-5, 5),      # z_imag
                (-50, 50)     # x0 (الإزاحة)
            ])
        
        # حدود للمعاملات الخطية
        bounds.extend([(-5, 5), (-5, 5)])  # beta, gamma
        
        return bounds
    
    def generate_initial_population(self, pop_size=50):
        """توليد مجموعة أولية من المعاملات"""
        bounds = self.get_parameter_bounds()
        population = []
        
        for _ in range(pop_size):
            individual = []
            for lower, upper in bounds:
                individual.append(np.random.uniform(lower, upper))
            population.append(np.array(individual))
        
        return population
    
    def optimize_differential_evolution(self, x_data, y_true, max_iter=1000, 
                                      popsize=15, seed=42, verbose=True):
        """تحسين باستخدام التطور التفاضلي"""
        if verbose:
            print("🧬 بدء التحسين بالتطور التفاضلي...")
        
        bounds = self.get_parameter_bounds()
        
        def objective(params):
            return self.loss_function(params, x_data, y_true)
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iter,
            popsize=popsize,
            seed=seed,
            disp=verbose,
            atol=1e-8,
            tol=1e-8,
            workers=1  # تجنب مشاكل التوازي
        )
        
        if verbose:
            print(f"✅ انتهى التطور التفاضلي. أفضل خطأ: {result.fun:.8f}")
        
        return result
    
    def optimize_simulated_annealing(self, x_data, y_true, max_iter=1000, verbose=True):
        """تحسين باستخدام التبريد المحاكي"""
        if verbose:
            print("🌡️ بدء التحسين بالتبريد المحاكي...")
        
        bounds = self.get_parameter_bounds()
        
        def objective(params):
            return self.loss_function(params, x_data, y_true)
        
        result = dual_annealing(
            objective,
            bounds,
            maxiter=max_iter,
            seed=42
        )
        
        if verbose:
            print(f"✅ انتهى التبريد المحاكي. أفضل خطأ: {result.fun:.8f}")
        
        return result
    
    def optimize_basin_hopping(self, x_data, y_true, n_iter=100, verbose=True):
        """تحسين باستخدام Basin Hopping"""
        if verbose:
            print("🏔️ بدء التحسين بـ Basin Hopping...")
        
        # نقطة بداية عشوائية
        initial_params = self.model.get_params_vector()
        bounds = self.get_parameter_bounds()
        
        def objective(params):
            return self.loss_function(params, x_data, y_true)
        
        # خيارات للمحسن المحلي
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds,
            "options": {"maxiter": 100}
        }
        
        result = basinhopping(
            objective,
            initial_params,
            niter=n_iter,
            minimizer_kwargs=minimizer_kwargs,
            seed=42,
            disp=verbose
        )
        
        if verbose:
            print(f"✅ انتهى Basin Hopping. أفضل خطأ: {result.fun:.8f}")
        
        return result
    
    def optimize_hybrid(self, x_data, y_true, verbose=True):
        """تحسين هجين يجمع عدة خوارزميات"""
        if verbose:
            print("🔄 بدء التحسين الهجين...")
        
        results = []
        
        # المرحلة 1: التطور التفاضلي (استكشاف عام)
        if verbose:
            print("المرحلة 1: التطور التفاضلي...")
        result1 = self.optimize_differential_evolution(
            x_data, y_true, max_iter=200, verbose=False
        )
        results.append(('differential_evolution', result1))
        
        # المرحلة 2: التبريد المحاكي (تحسين متوسط)
        if verbose:
            print("المرحلة 2: التبريد المحاكي...")
        result2 = self.optimize_simulated_annealing(
            x_data, y_true, max_iter=200, verbose=False
        )
        results.append(('simulated_annealing', result2))
        
        # المرحلة 3: Basin Hopping (تحسين محلي)
        if verbose:
            print("المرحلة 3: Basin Hopping...")
        result3 = self.optimize_basin_hopping(
            x_data, y_true, n_iter=50, verbose=False
        )
        results.append(('basin_hopping', result3))
        
        # اختيار أفضل نتيجة
        best_result = min(results, key=lambda x: x[1].fun)
        
        if verbose:
            print(f"✅ انتهى التحسين الهجين.")
            print(f"أفضل طريقة: {best_result[0]}")
            print(f"أفضل خطأ: {best_result[1].fun:.8f}")
        
        return best_result[1]
    
    def train_model(self, x_data, y_true, method='hybrid', **kwargs):
        """
        تدريب النموذج باستخدام الطريقة المحددة
        """
        print(f"🚀 بدء تدريب نموذج GSE...")
        print(f"📊 البيانات: {len(x_data)} نقطة")
        print(f"🎯 الهدف: {method}")
        
        # إعادة تعيين التاريخ
        self.optimization_history = []
        self.best_loss = float('inf')
        
        # اختيار طريقة التحسين
        if method == 'differential_evolution':
            result = self.optimize_differential_evolution(x_data, y_true, **kwargs)
        elif method == 'simulated_annealing':
            result = self.optimize_simulated_annealing(x_data, y_true, **kwargs)
        elif method == 'basin_hopping':
            result = self.optimize_basin_hopping(x_data, y_true, **kwargs)
        elif method == 'hybrid':
            result = self.optimize_hybrid(x_data, y_true, **kwargs)
        else:
            raise ValueError(f"طريقة غير مدعومة: {method}")
        
        # تحديث النموذج بأفضل معاملات
        self.model.set_params_from_vector(result.x)
        self.model.trained = True
        self.model.training_history = self.optimization_history
        
        print(f"✅ انتهى التدريب بنجاح!")
        print(f"📈 الخطأ النهائي: {result.fun:.8f}")
        print(f"🔄 عدد التكرارات: {len(self.optimization_history)}")
        
        return result
    
    def evaluate_convergence(self):
        """تقييم تقارب التحسين"""
        if len(self.optimization_history) < 10:
            return "غير كافي للتقييم"
        
        # حساب معدل التحسن في آخر 100 تكرار
        recent_history = self.optimization_history[-100:]
        improvement_rate = (recent_history[0] - recent_history[-1]) / recent_history[0]
        
        if improvement_rate < 0.001:
            return "متقارب"
        elif improvement_rate < 0.01:
            return "تقارب بطيء"
        else:
            return "يتحسن"
    
    def get_optimization_summary(self):
        """ملخص عملية التحسين"""
        return {
            'total_iterations': len(self.optimization_history),
            'best_loss': self.best_loss,
            'final_loss': self.optimization_history[-1] if self.optimization_history else None,
            'convergence_status': self.evaluate_convergence(),
            'improvement_ratio': (self.optimization_history[0] - self.best_loss) / self.optimization_history[0] if self.optimization_history else 0
        }
