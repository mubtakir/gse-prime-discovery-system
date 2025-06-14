"""
نموذج GSE المتقدم (Generalized Sigmoid Equation) 
لتقريب الدوال المعقدة مثل دالة عد الأعداد الأولية
تطوير متقدم بناءً على الأكواد الأولية
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.special import zeta
import warnings
warnings.filterwarnings('ignore')

class AdvancedGSEModel:
    """
    نموذج المعادلة السيجمويدية المعممة المتقدم
    يدعم الأعداد المركبة والتحسين المتقدم
    """
    
    def __init__(self):
        self.sigmoid_components = []
        self.linear_params = {'beta': 0.0, 'gamma': 0.0}
        self.trained = False
        self.training_history = []
        self.best_params = None
        
    def add_sigmoid(self, alpha=1.0, n=1.0, z=complex(1.0, 0.0), x0=0.0):
        """إضافة مكون سيجمويد جديد للنموذج"""
        component = {
            'alpha': complex(alpha) if not isinstance(alpha, complex) else alpha,
            'n': n,
            'z': z,
            'x0': x0
        }
        self.sigmoid_components.append(component)
        
    def complex_sigmoid(self, x, alpha, n, z, x0):
        """
        دالة السيجمويد المعممة المركبة المحسنة
        σ(x) = α / (1 + exp(-z * (x - x0)^n))
        """
        x = np.asarray(x, dtype=np.complex128)
        
        # حساب (x - x0)^n
        term = x - x0
        
        # تجنب المشاكل العددية
        epsilon = 1e-15
        term = np.where(np.abs(term) < epsilon, epsilon, term)
        
        # حساب الأس
        if n != 1.0:
            # استخدام exp(n * log(term)) للأس المركب
            power_term = np.exp(n * np.log(term))
        else:
            power_term = term
            
        # حساب الأس النهائي
        exponent = -z * power_term
        
        # تجنب overflow/underflow
        exponent_real = np.clip(exponent.real, -700, 700)
        exponent = exponent_real + 1j * exponent.imag
        
        # حساب السيجمويد
        result = alpha / (1 + np.exp(exponent))
        
        return result
    
    def evaluate(self, x):
        """تقييم النموذج عند النقاط المعطاة"""
        if not self.sigmoid_components:
            return np.zeros_like(x)
        
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=np.complex128)
        
        # جمع مساهمات جميع مكونات السيجمويد
        for comp in self.sigmoid_components:
            sigmoid_result = self.complex_sigmoid(
                x, comp['alpha'], comp['n'], 
                comp['z'], comp['x0']
            )
            result += sigmoid_result
        
        # إضافة المكون الخطي
        result += self.linear_params['beta'] * x + self.linear_params['gamma']
        
        # إرجاع الجزء الحقيقي فقط للتطبيقات العملية
        return result.real
    
    def loss_function(self, x_data, y_true):
        """حساب دالة الخطأ"""
        try:
            y_pred = self.evaluate(x_data)
            
            # متوسط الخطأ التربيعي
            mse = np.mean((y_true - y_pred) ** 2)
            
            # إضافة تنظيم للمعاملات الكبيرة
            regularization = 0.001 * sum(
                abs(comp['alpha'])**2 + abs(comp['z'])**2 
                for comp in self.sigmoid_components
            )
            
            total_loss = mse + regularization
            return total_loss
        except:
            return 1e10  # قيمة كبيرة في حالة الخطأ
    
    def get_params_vector(self):
        """تحويل معاملات النموذج إلى متجه للتحسين"""
        params = []
        for comp in self.sigmoid_components:
            params.extend([
                comp['alpha'].real,
                comp['alpha'].imag,
                comp['n'],
                comp['z'].real,
                comp['z'].imag,
                comp['x0']
            ])
        params.extend([self.linear_params['beta'], self.linear_params['gamma']])
        return np.array(params)
    
    def set_params_from_vector(self, param_vector):
        """تحديث معاملات النموذج من متجه"""
        idx = 0
        for comp in self.sigmoid_components:
            comp['alpha'] = complex(param_vector[idx], param_vector[idx + 1])
            comp['n'] = param_vector[idx + 2]
            comp['z'] = complex(param_vector[idx + 3], param_vector[idx + 4])
            comp['x0'] = param_vector[idx + 5]
            idx += 6
        
        self.linear_params['beta'] = param_vector[idx]
        self.linear_params['gamma'] = param_vector[idx + 1]
    
    def get_param_bounds(self):
        """الحصول على حدود المعاملات للتحسين"""
        bounds = []
        for _ in self.sigmoid_components:
            bounds.extend([
                (-10, 10),    # alpha.real
                (-10, 10),    # alpha.imag
                (0.1, 5),     # n
                (-5, 5),      # z.real
                (-5, 5),      # z.imag
                (-50, 50)     # x0
            ])
        bounds.extend([(-5, 5), (-5, 5)])  # beta, gamma
        return bounds
    
    def optimize_advanced(self, x_data, y_data, method='differential_evolution', 
                         max_iter=1000, verbose=True):
        """تحسين متقدم للنموذج"""
        if verbose:
            print(f"🚀 بدء التحسين المتقدم باستخدام {method}...")
        
        # دالة الهدف للتحسين
        def objective(params):
            try:
                self.set_params_from_vector(params)
                loss = self.loss_function(x_data, y_data)
                self.training_history.append(loss)
                return loss
            except:
                return 1e10
        
        # الحصول على المعاملات الأولية والحدود
        initial_params = self.get_params_vector()
        bounds = self.get_param_bounds()
        
        if method == 'differential_evolution':
            # استخدام التطور التفاضلي للتحسين العام
            result = differential_evolution(
                objective,
                bounds,
                maxiter=max_iter,
                popsize=15,
                seed=42,
                disp=verbose,
                atol=1e-8,
                tol=1e-8
            )
        else:
            # استخدام التحسين المحلي
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'disp': verbose}
            )
        
        # تحديث النموذج بأفضل معاملات
        self.set_params_from_vector(result.x)
        self.best_params = result.x
        self.trained = True
        
        if verbose:
            print(f"✅ انتهى التحسين. أفضل خطأ: {result.fun:.8f}")
            print(f"📊 عدد التكرارات: {len(self.training_history)}")
        
        return result
    
    def predict_primes(self, x_range, threshold=0.5):
        """توقع الأعداد الأولية في نطاق معين"""
        x_vals = np.arange(x_range[0], x_range[1] + 1)
        predictions = self.evaluate(x_vals)
        
        # تحويل إلى توقعات ثنائية
        binary_predictions = (predictions > threshold).astype(int)
        
        # استخراج الأعداد المتوقعة كأولية
        predicted_primes = x_vals[binary_predictions == 1]
        
        return predicted_primes, predictions, binary_predictions
    
    def plot_training_history(self):
        """رسم تاريخ التدريب"""
        if not self.training_history:
            print("⚠️ لا يوجد تاريخ تدريب للعرض")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history)
        plt.title('تاريخ التدريب - تطور دالة الخطأ')
        plt.xlabel('التكرار')
        plt.ylabel('قيمة الخطأ')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_model_summary(self):
        """ملخص النموذج"""
        summary = {
            'num_sigmoid_components': len(self.sigmoid_components),
            'trained': self.trained,
            'training_iterations': len(self.training_history),
            'final_loss': self.training_history[-1] if self.training_history else None,
            'components': []
        }
        
        for i, comp in enumerate(self.sigmoid_components):
            comp_summary = {
                'component_id': i,
                'alpha': comp['alpha'],
                'n': comp['n'],
                'z': comp['z'],
                'x0': comp['x0']
            }
            summary['components'].append(comp_summary)
        
        return summary
