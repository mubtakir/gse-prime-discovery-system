#!/usr/bin/env python3
"""
المعادلات المتكيفة لنموذج GSE
مستوحاة من نظام Baserah مع تطبيق النظريات الثلاث

الميزات:
- معادلات تتكيف مع البيانات تلقائياً
- تطبيق النظريات الثلاث في التكيف
- تتبع تاريخ التطوير والتحسين
- قدرة على التعلم من الأخطاء
"""

import numpy as np
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from .three_theories_core import ThreeTheoriesIntegrator
    from .gse_advanced_model import AdvancedGSEModel
except ImportError:
    from three_theories_core import ThreeTheoriesIntegrator
    try:
        from gse_advanced_model import AdvancedGSEModel
    except ImportError:
        # إذا لم يكن متوفر، استخدم كلاس بديل
        class AdvancedGSEModel:
            pass

logger = logging.getLogger(__name__)

class AdaptationDirection(Enum):
    """اتجاهات التكيف"""
    IMPROVE_ACCURACY = "improve_accuracy"
    REDUCE_COMPLEXITY = "reduce_complexity"
    BALANCE_BOTH = "balance_both"
    EXPLORE_NEW = "explore_new"

@dataclass
class AdaptationConfig:
    """إعدادات التكيف"""
    adaptation_rate: float = 0.01
    max_adaptations: int = 100
    convergence_threshold: float = 1e-6
    exploration_probability: float = 0.1
    balance_weight: float = 0.5

@dataclass
class AdaptationHistory:
    """تاريخ التكيف"""
    timestamp: datetime
    adaptation_type: str
    parameters_before: Dict[str, Any]
    parameters_after: Dict[str, Any]
    performance_before: float
    performance_after: float
    improvement: float

class AdaptiveGSEEquation:
    """
    معادلة GSE متكيفة
    
    تتكيف تلقائياً مع البيانات باستخدام النظريات الثلاث:
    - التوازن: توازن المعاملات
    - التعامد: استكشاف اتجاهات جديدة
    - الفتائل: ربط المكونات
    """
    
    def __init__(self, equation_id: str = None, 
                 initial_components: List[Dict] = None,
                 adaptation_config: AdaptationConfig = None):
        
        self.equation_id = equation_id or str(uuid.uuid4())[:8]
        self.creation_time = datetime.now()
        
        # إعدادات التكيف
        self.config = adaptation_config or AdaptationConfig()
        
        # النظريات الثلاث
        self.theories_integrator = ThreeTheoriesIntegrator()
        
        # مكونات المعادلة
        self.components = initial_components or []
        self.adaptive_weights = []
        
        # تاريخ التكيف
        self.adaptation_history: List[AdaptationHistory] = []
        self.performance_history: List[float] = []
        
        # حالة التكيف
        self.adaptation_count = 0
        self.is_converged = False
        self.best_performance = float('inf')
        self.best_parameters = None
        
        # إحصائيات
        self.successful_adaptations = 0
        self.failed_adaptations = 0
        
        logger.info(f"تم إنشاء معادلة متكيفة: {self.equation_id}")
    
    def add_sigmoid_component(self, alpha: float = 1.0, k: float = 1.0, 
                            x0: float = 0.0, adaptive: bool = True):
        """إضافة مكون سيجمويد متكيف"""
        
        component = {
            'type': 'sigmoid',
            'alpha': alpha,
            'k': k,
            'x0': x0,
            'adaptive': adaptive,
            'adaptation_rate': self.config.adaptation_rate,
            'creation_time': datetime.now()
        }
        
        self.components.append(component)
        self.adaptive_weights.append(1.0)
        
        logger.debug(f"أضيف مكون سيجمويد: α={alpha}, k={k}, x0={x0}")
    
    def add_linear_component(self, beta: float = 1.0, gamma: float = 0.0, 
                           adaptive: bool = True):
        """إضافة مكون خطي متكيف"""
        
        component = {
            'type': 'linear',
            'beta': beta,
            'gamma': gamma,
            'adaptive': adaptive,
            'adaptation_rate': self.config.adaptation_rate,
            'creation_time': datetime.now()
        }
        
        self.components.append(component)
        self.adaptive_weights.append(1.0)
        
        logger.debug(f"أضيف مكون خطي: β={beta}, γ={gamma}")
    
    def evaluate(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """تقييم المعادلة المتكيفة"""

        if len(self.components) == 0:
            return np.zeros_like(x, dtype=float) if isinstance(x, np.ndarray) else 0.0

        result = np.zeros_like(x, dtype=float) if isinstance(x, np.ndarray) else 0.0
        
        for i, component in enumerate(self.components):
            weight = self.adaptive_weights[i]
            
            if component['type'] == 'sigmoid':
                # مكون سيجمويد
                alpha = component['alpha']
                k = component['k']
                x0 = component['x0']
                
                sigmoid_value = alpha / (1 + np.exp(-k * (x - x0)))
                result += weight * sigmoid_value
                
            elif component['type'] == 'linear':
                # مكون خطي
                beta = component['beta']
                gamma = component['gamma']
                
                linear_value = beta * x + gamma
                result += weight * linear_value
        
        return result
    
    def calculate_error(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """حساب خطأ المعادلة"""
        
        y_pred = self.evaluate(x_data)
        mse = np.mean((y_data - y_pred) ** 2)
        
        return mse
    
    def adapt_to_data(self, x_data: np.ndarray, y_data: np.ndarray, 
                     direction: AdaptationDirection = AdaptationDirection.IMPROVE_ACCURACY) -> bool:
        """تكيف المعادلة مع البيانات"""
        
        if self.adaptation_count >= self.config.max_adaptations:
            logger.warning(f"وصل عدد التكيفات للحد الأقصى: {self.config.max_adaptations}")
            return False
        
        if self.is_converged:
            logger.info("المعادلة متقاربة، لا حاجة للتكيف")
            return False
        
        # حساب الأداء الحالي
        current_performance = self.calculate_error(x_data, y_data)
        
        # حفظ المعاملات الحالية
        current_params = self._get_current_parameters()
        
        # تطبيق التكيف باستخدام النظريات الثلاث
        adaptation_success = self._apply_three_theories_adaptation(
            x_data, y_data, direction
        )
        
        if adaptation_success:
            # حساب الأداء الجديد
            new_performance = self.calculate_error(x_data, y_data)
            improvement = current_performance - new_performance
            
            # تسجيل التكيف
            self._record_adaptation(
                direction.value, current_params, 
                current_performance, new_performance, improvement
            )
            
            # تحديث أفضل أداء
            if new_performance < self.best_performance:
                self.best_performance = new_performance
                self.best_parameters = self._get_current_parameters()
            
            # فحص التقارب
            if improvement < self.config.convergence_threshold:
                self.is_converged = True
                logger.info(f"المعادلة متقاربة: تحسن = {improvement:.2e}")
            
            self.successful_adaptations += 1
            return True
        else:
            self.failed_adaptations += 1
            return False
    
    def _apply_three_theories_adaptation(self, x_data: np.ndarray, y_data: np.ndarray,
                                       direction: AdaptationDirection) -> bool:
        """تطبيق النظريات الثلاث في التكيف"""
        
        try:
            # 1. تطبيق نظرية التوازن على المعاملات
            self._apply_balance_theory()
            
            # 2. تطبيق نظرية التعامد في الاستكشاف
            self._apply_perpendicular_exploration(x_data, y_data)
            
            # 3. تطبيق نظرية الفتائل في ربط المكونات
            self._apply_filament_connection()
            
            return True
            
        except Exception as e:
            logger.error(f"خطأ في تطبيق النظريات الثلاث: {e}")
            return False
    
    def _apply_balance_theory(self):
        """تطبيق نظرية التوازن على معاملات المكونات"""
        
        for component in self.components:
            if not component.get('adaptive', True):
                continue
                
            if component['type'] == 'sigmoid':
                # توازن معاملات السيجمويد
                alpha = component['alpha']
                k = component['k']
                
                # تطبيق التوازن
                balanced_alpha = self.theories_integrator.zero_duality.calculate_balance_point(
                    abs(alpha), 1.0
                )
                
                component['alpha'] = alpha * balanced_alpha
                
            elif component['type'] == 'linear':
                # توازن معاملات الخط المستقيم
                beta = component['beta']
                gamma = component['gamma']
                
                # تطبيق التوازن
                balanced_beta = self.theories_integrator.zero_duality.calculate_balance_point(
                    abs(beta), abs(gamma) + 1e-10
                )
                
                component['beta'] = beta * balanced_beta
    
    def _apply_perpendicular_exploration(self, x_data: np.ndarray, y_data: np.ndarray):
        """تطبيق نظرية التعامد في استكشاف معاملات جديدة"""
        
        # حساب التدرج التقريبي
        current_error = self.calculate_error(x_data, y_data)
        gradient = self._estimate_gradient(x_data, y_data)
        
        # تطبيق خطوة استكشاف متعامدة
        for i, component in enumerate(self.components):
            if not component.get('adaptive', True):
                continue
                
            if component['type'] == 'sigmoid':
                # استكشاف متعامد لمعاملات السيجمويد
                param_vector = np.array([component['alpha'], component['k'], component['x0']])
                
                if len(gradient) > i * 3:
                    param_gradient = gradient[i*3:(i+1)*3]
                    
                    new_params = self.theories_integrator.perpendicular_opt.perpendicular_optimization_step(
                        param_vector, param_gradient, self.config.adaptation_rate
                    )
                    
                    component['alpha'] = max(0.1, new_params[0])  # تجنب القيم السالبة
                    component['k'] = max(0.1, new_params[1])
                    component['x0'] = new_params[2]
    
    def _apply_filament_connection(self):
        """تطبيق نظرية الفتائل في ربط المكونات"""
        
        if len(self.components) < 2:
            return
            
        # تحسين المكونات باستخدام الفتائل
        enhanced_components = self.theories_integrator.filament_connection.apply_filament_enhancement(
            self.components
        )
        
        # تحديث المكونات
        for i, enhanced_component in enumerate(enhanced_components):
            if i < len(self.components):
                self.components[i].update(enhanced_component)
        
        # تحسين الأوزان التكيفية
        if len(self.adaptive_weights) == len(self.components):
            connection_matrix = self.theories_integrator.filament_connection.connection_matrix
            
            if connection_matrix is not None:
                # تحديث الأوزان بناءً على قوة الاتصالات
                for i in range(len(self.adaptive_weights)):
                    connection_strength = np.sum(connection_matrix[i, :]) if i < len(connection_matrix) else 0
                    enhancement_factor = 1 + 0.1 * connection_strength
                    self.adaptive_weights[i] *= enhancement_factor
    
    def _estimate_gradient(self, x_data: np.ndarray, y_data: np.ndarray, 
                          epsilon: float = 1e-6) -> np.ndarray:
        """تقدير التدرج عددياً"""
        
        gradient = []
        current_error = self.calculate_error(x_data, y_data)
        
        for component in self.components:
            if component['type'] == 'sigmoid':
                # تدرج معاملات السيجمويد
                for param in ['alpha', 'k', 'x0']:
                    original_value = component[param]
                    
                    # تغيير صغير في المعامل
                    component[param] = original_value + epsilon
                    new_error = self.calculate_error(x_data, y_data)
                    
                    # حساب التدرج
                    grad = (new_error - current_error) / epsilon
                    gradient.append(grad)
                    
                    # إرجاع القيمة الأصلية
                    component[param] = original_value
                    
            elif component['type'] == 'linear':
                # تدرج معاملات الخط المستقيم
                for param in ['beta', 'gamma']:
                    original_value = component[param]
                    
                    component[param] = original_value + epsilon
                    new_error = self.calculate_error(x_data, y_data)
                    
                    grad = (new_error - current_error) / epsilon
                    gradient.append(grad)
                    
                    component[param] = original_value
        
        return np.array(gradient)
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """الحصول على المعاملات الحالية"""
        
        params = {
            'components': [comp.copy() for comp in self.components],
            'adaptive_weights': self.adaptive_weights.copy(),
            'adaptation_count': self.adaptation_count
        }
        
        return params
    
    def _record_adaptation(self, adaptation_type: str, params_before: Dict,
                          performance_before: float, performance_after: float,
                          improvement: float):
        """تسجيل تكيف في التاريخ"""
        
        history_entry = AdaptationHistory(
            timestamp=datetime.now(),
            adaptation_type=adaptation_type,
            parameters_before=params_before,
            parameters_after=self._get_current_parameters(),
            performance_before=performance_before,
            performance_after=performance_after,
            improvement=improvement
        )
        
        self.adaptation_history.append(history_entry)
        self.performance_history.append(performance_after)
        self.adaptation_count += 1
        
        logger.info(f"تكيف مسجل: نوع={adaptation_type}, تحسن={improvement:.6f}")
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التكيف"""
        
        total_adaptations = len(self.adaptation_history)
        success_rate = self.successful_adaptations / max(1, total_adaptations)
        
        avg_improvement = 0.0
        if self.adaptation_history:
            improvements = [h.improvement for h in self.adaptation_history]
            avg_improvement = np.mean(improvements)
        
        return {
            'equation_id': self.equation_id,
            'total_adaptations': total_adaptations,
            'successful_adaptations': self.successful_adaptations,
            'failed_adaptations': self.failed_adaptations,
            'success_rate': success_rate,
            'average_improvement': avg_improvement,
            'best_performance': self.best_performance,
            'is_converged': self.is_converged,
            'components_count': len(self.components),
            'theories_integration_stats': self.theories_integrator.get_integration_statistics()
        }
    
    def reset_to_best(self):
        """إرجاع المعادلة لأفضل معاملات"""
        
        if self.best_parameters is not None:
            self.components = self.best_parameters['components']
            self.adaptive_weights = self.best_parameters['adaptive_weights']
            logger.info(f"تم إرجاع المعادلة لأفضل معاملات: أداء = {self.best_performance:.6f}")

if __name__ == "__main__":
    # اختبار المعادلات المتكيفة
    print("🧪 اختبار المعادلات المتكيفة")
    
    # إنشاء معادلة متكيفة
    adaptive_eq = AdaptiveGSEEquation()
    adaptive_eq.add_sigmoid_component(alpha=1.0, k=1.0, x0=0.0)
    adaptive_eq.add_linear_component(beta=0.5, gamma=0.1)
    
    # بيانات اختبار
    x_data = np.linspace(-5, 5, 100)
    y_target = np.sin(x_data)  # هدف: دالة الجيب
    
    print(f"خطأ أولي: {adaptive_eq.calculate_error(x_data, y_target):.6f}")
    
    # تطبيق التكيف
    for i in range(10):
        success = adaptive_eq.adapt_to_data(x_data, y_target)
        if success:
            current_error = adaptive_eq.calculate_error(x_data, y_target)
            print(f"تكيف {i+1}: خطأ = {current_error:.6f}")
        
        if adaptive_eq.is_converged:
            break
    
    # عرض الإحصائيات
    stats = adaptive_eq.get_adaptation_statistics()
    print(f"\nإحصائيات التكيف:")
    print(f"  تكيفات ناجحة: {stats['successful_adaptations']}")
    print(f"  معدل النجاح: {stats['success_rate']:.2%}")
    print(f"  أفضل أداء: {stats['best_performance']:.6f}")
    
    print("✅ اختبار المعادلات المتكيفة مكتمل!")
