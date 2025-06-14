#!/usr/bin/env python3
"""
نموذج GSE المحسن بالنظريات الثلاث والذكاء التكيفي
يدمج جميع التحسينات المستوحاة من نظام Baserah

الميزات الجديدة:
- تطبيق النظريات الثلاث (التوازن، التعامد، الفتائل)
- معادلات متكيفة تتطور تلقائياً
- نظام خبير/مستكشف للتحليل والاستكشاف
- تحسين ذكي متعدد المستويات
- حفظ وتتبع تاريخ التطوير
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import json
import pickle

try:
    from .gse_advanced_model import AdvancedGSEModel
    from .three_theories_core import ThreeTheoriesIntegrator
    from .adaptive_equations import AdaptiveGSEEquation, AdaptationDirection, AdaptationConfig
    from .expert_explorer_system import IntegratedExpertExplorer, ExplorationConfig
except ImportError:
    from gse_advanced_model import AdvancedGSEModel
    from three_theories_core import ThreeTheoriesIntegrator
    from adaptive_equations import AdaptiveGSEEquation, AdaptationDirection, AdaptationConfig
    from expert_explorer_system import IntegratedExpertExplorer, ExplorationConfig

logger = logging.getLogger(__name__)

class EnhancedGSEModel(AdvancedGSEModel):
    """
    نموذج GSE المحسن بالذكاء التكيفي والنظريات الثلاث
    
    يجمع بين:
    - النموذج الأساسي GSE
    - النظريات الثلاث المستوحاة من Baserah
    - المعادلات المتكيفة
    - نظام الخبير/المستكشف
    """
    
    def __init__(self, adaptation_config: AdaptationConfig = None,
                 exploration_config: ExplorationConfig = None,
                 enable_theories: bool = True):
        
        # تهيئة النموذج الأساسي
        super().__init__()
        
        # إعدادات التحسين
        self.adaptation_config = adaptation_config or AdaptationConfig()
        self.exploration_config = exploration_config or ExplorationConfig()
        self.enable_theories = enable_theories
        
        # النظريات الثلاث
        if self.enable_theories:
            self.theories_integrator = ThreeTheoriesIntegrator()
        
        # المعادلات المتكيفة
        self.adaptive_equations: List[AdaptiveGSEEquation] = []
        self.primary_adaptive_equation: Optional[AdaptiveGSEEquation] = None
        
        # نظام الخبير/المستكشف
        self.expert_explorer = IntegratedExpertExplorer(exploration_config)
        
        # تاريخ التطوير
        self.enhancement_history = []
        self.performance_timeline = []
        
        # إحصائيات التحسين
        self.total_enhancements = 0
        self.successful_enhancements = 0
        self.adaptation_cycles = 0
        
        # حالة النموذج
        self.is_enhanced = False
        self.enhancement_level = 0
        self.best_performance = float('inf')
        
        logger.info("تم تهيئة نموذج GSE المحسن بالذكاء التكيفي")

    def add_sigmoid_component(self, alpha: float = 1.0, k: float = 1.0, x0: float = 0.0):
        """إضافة مكون سيجمويد للنموذج"""

        # إضافة مباشرة للمصفوفات
        if not hasattr(self, 'alpha_values') or self.alpha_values is None:
            self.alpha_values = np.array([])
            self.k_values = np.array([])
            self.x0_values = np.array([])

        self.alpha_values = np.append(self.alpha_values, alpha)
        self.k_values = np.append(self.k_values, k)
        self.x0_values = np.append(self.x0_values, x0)

        logger.debug(f"أضيف مكون سيجمويد: α={alpha}, k={k}, x0={x0}")

    def add_linear_component(self, beta: float = 1.0, gamma: float = 0.0):
        """إضافة مكون خطي للنموذج"""

        # إنشاء مكون خطي
        if not hasattr(self, 'linear_components'):
            self.linear_components = []

        linear_component = {'beta': beta, 'gamma': gamma}
        self.linear_components.append(linear_component)

        logger.debug(f"أضيف مكون خطي: β={beta}, γ={gamma}")

    def calculate_loss(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """حساب خطأ النموذج"""

        try:
            # استخدام دالة التقييم من النموذج الأساسي
            y_pred = self.evaluate(x_data)
            mse = np.mean((y_data - y_pred) ** 2)
            return mse
        except Exception as e:
            logger.error(f"خطأ في حساب الخطأ: {e}")
            return float('inf')

    def _estimate_parameter_gradient(self, x_data: np.ndarray, y_data: np.ndarray,
                                   epsilon: float = 1e-6) -> np.ndarray:
        """تقدير التدرج للمعاملات"""

        gradient = []
        current_loss = self.calculate_loss(x_data, y_data)

        # تدرج معاملات ألفا
        for i in range(len(self.alpha_values)):
            original_value = self.alpha_values[i]

            self.alpha_values[i] = original_value + epsilon
            new_loss = self.calculate_loss(x_data, y_data)

            grad = (new_loss - current_loss) / epsilon
            gradient.append(grad)

            self.alpha_values[i] = original_value

        # تدرج معاملات k
        for i in range(len(self.k_values)):
            original_value = self.k_values[i]

            self.k_values[i] = original_value + epsilon
            new_loss = self.calculate_loss(x_data, y_data)

            grad = (new_loss - current_loss) / epsilon
            gradient.append(grad)

            self.k_values[i] = original_value

        # تدرج معاملات x0
        for i in range(len(self.x0_values)):
            original_value = self.x0_values[i]

            self.x0_values[i] = original_value + epsilon
            new_loss = self.calculate_loss(x_data, y_data)

            grad = (new_loss - current_loss) / epsilon
            gradient.append(grad)

            self.x0_values[i] = original_value

        return np.array(gradient)

    def create_adaptive_equation_from_current_model(self) -> AdaptiveGSEEquation:
        """إنشاء معادلة متكيفة من النموذج الحالي"""
        
        adaptive_eq = AdaptiveGSEEquation(adaptation_config=self.adaptation_config)
        
        # تحويل مكونات النموذج الحالي إلى مكونات متكيفة
        for i in range(len(self.alpha_values)):
            adaptive_eq.add_sigmoid_component(
                alpha=self.alpha_values[i],
                k=self.k_values[i],
                x0=self.x0_values[i],
                adaptive=True
            )
        
        # إضافة مكونات خطية إذا وجدت
        if hasattr(self, 'linear_components') and self.linear_components:
            for linear_comp in self.linear_components:
                adaptive_eq.add_linear_component(
                    beta=linear_comp.get('beta', 1.0),
                    gamma=linear_comp.get('gamma', 0.0),
                    adaptive=True
                )
        
        self.adaptive_equations.append(adaptive_eq)
        
        if self.primary_adaptive_equation is None:
            self.primary_adaptive_equation = adaptive_eq
        
        logger.info(f"تم إنشاء معادلة متكيفة بـ {len(adaptive_eq.components)} مكونات")
        
        return adaptive_eq
    
    def enhance_with_three_theories(self, x_data: np.ndarray, y_data: np.ndarray,
                                  max_enhancement_cycles: int = 5) -> Dict[str, Any]:
        """تحسين النموذج باستخدام النظريات الثلاث"""
        
        if not self.enable_theories:
            logger.warning("النظريات الثلاث غير مفعلة")
            return {'success': False, 'reason': 'theories_disabled'}
        
        logger.info("بدء التحسين بالنظريات الثلاث")
        
        enhancement_results = []
        initial_performance = self.calculate_loss(x_data, y_data)
        best_performance = initial_performance
        
        for cycle in range(max_enhancement_cycles):
            logger.info(f"دورة التحسين {cycle + 1}/{max_enhancement_cycles}")
            
            cycle_start_performance = self.calculate_loss(x_data, y_data)
            
            # 1. تطبيق نظرية التوازن
            balance_improvement = self._apply_balance_theory_enhancement(x_data, y_data)
            
            # 2. تطبيق نظرية التعامد
            perpendicular_improvement = self._apply_perpendicular_theory_enhancement(x_data, y_data)
            
            # 3. تطبيق نظرية الفتائل
            filament_improvement = self._apply_filament_theory_enhancement(x_data, y_data)
            
            cycle_end_performance = self.calculate_loss(x_data, y_data)
            cycle_improvement = cycle_start_performance - cycle_end_performance
            
            cycle_result = {
                'cycle': cycle + 1,
                'balance_improvement': balance_improvement,
                'perpendicular_improvement': perpendicular_improvement,
                'filament_improvement': filament_improvement,
                'total_cycle_improvement': cycle_improvement,
                'performance_after_cycle': cycle_end_performance
            }
            
            enhancement_results.append(cycle_result)
            
            if cycle_end_performance < best_performance:
                best_performance = cycle_end_performance
                self.best_performance = best_performance
            
            # فحص التقارب
            if abs(cycle_improvement) < 1e-8:
                logger.info(f"تقارب في الدورة {cycle + 1}")
                break
            
            self.adaptation_cycles += 1
        
        total_improvement = initial_performance - best_performance
        
        final_result = {
            'success': True,
            'total_improvement': total_improvement,
            'improvement_percentage': (total_improvement / initial_performance) * 100,
            'cycles_completed': len(enhancement_results),
            'best_performance': best_performance,
            'enhancement_results': enhancement_results,
            'theories_applied': ['balance', 'perpendicular', 'filament']
        }
        
        self.enhancement_history.append(final_result)
        self.is_enhanced = True
        self.enhancement_level += 1
        
        if total_improvement > 0:
            self.successful_enhancements += 1
        
        self.total_enhancements += 1
        
        logger.info(f"انتهى التحسين: تحسن إجمالي = {total_improvement:.6f} ({(total_improvement/initial_performance)*100:.2f}%)")
        
        return final_result
    
    def _apply_balance_theory_enhancement(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """تطبيق نظرية التوازن لتحسين النموذج"""
        
        initial_loss = self.calculate_loss(x_data, y_data)
        
        # توازن معاملات ألفا
        balanced_alphas = self.theories_integrator.zero_duality.balance_coefficients(
            self.alpha_values, target_balance=0.5
        )
        
        # تطبيق التوازن
        old_alphas = self.alpha_values.copy()
        self.alpha_values = np.array(balanced_alphas)
        
        new_loss = self.calculate_loss(x_data, y_data)
        improvement = initial_loss - new_loss
        
        # إذا لم يحدث تحسن، أرجع القيم القديمة
        if improvement <= 0:
            self.alpha_values = old_alphas
            improvement = 0
        
        logger.debug(f"تحسين التوازن: {improvement:.6f}")
        return improvement
    
    def _apply_perpendicular_theory_enhancement(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """تطبيق نظرية التعامد لتحسين النموذج"""
        
        initial_loss = self.calculate_loss(x_data, y_data)
        
        # حساب التدرج التقريبي
        gradient = self._estimate_parameter_gradient(x_data, y_data)
        
        if len(gradient) == 0:
            return 0.0
        
        # تطبيق خطوة تحسين متعامدة
        current_params = np.concatenate([self.alpha_values, self.k_values, self.x0_values])
        
        if len(gradient) == len(current_params):
            new_params = self.theories_integrator.perpendicular_opt.perpendicular_optimization_step(
                current_params, gradient, learning_rate=0.01
            )
            
            # تحديث المعاملات
            n_components = len(self.alpha_values)
            old_params = current_params.copy()
            
            self.alpha_values = np.maximum(0.1, new_params[:n_components])  # تجنب القيم السالبة
            self.k_values = np.maximum(0.1, new_params[n_components:2*n_components])
            self.x0_values = new_params[2*n_components:3*n_components]
            
            new_loss = self.calculate_loss(x_data, y_data)
            improvement = initial_loss - new_loss
            
            # إذا لم يحدث تحسن، أرجع القيم القديمة
            if improvement <= 0:
                self.alpha_values = old_params[:n_components]
                self.k_values = old_params[n_components:2*n_components]
                self.x0_values = old_params[2*n_components:3*n_components]
                improvement = 0
            
            logger.debug(f"تحسين التعامد: {improvement:.6f}")
            return improvement
        
        return 0.0
    
    def _apply_filament_theory_enhancement(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """تطبيق نظرية الفتائل لتحسين النموذج"""
        
        initial_loss = self.calculate_loss(x_data, y_data)
        
        # إنشاء مكونات للفتائل
        components = []
        for i in range(len(self.alpha_values)):
            component = {
                'alpha': self.alpha_values[i],
                'k': self.k_values[i],
                'x0': self.x0_values[i]
            }
            components.append(component)
        
        if len(components) < 2:
            return 0.0
        
        # تطبيق تحسين الفتائل
        enhanced_alphas, enhanced_k, enhanced_x0 = self.theories_integrator.filament_connection.optimize_component_cooperation(
            self.alpha_values.tolist(), self.k_values.tolist(), self.x0_values.tolist()
        )
        
        # حفظ القيم القديمة
        old_alphas = self.alpha_values.copy()
        old_k = self.k_values.copy()
        old_x0 = self.x0_values.copy()
        
        # تطبيق التحسين
        self.alpha_values = np.array(enhanced_alphas)
        self.k_values = np.array(enhanced_k)
        self.x0_values = np.array(enhanced_x0)
        
        new_loss = self.calculate_loss(x_data, y_data)
        improvement = initial_loss - new_loss
        
        # إذا لم يحدث تحسن، أرجع القيم القديمة
        if improvement <= 0:
            self.alpha_values = old_alphas
            self.k_values = old_k
            self.x0_values = old_x0
            improvement = 0
        
        logger.debug(f"تحسين الفتائل: {improvement:.6f}")
        return improvement
    
    def intelligent_adaptive_optimization(self, x_data: np.ndarray, y_data: np.ndarray,
                                        max_iterations: int = 10) -> Dict[str, Any]:
        """تحسين ذكي تكيفي شامل"""
        
        logger.info("بدء التحسين الذكي التكيفي الشامل")
        
        # 1. إنشاء معادلة متكيفة من النموذج الحالي
        adaptive_eq = self.create_adaptive_equation_from_current_model()
        
        # 2. تطبيق التحسين الذكي بنظام الخبير/المستكشف
        expert_result = self.expert_explorer.intelligent_optimization(
            adaptive_eq, x_data, y_data, max_iterations
        )
        
        # 3. تطبيق النظريات الثلاث للتحسين الإضافي
        theories_result = self.enhance_with_three_theories(x_data, y_data)
        
        # 4. دمج النتائج وتحديث النموذج
        if expert_result['best_equation']:
            self._update_model_from_adaptive_equation(expert_result['best_equation'])
        
        # 5. حساب التحسن الإجمالي
        final_performance = self.calculate_loss(x_data, y_data)
        
        comprehensive_result = {
            'success': True,
            'final_performance': final_performance,
            'expert_explorer_result': expert_result,
            'theories_enhancement_result': theories_result,
            'adaptive_equations_created': len(self.adaptive_equations),
            'enhancement_level': self.enhancement_level,
            'total_enhancements': self.total_enhancements,
            'successful_enhancements': self.successful_enhancements,
            'success_rate': self.successful_enhancements / max(1, self.total_enhancements)
        }
        
        self.performance_timeline.append({
            'timestamp': datetime.now(),
            'performance': final_performance,
            'enhancement_type': 'comprehensive_intelligent_adaptive'
        })
        
        logger.info(f"انتهى التحسين الشامل: أداء نهائي = {final_performance:.6f}")
        
        return comprehensive_result
    
    def _update_model_from_adaptive_equation(self, adaptive_eq: AdaptiveGSEEquation):
        """تحديث النموذج من معادلة متكيفة"""
        
        sigmoid_components = [comp for comp in adaptive_eq.components if comp['type'] == 'sigmoid']
        
        if sigmoid_components:
            self.alpha_values = np.array([comp['alpha'] for comp in sigmoid_components])
            self.k_values = np.array([comp['k'] for comp in sigmoid_components])
            self.x0_values = np.array([comp['x0'] for comp in sigmoid_components])
            
            logger.info(f"تم تحديث النموذج من معادلة متكيفة: {len(sigmoid_components)} مكونات")

if __name__ == "__main__":
    # اختبار النموذج المحسن
    print("🧪 اختبار النموذج المحسن GSE")
    
    # إنشاء النموذج المحسن
    enhanced_model = EnhancedGSEModel()
    
    # إضافة مكونات أساسية
    enhanced_model.add_sigmoid_component(alpha=1.0, k=1.0, x0=0.0)
    enhanced_model.add_sigmoid_component(alpha=0.8, k=0.5, x0=2.0)
    
    # بيانات اختبار (الأعداد الأولية الأولى)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    x_data = np.array(range(1, len(primes) + 1))
    y_data = np.array([1 if i+1 in primes else 0 for i in range(len(primes))])
    
    print(f"أداء أولي: {enhanced_model.calculate_loss(x_data, y_data):.6f}")
    
    # تطبيق التحسين الشامل
    result = enhanced_model.intelligent_adaptive_optimization(x_data, y_data, max_iterations=5)
    
    print(f"أداء نهائي: {result['final_performance']:.6f}")
    print(f"معدل نجاح التحسينات: {result['success_rate']:.2%}")
    print(f"مستوى التحسين: {result['enhancement_level']}")
    
    print("✅ اختبار النموذج المحسن مكتمل!")
