#!/usr/bin/env python3
"""
النظريات الثلاث الأساسية المستوحاة من نظام Baserah
مطبقة على مشروع GSE للأعداد الأولية

المطور الأصلي للنظريات: باسل يحيى عبدالله
التطبيق على GSE: فريق مشروع GSE

النظريات الثلاث:
1. نظرية التوازن (ثنائية الصفر)
2. نظرية التعامد في التحسين  
3. نظرية الفتائل (ربط المكونات)
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional
from scipy.optimize import minimize
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZeroDualityTheory:
    """
    نظرية التوازن (ثنائية الصفر)
    
    المبدأ: لكل قوة قوة مضادة، والتوازن المثالي يحدث عند نقطة الصفر
    التطبيق: توازن المعاملات وتحسين الاستقرار
    """
    
    def __init__(self, balance_sensitivity: float = 1.0):
        self.balance_sensitivity = balance_sensitivity
        self.balance_history = []
        
    def calculate_balance_point(self, positive_force: float, negative_force: float) -> float:
        """حساب نقطة التوازن بين قوتين متضادتين"""
        balance_difference = positive_force - negative_force
        
        # تطبيق sigmoid للحصول على قيمة متوازنة
        balance_point = 1 / (1 + np.exp(-self.balance_sensitivity * balance_difference))
        
        self.balance_history.append(balance_point)
        return balance_point
    
    def apply_zero_duality_balance(self, values: np.ndarray) -> np.ndarray:
        """تطبيق التوازن على مجموعة من القيم"""
        if len(values) == 0:
            return values
            
        # فصل القيم الموجبة والسالبة
        positive_values = values[values > 0]
        negative_values = values[values < 0]
        
        if len(positive_values) == 0 or len(negative_values) == 0:
            return values
            
        # حساب القوى
        positive_force = np.sum(positive_values)
        negative_force = np.sum(np.abs(negative_values))
        
        # حساب عامل التوازن
        balance_factor = self.calculate_balance_point(positive_force, negative_force)
        
        # تطبيق التوازن
        balanced_values = values * balance_factor
        
        logger.info(f"تطبيق التوازن: عامل التوازن = {balance_factor:.4f}")
        return balanced_values
    
    def balance_coefficients(self, alpha_values: List[float], 
                           target_balance: float = 0.5) -> List[float]:
        """توازن معاملات النموذج"""
        alpha_array = np.array(alpha_values)
        
        # حساب الانحراف عن التوازن المطلوب
        current_balance = np.mean(alpha_array)
        balance_deviation = target_balance - current_balance
        
        # تطبيق تصحيح التوازن
        balance_correction = self.calculate_balance_point(
            abs(balance_deviation), 
            1.0 - abs(balance_deviation)
        )
        
        # تطبيق التصحيح
        balanced_alphas = alpha_array + balance_deviation * balance_correction
        
        return balanced_alphas.tolist()

class PerpendicularOptimizationTheory:
    """
    نظرية التعامد في التحسين
    
    المبدأ: استخدام اتجاهات متعامدة في البحث عن الحل الأمثل
    التطبيق: تحسين متعدد الاتجاهات وتجنب الحد الأدنى المحلي
    """
    
    def __init__(self, perpendicular_strength: float = 0.3):
        self.perpendicular_strength = perpendicular_strength
        self.optimization_history = []
        
    def calculate_perpendicular_vector(self, gradient: np.ndarray) -> np.ndarray:
        """حساب متجه متعامد على التدرج"""
        if len(gradient) < 2:
            return np.zeros_like(gradient)
            
        # إنشاء متجه متعامد باستخدام دوران 90 درجة
        perpendicular = np.zeros_like(gradient)
        
        if len(gradient) == 2:
            # في البعد الثاني: (x, y) -> (-y, x)
            perpendicular[0] = -gradient[1]
            perpendicular[1] = gradient[0]
        else:
            # في الأبعاد الأعلى: استخدام طريقة Gram-Schmidt
            # إنشاء متجه عشوائي
            random_vector = np.random.randn(len(gradient))
            
            # إزالة المكون الموازي للتدرج
            parallel_component = np.dot(random_vector, gradient) / np.dot(gradient, gradient)
            perpendicular = random_vector - parallel_component * gradient
            
            # تطبيع المتجه
            norm = np.linalg.norm(perpendicular)
            if norm > 1e-10:
                perpendicular = perpendicular / norm
        
        return perpendicular
    
    def perpendicular_optimization_step(self, current_params: np.ndarray, 
                                      gradient: np.ndarray,
                                      learning_rate: float = 0.01) -> np.ndarray:
        """خطوة تحسين باستخدام الاتجاه المتعامد"""
        
        # الاتجاه الأساسي (التدرج العادي)
        primary_direction = -gradient
        
        # الاتجاه المتعامد
        perpendicular_direction = self.calculate_perpendicular_vector(gradient)
        
        # دمج الاتجاهين
        combined_direction = (
            (1 - self.perpendicular_strength) * primary_direction + 
            self.perpendicular_strength * perpendicular_direction
        )
        
        # تطبيق خطوة التحسين
        new_params = current_params + learning_rate * combined_direction
        
        self.optimization_history.append({
            'primary_direction': primary_direction,
            'perpendicular_direction': perpendicular_direction,
            'combined_direction': combined_direction
        })
        
        logger.info(f"خطوة تحسين متعامدة: قوة التعامد = {self.perpendicular_strength}")
        return new_params
    
    def escape_local_minimum(self, params: np.ndarray, 
                           loss_history: List[float],
                           threshold: float = 1e-6) -> np.ndarray:
        """الهروب من الحد الأدنى المحلي باستخدام القفزة المتعامدة"""
        
        if len(loss_history) < 5:
            return params
            
        # فحص الجمود في التحسين
        recent_losses = loss_history[-5:]
        loss_variance = np.var(recent_losses)
        
        if loss_variance < threshold:
            logger.info("تم اكتشاف حد أدنى محلي، تطبيق قفزة متعامدة")
            
            # إنشاء اتجاه عشوائي متعامد
            random_direction = np.random.randn(len(params))
            perpendicular_jump = self.calculate_perpendicular_vector(random_direction)
            
            # تطبيق القفزة
            jump_magnitude = 0.1 * np.linalg.norm(params)
            escaped_params = params + jump_magnitude * perpendicular_jump
            
            return escaped_params
        
        return params

class FilamentConnectionTheory:
    """
    نظرية الفتائل (ربط المكونات)
    
    المبدأ: ربط مكونات النموذج بطريقة ذكية لتحسين الأداء الإجمالي
    التطبيق: تحسين التعاون بين مكونات السيجمويد
    """
    
    def __init__(self, connection_strength: float = 0.1):
        self.connection_strength = connection_strength
        self.connection_matrix = None
        self.component_interactions = {}
        
    def calculate_component_similarity(self, component1: Dict, component2: Dict) -> float:
        """حساب التشابه بين مكونين"""
        
        # استخراج المعاملات
        alpha1, k1, x01 = component1.get('alpha', 1.0), component1.get('k', 1.0), component1.get('x0', 0.0)
        alpha2, k2, x02 = component2.get('alpha', 1.0), component2.get('k', 1.0), component2.get('x0', 0.0)
        
        # حساب المسافة في فضاء المعاملات
        param_distance = np.sqrt((alpha1 - alpha2)**2 + (k1 - k2)**2 + (x01 - x02)**2)
        
        # تحويل المسافة إلى تشابه
        similarity = np.exp(-param_distance)
        
        return similarity
    
    def build_connection_matrix(self, components: List[Dict]) -> np.ndarray:
        """بناء مصفوفة الاتصالات بين المكونات"""
        
        n_components = len(components)
        self.connection_matrix = np.zeros((n_components, n_components))
        
        for i in range(n_components):
            for j in range(n_components):
                if i != j:
                    similarity = self.calculate_component_similarity(components[i], components[j])
                    self.connection_matrix[i][j] = similarity
        
        logger.info(f"تم بناء مصفوفة الاتصالات: {n_components}x{n_components}")
        return self.connection_matrix
    
    def apply_filament_enhancement(self, components: List[Dict]) -> List[Dict]:
        """تطبيق تحسين الفتائل على المكونات"""
        
        if len(components) < 2:
            return components
            
        # بناء مصفوفة الاتصالات
        self.build_connection_matrix(components)
        
        enhanced_components = []
        
        for i, component in enumerate(components):
            # حساب قوة الاتصال الإجمالية
            total_connection_strength = np.sum(self.connection_matrix[i, :])
            
            # تطبيق التحسين
            enhancement_factor = 1 + self.connection_strength * total_connection_strength
            
            enhanced_component = component.copy()
            enhanced_component['alpha'] = component.get('alpha', 1.0) * enhancement_factor
            
            enhanced_components.append(enhanced_component)
            
            logger.debug(f"مكون {i}: عامل التحسين = {enhancement_factor:.4f}")
        
        return enhanced_components
    
    def optimize_component_cooperation(self, alpha_values: List[float], 
                                     k_values: List[float], 
                                     x0_values: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """تحسين التعاون بين معاملات المكونات"""
        
        n_components = len(alpha_values)
        if n_components < 2:
            return alpha_values, k_values, x0_values
            
        # إنشاء مصفوفة التفاعل
        interaction_matrix = np.zeros((n_components, n_components))
        
        for i in range(n_components):
            for j in range(n_components):
                if i != j:
                    # حساب قوة التفاعل بين المعاملات
                    alpha_interaction = abs(alpha_values[i] - alpha_values[j])
                    k_interaction = abs(k_values[i] - k_values[j])
                    x0_interaction = abs(x0_values[i] - x0_values[j])
                    
                    total_interaction = np.exp(-(alpha_interaction + k_interaction + x0_interaction))
                    interaction_matrix[i][j] = total_interaction
        
        # تطبيق التحسين التعاوني
        cooperation_factors = 1 + self.connection_strength * np.sum(interaction_matrix, axis=1)
        
        enhanced_alphas = [alpha_values[i] * cooperation_factors[i] for i in range(n_components)]
        
        logger.info(f"تحسين التعاون: متوسط عامل التعاون = {np.mean(cooperation_factors):.4f}")
        
        return enhanced_alphas, k_values, x0_values

class ThreeTheoriesIntegrator:
    """
    مدمج النظريات الثلاث
    
    يدمج النظريات الثلاث في نظام موحد لتحسين نموذج GSE
    """
    
    def __init__(self, balance_sensitivity: float = 1.0,
                 perpendicular_strength: float = 0.3,
                 connection_strength: float = 0.1):
        
        self.zero_duality = ZeroDualityTheory(balance_sensitivity)
        self.perpendicular_opt = PerpendicularOptimizationTheory(perpendicular_strength)
        self.filament_connection = FilamentConnectionTheory(connection_strength)
        
        self.integration_history = []
        
    def integrated_optimization_step(self, current_params: np.ndarray,
                                   gradient: np.ndarray,
                                   components: List[Dict],
                                   learning_rate: float = 0.01) -> Tuple[np.ndarray, List[Dict]]:
        """خطوة تحسين متكاملة باستخدام النظريات الثلاث"""
        
        logger.info("بدء خطوة التحسين المتكاملة")
        
        # 1. تطبيق نظرية التوازن على المعاملات
        balanced_params = self.zero_duality.apply_zero_duality_balance(current_params)
        
        # 2. تطبيق التحسين المتعامد
        optimized_params = self.perpendicular_opt.perpendicular_optimization_step(
            balanced_params, gradient, learning_rate
        )
        
        # 3. تطبيق تحسين الفتائل على المكونات
        enhanced_components = self.filament_connection.apply_filament_enhancement(components)
        
        # تسجيل النتائج
        self.integration_history.append({
            'balance_applied': True,
            'perpendicular_applied': True,
            'filament_applied': True,
            'param_change': np.linalg.norm(optimized_params - current_params)
        })
        
        logger.info("انتهاء خطوة التحسين المتكاملة")
        
        return optimized_params, enhanced_components
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التكامل"""
        
        if not self.integration_history:
            return {}
            
        total_steps = len(self.integration_history)
        avg_param_change = np.mean([step['param_change'] for step in self.integration_history])
        
        return {
            'total_integration_steps': total_steps,
            'average_parameter_change': avg_param_change,
            'balance_applications': total_steps,
            'perpendicular_applications': total_steps,
            'filament_applications': total_steps,
            'zero_duality_balance_history': self.zero_duality.balance_history,
            'perpendicular_optimization_history': len(self.perpendicular_opt.optimization_history)
        }

if __name__ == "__main__":
    # اختبار سريع للنظريات
    print("🧪 اختبار النظريات الثلاث الأساسية")
    
    # اختبار نظرية التوازن
    zero_duality = ZeroDualityTheory()
    test_values = np.array([1.5, -0.8, 2.1, -1.2, 0.9])
    balanced = zero_duality.apply_zero_duality_balance(test_values)
    print(f"التوازن: {test_values} -> {balanced}")
    
    # اختبار التحسين المتعامد
    perpendicular = PerpendicularOptimizationTheory()
    test_gradient = np.array([1.0, 0.5])
    perp_vector = perpendicular.calculate_perpendicular_vector(test_gradient)
    print(f"المتجه المتعامد: {test_gradient} -> {perp_vector}")
    
    # اختبار الفتائل
    filament = FilamentConnectionTheory()
    test_components = [
        {'alpha': 1.0, 'k': 1.0, 'x0': 0.0},
        {'alpha': 1.5, 'k': 0.8, 'x0': 0.5}
    ]
    enhanced = filament.apply_filament_enhancement(test_components)
    print(f"تحسين الفتائل: {len(enhanced)} مكونات محسنة")
    
    print("✅ اختبار النظريات مكتمل!")
