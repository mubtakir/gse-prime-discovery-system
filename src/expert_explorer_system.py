#!/usr/bin/env python3
"""
نظام الخبير/المستكشف لنموذج GSE
مستوحى من نظام Baserah مع تطبيق النظريات الثلاث

الميزات:
- وضع الخبير: تحليل ذكي للأنماط واقتراح أفضل معاملات
- وضع المستكشف: استكشاف أنماط جديدة وتجريب معاملات مبتكرة
- تطبيق النظريات الثلاث في التحليل والاستكشاف
- تعلم من التجارب السابقة وتحسين الأداء
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import json

try:
    from .three_theories_core import ThreeTheoriesIntegrator
    from .adaptive_equations import AdaptiveGSEEquation, AdaptationDirection
except ImportError:
    from three_theories_core import ThreeTheoriesIntegrator
    from adaptive_equations import AdaptiveGSEEquation, AdaptationDirection

logger = logging.getLogger(__name__)

class ExpertMode(Enum):
    """أوضاع النظام الخبير"""
    ANALYSIS = "analysis"          # تحليل الأنماط
    OPTIMIZATION = "optimization"  # تحسين المعاملات
    PREDICTION = "prediction"      # التنبؤ بالأداء
    DIAGNOSIS = "diagnosis"        # تشخيص المشاكل

class ExplorerMode(Enum):
    """أوضاع النظام المستكشف"""
    RANDOM = "random"              # استكشاف عشوائي
    GUIDED = "guided"              # استكشاف موجه
    FOCUSED = "focused"            # استكشاف مركز
    CREATIVE = "creative"          # استكشاف إبداعي

@dataclass
class ExplorationConfig:
    """إعدادات الاستكشاف"""
    exploration_radius: float = 1.0
    max_explorations: int = 50
    creativity_factor: float = 0.2
    risk_tolerance: float = 0.1
    learning_rate: float = 0.01

@dataclass
class ExpertAnalysis:
    """نتيجة تحليل الخبير"""
    pattern_type: str
    confidence: float
    recommendations: List[str]
    optimal_parameters: Dict[str, float]
    risk_assessment: str
    expected_improvement: float

@dataclass
class ExplorationResult:
    """نتيجة الاستكشاف"""
    discovered_patterns: List[Dict]
    new_parameters: List[Dict]
    performance_scores: List[float]
    exploration_path: List[Dict]
    success_rate: float

class GSEExpertSystem:
    """
    النظام الخبير لـ GSE
    
    يحلل الأنماط ويقترح أفضل المعاملات بناءً على الخبرة المتراكمة
    """
    
    def __init__(self):
        self.knowledge_base = {}
        self.pattern_library = {}
        self.experience_history = []
        self.theories_integrator = ThreeTheoriesIntegrator()
        
        # قاعدة معرفة الأنماط الشائعة
        self._initialize_pattern_knowledge()
        
        logger.info("تم تهيئة النظام الخبير GSE")
    
    def _initialize_pattern_knowledge(self):
        """تهيئة قاعدة معرفة الأنماط"""
        
        self.pattern_library = {
            'linear': {
                'description': 'نمط خطي',
                'optimal_components': [{'type': 'linear', 'beta': 1.0, 'gamma': 0.0}],
                'confidence_threshold': 0.9
            },
            'exponential': {
                'description': 'نمط أسي',
                'optimal_components': [{'type': 'sigmoid', 'alpha': 1.0, 'k': 2.0, 'x0': 0.0}],
                'confidence_threshold': 0.8
            },
            'oscillatory': {
                'description': 'نمط متذبذب',
                'optimal_components': [
                    {'type': 'sigmoid', 'alpha': 1.0, 'k': 1.0, 'x0': -1.0},
                    {'type': 'sigmoid', 'alpha': -1.0, 'k': 1.0, 'x0': 1.0}
                ],
                'confidence_threshold': 0.7
            },
            'prime_like': {
                'description': 'نمط شبيه بالأعداد الأولية',
                'optimal_components': [
                    {'type': 'sigmoid', 'alpha': 1.0, 'k': 0.5, 'x0': 2.0},
                    {'type': 'sigmoid', 'alpha': 0.8, 'k': 0.3, 'x0': 3.0},
                    {'type': 'linear', 'beta': 0.1, 'gamma': 0.0}
                ],
                'confidence_threshold': 0.85
            }
        }
    
    def analyze_data_pattern(self, x_data: np.ndarray, y_data: np.ndarray) -> ExpertAnalysis:
        """تحليل نمط البيانات وتقديم توصيات الخبير"""
        
        logger.info("بدء تحليل نمط البيانات")
        
        # تحليل خصائص البيانات
        data_characteristics = self._analyze_data_characteristics(x_data, y_data)
        
        # تحديد نوع النمط
        pattern_type = self._identify_pattern_type(data_characteristics)
        
        # حساب الثقة في التحليل
        confidence = self._calculate_analysis_confidence(data_characteristics, pattern_type)
        
        # تطبيق النظريات الثلاث في التحليل
        theory_insights = self._apply_theories_to_analysis(x_data, y_data)
        
        # توليد التوصيات
        recommendations = self._generate_recommendations(pattern_type, theory_insights)
        
        # اقتراح المعاملات المثلى
        optimal_parameters = self._suggest_optimal_parameters(pattern_type, data_characteristics)
        
        # تقييم المخاطر
        risk_assessment = self._assess_risks(pattern_type, data_characteristics)
        
        # تقدير التحسن المتوقع
        expected_improvement = self._estimate_improvement(pattern_type, confidence)
        
        analysis = ExpertAnalysis(
            pattern_type=pattern_type,
            confidence=confidence,
            recommendations=recommendations,
            optimal_parameters=optimal_parameters,
            risk_assessment=risk_assessment,
            expected_improvement=expected_improvement
        )
        
        # حفظ التحليل في قاعدة المعرفة
        self._update_knowledge_base(analysis, x_data, y_data)
        
        logger.info(f"انتهى التحليل: نمط={pattern_type}, ثقة={confidence:.2%}")
        
        return analysis
    
    def _analyze_data_characteristics(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, float]:
        """تحليل خصائص البيانات"""
        
        characteristics = {}
        
        # الخصائص الإحصائية الأساسية
        characteristics['mean'] = np.mean(y_data)
        characteristics['std'] = np.std(y_data)
        characteristics['min'] = np.min(y_data)
        characteristics['max'] = np.max(y_data)
        characteristics['range'] = characteristics['max'] - characteristics['min']
        
        # خصائص التغير
        if len(y_data) > 1:
            differences = np.diff(y_data)
            characteristics['monotonicity'] = np.sum(np.sign(differences)) / len(differences)
            characteristics['volatility'] = np.std(differences)
            characteristics['trend'] = np.polyfit(x_data, y_data, 1)[0]
        
        # خصائص التذبذب
        if len(y_data) > 2:
            second_diff = np.diff(y_data, 2)
            characteristics['curvature'] = np.mean(np.abs(second_diff))
            characteristics['oscillation'] = np.std(second_diff)
        
        # خصائص التوزيع
        characteristics['skewness'] = self._calculate_skewness(y_data)
        characteristics['kurtosis'] = self._calculate_kurtosis(y_data)
        
        return characteristics
    
    def _identify_pattern_type(self, characteristics: Dict[str, float]) -> str:
        """تحديد نوع النمط بناءً على الخصائص"""
        
        # قواعد تحديد النمط
        if abs(characteristics.get('monotonicity', 0)) > 0.8:
            if abs(characteristics.get('trend', 0)) > 0.1:
                return 'linear'
            else:
                return 'exponential'
        
        elif characteristics.get('oscillation', 0) > characteristics.get('std', 1) * 0.5:
            return 'oscillatory'
        
        elif characteristics.get('curvature', 0) > 0.1:
            # فحص إذا كان شبيه بنمط الأعداد الأولية
            if self._is_prime_like_pattern(characteristics):
                return 'prime_like'
            else:
                return 'exponential'
        
        else:
            return 'linear'
    
    def _is_prime_like_pattern(self, characteristics: Dict[str, float]) -> bool:
        """فحص إذا كان النمط شبيه بالأعداد الأولية"""
        
        # خصائص مميزة للأعداد الأولية
        irregular_spacing = characteristics.get('volatility', 0) > 0.5
        moderate_growth = 0.1 < characteristics.get('trend', 0) < 2.0
        non_uniform_distribution = characteristics.get('skewness', 0) > 0.2
        
        return irregular_spacing and moderate_growth and non_uniform_distribution

    def _calculate_analysis_confidence(self, characteristics: Dict[str, float], pattern_type: str) -> float:
        """حساب مستوى الثقة في التحليل"""

        confidence_factors = []

        # ثقة بناءً على وضوح النمط
        if pattern_type in self.pattern_library:
            threshold = self.pattern_library[pattern_type]['confidence_threshold']
            confidence_factors.append(threshold)

        # ثقة بناءً على جودة البيانات
        data_quality = 1.0 - min(characteristics.get('volatility', 0) / 2.0, 0.5)
        confidence_factors.append(data_quality)

        # ثقة بناءً على اتساق النمط
        consistency = 1.0 - abs(characteristics.get('skewness', 0)) / 3.0
        confidence_factors.append(max(0.1, consistency))

        return np.mean(confidence_factors)

    def _generate_recommendations(self, pattern_type: str, theory_insights: Dict[str, Any]) -> List[str]:
        """توليد توصيات بناءً على التحليل"""

        recommendations = []

        if pattern_type == 'linear':
            recommendations.append("استخدم مكون خطي بسيط")
            recommendations.append("قلل عدد مكونات السيجمويد")
        elif pattern_type == 'exponential':
            recommendations.append("استخدم مكون سيجمويد واحد قوي")
            recommendations.append("اضبط معامل k ليكون أكبر من 1")
        elif pattern_type == 'prime_like':
            recommendations.append("استخدم عدة مكونات سيجمويد")
            recommendations.append("طبق النظريات الثلاث للتحسين")
            recommendations.append("استخدم التكيف التدريجي")

        # توصيات بناءً على النظريات
        if theory_insights.get('balance_score', 0.5) < 0.3:
            recommendations.append("طبق نظرية التوازن لتحسين الاستقرار")

        if theory_insights.get('perpendicular_strength', 0) > 1.0:
            recommendations.append("استخدم التحسين المتعامد للاستكشاف")

        return recommendations

    def _suggest_optimal_parameters(self, pattern_type: str, characteristics: Dict[str, float]) -> Dict[str, float]:
        """اقتراح المعاملات المثلى"""

        if pattern_type in self.pattern_library:
            base_components = self.pattern_library[pattern_type]['optimal_components']

            # تعديل المعاملات بناءً على خصائص البيانات
            suggested_params = {}

            for i, component in enumerate(base_components):
                if component['type'] == 'sigmoid':
                    # تعديل بناءً على نطاق البيانات
                    range_factor = characteristics.get('range', 1.0)
                    suggested_params[f'alpha_{i}'] = component['alpha'] * min(range_factor, 3.0)
                    suggested_params[f'k_{i}'] = component['k']
                    suggested_params[f'x0_{i}'] = component['x0']
                elif component['type'] == 'linear':
                    trend = characteristics.get('trend', 0.0)
                    suggested_params[f'beta_{i}'] = component['beta'] * (1 + trend)
                    suggested_params[f'gamma_{i}'] = component['gamma']

            return suggested_params

        return {}

    def _assess_risks(self, pattern_type: str, characteristics: Dict[str, float]) -> str:
        """تقييم المخاطر"""

        risk_factors = []

        # مخاطر عدم الاستقرار
        if characteristics.get('volatility', 0) > 1.0:
            risk_factors.append("عدم استقرار عالي")

        # مخاطر التعقيد الزائد
        if pattern_type == 'oscillatory':
            risk_factors.append("تعقيد في النمط")

        # مخاطر البيانات المحدودة
        if characteristics.get('range', 1.0) < 0.5:
            risk_factors.append("نطاق بيانات محدود")

        if not risk_factors:
            return "مخاطر منخفضة"
        elif len(risk_factors) == 1:
            return f"مخاطر متوسطة: {risk_factors[0]}"
        else:
            return f"مخاطر عالية: {', '.join(risk_factors)}"

    def _estimate_improvement(self, pattern_type: str, confidence: float) -> float:
        """تقدير التحسن المتوقع"""

        base_improvement = 0.1  # تحسن أساسي 10%

        # تحسن بناءً على نوع النمط
        pattern_multipliers = {
            'linear': 1.5,
            'exponential': 1.2,
            'oscillatory': 0.8,
            'prime_like': 1.0
        }

        pattern_factor = pattern_multipliers.get(pattern_type, 1.0)
        confidence_factor = confidence

        estimated_improvement = base_improvement * pattern_factor * confidence_factor

        return min(estimated_improvement, 0.5)  # حد أقصى 50%

    def _update_knowledge_base(self, analysis: 'ExpertAnalysis', x_data: np.ndarray, y_data: np.ndarray):
        """تحديث قاعدة المعرفة"""

        knowledge_entry = {
            'pattern_type': analysis.pattern_type,
            'confidence': analysis.confidence,
            'data_size': len(x_data),
            'data_range': np.max(x_data) - np.min(x_data),
            'target_range': np.max(y_data) - np.min(y_data),
            'timestamp': datetime.now(),
            'recommendations': analysis.recommendations
        }

        # إضافة للقاعدة
        if analysis.pattern_type not in self.knowledge_base:
            self.knowledge_base[analysis.pattern_type] = []

        self.knowledge_base[analysis.pattern_type].append(knowledge_entry)

        # حفظ في تاريخ الخبرة
        self.experience_history.append(knowledge_entry)

    def _apply_theories_to_analysis(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        """تطبيق النظريات الثلاث في التحليل"""
        
        insights = {}
        
        # نظرية التوازن: تحليل توازن البيانات
        positive_values = y_data[y_data > 0]
        negative_values = y_data[y_data < 0]
        
        if len(positive_values) > 0 and len(negative_values) > 0:
            balance_score = self.theories_integrator.zero_duality.calculate_balance_point(
                np.sum(positive_values), np.sum(np.abs(negative_values))
            )
            insights['balance_score'] = balance_score
        
        # نظرية التعامد: تحليل الاتجاهات المتعامدة
        if len(y_data) > 1:
            gradient = np.gradient(y_data)
            perpendicular_strength = np.std(gradient)
            insights['perpendicular_strength'] = perpendicular_strength
        
        # نظرية الفتائل: تحليل الترابط
        if len(y_data) > 2:
            correlation_strength = np.corrcoef(x_data, y_data)[0, 1]
            insights['correlation_strength'] = abs(correlation_strength)
        
        return insights
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """حساب الانحراف"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """حساب التفلطح"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

class GSEExplorerSystem:
    """
    النظام المستكشف لـ GSE
    
    يستكشف أنماط جديدة ويجرب معاملات مبتكرة
    """
    
    def __init__(self, config: ExplorationConfig = None):
        self.config = config or ExplorationConfig()
        self.exploration_history = []
        self.discovered_patterns = []
        self.theories_integrator = ThreeTheoriesIntegrator()
        
        # إحصائيات الاستكشاف
        self.total_explorations = 0
        self.successful_explorations = 0
        
        logger.info("تم تهيئة النظام المستكشف GSE")
    
    def explore_parameter_space(self, base_equation: AdaptiveGSEEquation,
                               x_data: np.ndarray, y_data: np.ndarray,
                               mode: ExplorerMode = ExplorerMode.GUIDED) -> ExplorationResult:
        """استكشاف فضاء المعاملات"""
        
        logger.info(f"بدء الاستكشاف: وضع={mode.value}")
        
        discovered_patterns = []
        new_parameters = []
        performance_scores = []
        exploration_path = []
        
        base_performance = base_equation.calculate_error(x_data, y_data)
        
        for i in range(self.config.max_explorations):
            # توليد معاملات جديدة حسب الوضع
            if mode == ExplorerMode.RANDOM:
                new_params = self._random_exploration(base_equation)
            elif mode == ExplorerMode.GUIDED:
                new_params = self._guided_exploration(base_equation, x_data, y_data)
            elif mode == ExplorerMode.FOCUSED:
                new_params = self._focused_exploration(base_equation, x_data, y_data)
            else:  # CREATIVE
                new_params = self._creative_exploration(base_equation)
            
            # اختبار المعاملات الجديدة
            test_equation = self._create_test_equation(new_params)
            performance = test_equation.calculate_error(x_data, y_data)
            
            # تسجيل النتائج
            exploration_step = {
                'step': i,
                'parameters': new_params,
                'performance': performance,
                'improvement': base_performance - performance,
                'mode': mode.value
            }
            
            exploration_path.append(exploration_step)
            
            # إذا كان الأداء أفضل، احفظ النتيجة
            if performance < base_performance * (1 + self.config.risk_tolerance):
                new_parameters.append(new_params)
                performance_scores.append(performance)
                
                # تحليل النمط المكتشف
                pattern = self._analyze_discovered_pattern(new_params, performance)
                discovered_patterns.append(pattern)
                
                self.successful_explorations += 1
            
            self.total_explorations += 1
        
        # حساب معدل النجاح
        success_rate = self.successful_explorations / max(1, self.total_explorations)
        
        result = ExplorationResult(
            discovered_patterns=discovered_patterns,
            new_parameters=new_parameters,
            performance_scores=performance_scores,
            exploration_path=exploration_path,
            success_rate=success_rate
        )
        
        # حفظ نتائج الاستكشاف
        self.exploration_history.append(result)
        
        logger.info(f"انتهى الاستكشاف: اكتشافات={len(discovered_patterns)}, نجاح={success_rate:.2%}")
        
        return result
    
    def _random_exploration(self, base_equation: AdaptiveGSEEquation) -> Dict[str, Any]:
        """استكشاف عشوائي"""
        
        new_params = {'components': []}
        
        # عدد عشوائي من المكونات
        num_components = random.randint(1, 4)
        
        for _ in range(num_components):
            component_type = random.choice(['sigmoid', 'linear'])
            
            if component_type == 'sigmoid':
                component = {
                    'type': 'sigmoid',
                    'alpha': random.uniform(0.1, 3.0),
                    'k': random.uniform(0.1, 5.0),
                    'x0': random.uniform(-2.0, 2.0)
                }
            else:
                component = {
                    'type': 'linear',
                    'beta': random.uniform(-2.0, 2.0),
                    'gamma': random.uniform(-1.0, 1.0)
                }
            
            new_params['components'].append(component)
        
        return new_params
    
    def _guided_exploration(self, base_equation: AdaptiveGSEEquation,
                          x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        """استكشاف موجه بالنظريات الثلاث"""
        
        # تطبيق النظريات الثلاث لتوجيه الاستكشاف
        base_components = base_equation.components
        
        if not base_components:
            return self._random_exploration(base_equation)
        
        new_params = {'components': []}
        
        for component in base_components:
            new_component = component.copy()
            
            if component['type'] == 'sigmoid':
                # تطبيق نظرية التوازن
                balance_factor = self.theories_integrator.zero_duality.calculate_balance_point(
                    component['alpha'], 1.0
                )
                new_component['alpha'] *= balance_factor
                
                # تطبيق استكشاف متعامد
                k_perturbation = random.uniform(-0.5, 0.5)
                new_component['k'] = max(0.1, component['k'] + k_perturbation)
                
            elif component['type'] == 'linear':
                # تطبيق تحسين خطي موجه
                beta_perturbation = random.uniform(-0.2, 0.2)
                new_component['beta'] += beta_perturbation
            
            new_params['components'].append(new_component)
        
        return new_params

    def _focused_exploration(self, base_equation: AdaptiveGSEEquation,
                           x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        """استكشاف مركز على تحسين معامل واحد"""

        if not base_equation.components:
            return self._random_exploration(base_equation)

        # اختيار مكون عشوائي للتركيز عليه
        component_idx = random.randint(0, len(base_equation.components) - 1)
        target_component = base_equation.components[component_idx].copy()

        new_params = {'components': [comp.copy() for comp in base_equation.components]}

        if target_component['type'] == 'sigmoid':
            # تركيز على تحسين معامل واحد
            param_to_improve = random.choice(['alpha', 'k', 'x0'])
            improvement_factor = 1 + random.uniform(-0.3, 0.3)

            new_params['components'][component_idx][param_to_improve] *= improvement_factor

            # تطبيق نظرية الفتائل للحفاظ على التوازن
            if param_to_improve == 'alpha':
                new_params['components'][component_idx]['alpha'] = max(0.1,
                    new_params['components'][component_idx]['alpha'])

        return new_params

    def _creative_exploration(self, base_equation: AdaptiveGSEEquation) -> Dict[str, Any]:
        """استكشاف إبداعي بتركيبات جديدة"""

        new_params = {'components': []}

        # إضافة مكونات إبداعية
        creativity_level = self.config.creativity_factor

        # مكون سيجمويد إبداعي
        creative_sigmoid = {
            'type': 'sigmoid',
            'alpha': random.uniform(0.5, 2.0) * (1 + creativity_level),
            'k': random.uniform(0.2, 3.0) * (1 + creativity_level),
            'x0': random.uniform(-3.0, 3.0) * (1 + creativity_level)
        }
        new_params['components'].append(creative_sigmoid)

        # مكون خطي مكمل
        complementary_linear = {
            'type': 'linear',
            'beta': random.uniform(-1.0, 1.0) * creativity_level,
            'gamma': random.uniform(-0.5, 0.5)
        }
        new_params['components'].append(complementary_linear)

        return new_params

    def _create_test_equation(self, params: Dict[str, Any]) -> AdaptiveGSEEquation:
        """إنشاء معادلة اختبار من المعاملات"""

        test_eq = AdaptiveGSEEquation()

        for component in params.get('components', []):
            if component['type'] == 'sigmoid':
                test_eq.add_sigmoid_component(
                    alpha=component['alpha'],
                    k=component['k'],
                    x0=component['x0']
                )
            elif component['type'] == 'linear':
                test_eq.add_linear_component(
                    beta=component['beta'],
                    gamma=component['gamma']
                )

        return test_eq

    def _analyze_discovered_pattern(self, params: Dict[str, Any],
                                  performance: float) -> Dict[str, Any]:
        """تحليل النمط المكتشف"""

        pattern = {
            'parameters': params,
            'performance': performance,
            'discovery_time': datetime.now(),
            'pattern_signature': self._calculate_pattern_signature(params),
            'complexity': len(params.get('components', [])),
            'novelty_score': self._calculate_novelty_score(params)
        }

        return pattern

    def _calculate_pattern_signature(self, params: Dict[str, Any]) -> str:
        """حساب بصمة النمط"""

        signature_parts = []

        for component in params.get('components', []):
            if component['type'] == 'sigmoid':
                sig = f"S({component['alpha']:.2f},{component['k']:.2f},{component['x0']:.2f})"
            else:
                sig = f"L({component['beta']:.2f},{component['gamma']:.2f})"

            signature_parts.append(sig)

        return "+".join(signature_parts)

    def _calculate_novelty_score(self, params: Dict[str, Any]) -> float:
        """حساب درجة الجدة للنمط"""

        if not self.discovered_patterns:
            return 1.0

        # مقارنة مع الأنماط المكتشفة سابقاً
        current_signature = self._calculate_pattern_signature(params)

        similarity_scores = []
        for pattern in self.discovered_patterns:
            existing_signature = pattern.get('pattern_signature', '')
            similarity = self._calculate_signature_similarity(current_signature, existing_signature)
            similarity_scores.append(similarity)

        # الجدة = 1 - أعلى تشابه
        max_similarity = max(similarity_scores) if similarity_scores else 0
        novelty = 1.0 - max_similarity

        return novelty

    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """حساب التشابه بين بصمتين"""

        # تشابه بسيط بناءً على طول التسلسل المشترك
        if not sig1 or not sig2:
            return 0.0

        common_length = 0
        min_length = min(len(sig1), len(sig2))

        for i in range(min_length):
            if sig1[i] == sig2[i]:
                common_length += 1
            else:
                break

        return common_length / max(len(sig1), len(sig2))

class IntegratedExpertExplorer:
    """
    النظام المتكامل للخبير والمستكشف

    يدمج قدرات التحليل والاستكشاف في نظام موحد
    """

    def __init__(self, exploration_config: ExplorationConfig = None):
        self.expert_system = GSEExpertSystem()
        self.explorer_system = GSEExplorerSystem(exploration_config)
        self.integration_history = []

        logger.info("تم تهيئة النظام المتكامل للخبير والمستكشف")

    def intelligent_optimization(self, base_equation: AdaptiveGSEEquation,
                                x_data: np.ndarray, y_data: np.ndarray,
                                max_iterations: int = 10) -> Dict[str, Any]:
        """تحسين ذكي متكامل"""

        logger.info("بدء التحسين الذكي المتكامل")

        best_equation = base_equation
        best_performance = base_equation.calculate_error(x_data, y_data)

        optimization_history = []

        for iteration in range(max_iterations):
            logger.info(f"التكرار {iteration + 1}/{max_iterations}")

            # 1. تحليل الخبير للوضع الحالي
            expert_analysis = self.expert_system.analyze_data_pattern(x_data, y_data)

            # 2. استكشاف موجه بناءً على تحليل الخبير
            if expert_analysis.confidence > 0.7:
                exploration_mode = ExplorerMode.FOCUSED
            elif expert_analysis.confidence > 0.4:
                exploration_mode = ExplorerMode.GUIDED
            else:
                exploration_mode = ExplorerMode.CREATIVE

            exploration_result = self.explorer_system.explore_parameter_space(
                best_equation, x_data, y_data, exploration_mode
            )

            # 3. تقييم النتائج واختيار الأفضل
            if exploration_result.performance_scores:
                best_exploration_idx = np.argmin(exploration_result.performance_scores)
                best_exploration_performance = exploration_result.performance_scores[best_exploration_idx]

                if best_exploration_performance < best_performance:
                    # تحديث أفضل معادلة
                    best_params = exploration_result.new_parameters[best_exploration_idx]
                    best_equation = self.explorer_system._create_test_equation(best_params)
                    best_performance = best_exploration_performance

                    logger.info(f"تحسن في التكرار {iteration + 1}: {best_performance:.6f}")

            # تسجيل تاريخ التحسين
            iteration_result = {
                'iteration': iteration + 1,
                'expert_analysis': expert_analysis,
                'exploration_result': exploration_result,
                'best_performance': best_performance,
                'improvement': base_equation.calculate_error(x_data, y_data) - best_performance
            }

            optimization_history.append(iteration_result)

        final_result = {
            'best_equation': best_equation,
            'best_performance': best_performance,
            'total_improvement': base_equation.calculate_error(x_data, y_data) - best_performance,
            'optimization_history': optimization_history,
            'expert_insights': self.expert_system.knowledge_base,
            'exploration_statistics': {
                'total_explorations': self.explorer_system.total_explorations,
                'successful_explorations': self.explorer_system.successful_explorations,
                'success_rate': self.explorer_system.successful_explorations / max(1, self.explorer_system.total_explorations)
            }
        }

        self.integration_history.append(final_result)

        logger.info(f"انتهى التحسين الذكي: تحسن إجمالي = {final_result['total_improvement']:.6f}")

        return final_result

if __name__ == "__main__":
    # اختبار نظام الخبير/المستكشف
    print("🧪 اختبار نظام الخبير/المستكشف")
    
    # إنشاء النظام الخبير
    expert = GSEExpertSystem()
    
    # بيانات اختبار
    x_data = np.linspace(0, 10, 50)
    y_data = np.sin(x_data) + 0.1 * np.random.randn(50)
    
    # تحليل الخبير
    analysis = expert.analyze_data_pattern(x_data, y_data)
    print(f"تحليل الخبير: نمط={analysis.pattern_type}, ثقة={analysis.confidence:.2%}")
    
    # إنشاء النظام المستكشف
    explorer = GSEExplorerSystem()
    
    # معادلة أساسية للاستكشاف
    base_eq = AdaptiveGSEEquation()
    base_eq.add_sigmoid_component(alpha=1.0, k=1.0, x0=0.0)
    
    # استكشاف
    exploration_result = explorer.explore_parameter_space(
        base_eq, x_data, y_data, ExplorerMode.GUIDED
    )
    
    print(f"نتائج الاستكشاف: اكتشافات={len(exploration_result.discovered_patterns)}")
    print(f"معدل النجاح: {exploration_result.success_rate:.2%}")
    
    print("✅ اختبار نظام الخبير/المستكشف مكتمل!")
