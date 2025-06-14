#!/usr/bin/env python3
"""
النظام الهجين المتقدم: تطوير شامل للنموذج
- توسيع النطاق لأعداد أكبر
- تحسين دقة التنبؤ
- تقليل الأخطاء الإيجابية الخاطئة
- نماذج متعددة المراحل
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# إضافة مسار المشروع
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_matrix_sieve import enhanced_matrix_sieve, extract_matrix_features
    from adaptive_equations import AdaptiveGSEEquation, AdaptationDirection
    from three_theories_core import ThreeTheoriesIntegrator
    from expert_explorer_system import GSEExpertSystem, GSEExplorerSystem
    print("✅ تم تحميل جميع المكونات بنجاح")
except ImportError as e:
    print(f"❌ خطأ في تحميل المكونات: {e}")
    sys.exit(1)

class AdvancedHybridSystem:
    """
    النظام الهجين المتقدم مع تحسينات شاملة
    """
    
    def __init__(self):
        self.matrix_sieve_cache = {}
        self.gse_models = {}
        self.expert_system = GSEExpertSystem()
        self.explorer_system = GSEExplorerSystem()
        self.theories_integrator = ThreeTheoriesIntegrator()
        
        # إعدادات متقدمة
        self.adaptive_thresholds = {}
        self.performance_history = []
        self.model_ensemble = []
        
        print("🚀 تم إنشاء النظام الهجين المتقدم")
    
    def scalable_matrix_sieve(self, max_num=1000, chunk_size=200):
        """
        غربال مصفوفي قابل للتوسع للأعداد الكبيرة
        """
        
        print(f"\n🔍 الغربال المصفوفي القابل للتوسع (حتى {max_num})")
        print("="*60)
        
        if max_num in self.matrix_sieve_cache:
            print("   📋 استخدام النتائج المحفوظة...")
            return self.matrix_sieve_cache[max_num]
        
        # تقسيم النطاق إلى أجزاء
        chunks = []
        for start in range(2, max_num + 1, chunk_size):
            end = min(start + chunk_size - 1, max_num)
            chunks.append((start, end))
        
        print(f"   📊 تقسيم النطاق إلى {len(chunks)} جزء")
        
        # معالجة متوازية للأجزاء
        all_candidates = [2]  # البدء بالعدد 2
        all_removed = set()
        
        for i, (start, end) in enumerate(chunks):
            print(f"   🔄 معالجة الجزء {i+1}/{len(chunks)}: {start}-{end}")
            
            # تطبيق الغربال على الجزء
            chunk_result = self._process_chunk(start, end)
            
            # دمج النتائج
            all_candidates.extend(chunk_result['candidates'])
            all_removed.update(chunk_result['removed'])
        
        # إزالة التكرارات وترتيب
        final_candidates = sorted(list(set(all_candidates)))
        
        result = {
            'candidates': final_candidates,
            'removed': all_removed,
            'chunks_processed': len(chunks),
            'total_candidates': len(final_candidates)
        }
        
        # حفظ في الذاكرة المؤقتة
        self.matrix_sieve_cache[max_num] = result
        
        print(f"   ✅ انتهت المعالجة: {len(final_candidates)} مرشح")
        
        return result
    
    def _process_chunk(self, start, end):
        """
        معالجة جزء من النطاق
        """
        
        # الأعداد الفردية في الجزء
        if start <= 2:
            odd_numbers = [2] + [n for n in range(3, end + 1, 2)]
        else:
            odd_numbers = [n for n in range(start if start % 2 == 1 else start + 1, end + 1, 2)]
        
        # الأعداد الأولية الصغيرة للمصفوفة
        sqrt_end = int(np.sqrt(end)) + 1
        small_primes = self._get_small_primes(sqrt_end)
        
        # إنشاء نواتج الضرب
        products_to_remove = set()
        
        for prime in small_primes:
            for num in odd_numbers:
                if prime != num and prime * num <= end:
                    products_to_remove.add(prime * num)
        
        # الأعداد المتبقية
        candidates = [num for num in odd_numbers if num not in products_to_remove]
        
        return {
            'candidates': candidates,
            'removed': products_to_remove
        }
    
    def _get_small_primes(self, max_num):
        """
        الحصول على الأعداد الأولية الصغيرة
        """
        
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        return [n for n in range(2, max_num + 1) if is_prime(n)]
    
    def ensemble_gse_models(self, training_data, num_models=5):
        """
        إنشاء مجموعة من نماذج GSE المتنوعة
        """
        
        print(f"\n🧠 إنشاء مجموعة من {num_models} نماذج GSE متنوعة")
        print("="*60)
        
        x_train, y_train = training_data
        models = []
        
        # إعدادات متنوعة للنماذج
        model_configs = [
            # نموذج محافظ
            {'alphas': [1.0, 0.8, 0.6], 'ks': [0.5, 0.4, 0.3], 'x0s': [10, 30, 60]},
            # نموذج متوسط
            {'alphas': [1.5, 1.2, 1.0], 'ks': [0.8, 0.6, 0.4], 'x0s': [15, 35, 70]},
            # نموذج جريء
            {'alphas': [2.0, 1.8, 1.5], 'ks': [1.2, 1.0, 0.8], 'x0s': [8, 25, 50]},
            # نموذج متخصص في الأعداد الصغيرة
            {'alphas': [2.5, 2.0, 1.0], 'ks': [1.5, 1.2, 0.6], 'x0s': [5, 15, 40]},
            # نموذج متخصص في الأعداد الكبيرة
            {'alphas': [1.2, 1.5, 2.0], 'ks': [0.3, 0.5, 1.0], 'x0s': [20, 50, 100]}
        ]
        
        for i, config in enumerate(model_configs[:num_models]):
            print(f"   🔧 تدريب النموذج {i+1}/{num_models}")
            
            # إنشاء النموذج
            model = AdaptiveGSEEquation()
            
            # إضافة المكونات
            for alpha, k, x0 in zip(config['alphas'], config['ks'], config['x0s']):
                model.add_sigmoid_component(alpha=alpha, k=k, x0=x0)
            
            model.add_linear_component(beta=0.001, gamma=0.0)
            
            # تدريب النموذج
            initial_error = model.calculate_error(x_train, y_train)
            
            for j in range(3):
                success = model.adapt_to_data(x_train, y_train, AdaptationDirection.IMPROVE_ACCURACY)
                if not success:
                    break
            
            final_error = model.calculate_error(x_train, y_train)
            improvement = ((initial_error - final_error) / initial_error) * 100
            
            # تقييم النموذج
            predictions = model.evaluate(x_train)
            accuracy = np.mean((predictions > 0.3).astype(int) == y_train)
            
            model_info = {
                'model': model,
                'config': config,
                'training_improvement': improvement,
                'accuracy': accuracy,
                'final_error': final_error
            }
            
            models.append(model_info)
            
            print(f"      تحسن: {improvement:.2f}%, دقة: {accuracy:.2%}")
        
        # ترتيب النماذج حسب الأداء
        models.sort(key=lambda x: x['accuracy'], reverse=True)
        
        self.model_ensemble = models
        
        print(f"   ✅ تم إنشاء {len(models)} نماذج")
        print(f"   🏆 أفضل دقة: {models[0]['accuracy']:.2%}")
        
        return models
    
    def adaptive_threshold_optimization(self, models, validation_data):
        """
        تحسين العتبات التكيفية لكل نموذج
        """
        
        print(f"\n🎯 تحسين العتبات التكيفية")
        print("="*60)
        
        x_val, y_val = validation_data
        
        for i, model_info in enumerate(models):
            model = model_info['model']
            predictions = model.evaluate(x_val)
            
            # اختبار عتبات مختلفة
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_threshold = 0.5
            best_f1 = 0
            
            for threshold in thresholds:
                binary_preds = (predictions > threshold).astype(int)
                
                # حساب F1-Score
                tp = np.sum((binary_preds == 1) & (y_val == 1))
                fp = np.sum((binary_preds == 1) & (y_val == 0))
                fn = np.sum((binary_preds == 0) & (y_val == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            model_info['optimal_threshold'] = best_threshold
            model_info['best_f1'] = best_f1
            
            print(f"   النموذج {i+1}: عتبة مثلى = {best_threshold:.3f}, F1 = {best_f1:.3f}")
        
        return models
    
    def ensemble_prediction(self, models, x_data, method='weighted_voting'):
        """
        تنبؤ مجمع من عدة نماذج
        """
        
        if method == 'weighted_voting':
            # تصويت مرجح حسب دقة كل نموذج
            weights = [model['accuracy'] for model in models]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            ensemble_predictions = np.zeros(len(x_data))
            
            for model_info, weight in zip(models, weights):
                model = model_info['model']
                threshold = model_info.get('optimal_threshold', 0.5)
                
                predictions = model.evaluate(x_data)
                binary_predictions = (predictions > threshold).astype(float)
                
                ensemble_predictions += weight * binary_predictions
            
            # القرار النهائي
            final_predictions = (ensemble_predictions > 0.5).astype(int)
            
        elif method == 'majority_voting':
            # تصويت الأغلبية
            all_predictions = []
            
            for model_info in models:
                model = model_info['model']
                threshold = model_info.get('optimal_threshold', 0.5)
                
                predictions = model.evaluate(x_data)
                binary_predictions = (predictions > threshold).astype(int)
                all_predictions.append(binary_predictions)
            
            # حساب الأغلبية
            votes = np.array(all_predictions)
            final_predictions = (np.mean(votes, axis=0) > 0.5).astype(int)
        
        return final_predictions, ensemble_predictions if method == 'weighted_voting' else np.mean(votes, axis=0)
    
    def comprehensive_evaluation_advanced(self, max_num=500):
        """
        تقييم شامل متقدم للنظام
        """
        
        print(f"\n📊 تقييم شامل متقدم (حتى {max_num})")
        print("="*80)
        
        start_time = time.time()
        
        # المرحلة 1: الغربال المصفوفي القابل للتوسع
        print("🔍 المرحلة 1: الغربال المصفوفي...")
        matrix_result = self.scalable_matrix_sieve(max_num)
        
        # المرحلة 2: تحضير بيانات التدريب والتحقق
        print("📊 المرحلة 2: تحضير البيانات...")
        candidates = matrix_result['candidates']
        
        # الأعداد الأولية الحقيقية
        true_primes = self._get_small_primes(max_num)
        
        # تقسيم البيانات
        split_point = len(candidates) // 2
        train_candidates = candidates[:split_point]
        val_candidates = candidates[split_point:]
        
        # إنشاء بيانات التدريب
        x_train = np.array(train_candidates)
        y_train = np.array([1 if x in true_primes else 0 for x in train_candidates])
        
        x_val = np.array(val_candidates)
        y_val = np.array([1 if x in true_primes else 0 for x in val_candidates])
        
        print(f"   بيانات التدريب: {len(x_train)} عينة")
        print(f"   بيانات التحقق: {len(x_val)} عينة")
        
        # المرحلة 3: إنشاء مجموعة النماذج
        print("🧠 المرحلة 3: إنشاء مجموعة النماذج...")
        models = self.ensemble_gse_models((x_train, y_train))
        
        # المرحلة 4: تحسين العتبات
        print("🎯 المرحلة 4: تحسين العتبات...")
        models = self.adaptive_threshold_optimization(models, (x_val, y_val))
        
        # المرحلة 5: التنبؤ المجمع
        print("🔮 المرحلة 5: التنبؤ المجمع...")
        
        # اختبار على جميع المرشحين
        all_candidates = np.array(candidates)
        y_true = np.array([1 if x in true_primes else 0 for x in candidates])
        
        # تنبؤ مرجح
        weighted_preds, weighted_scores = self.ensemble_prediction(models, all_candidates, 'weighted_voting')
        
        # تنبؤ الأغلبية
        majority_preds, majority_scores = self.ensemble_prediction(models, all_candidates, 'majority_voting')
        
        # المرحلة 6: التقييم النهائي
        print("📈 المرحلة 6: التقييم النهائي...")
        
        results = {}
        
        for method, predictions in [('weighted', weighted_preds), ('majority', majority_preds)]:
            tp = np.sum((predictions == 1) & (y_true == 1))
            fp = np.sum((predictions == 1) & (y_true == 0))
            tn = np.sum((predictions == 0) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))
            
            accuracy = (tp + tn) / len(y_true) * 100
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[method] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            }
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # طباعة النتائج
        print(f"\n🎉 النتائج النهائية:")
        print(f"   نطاق الاختبار: 2-{max_num}")
        print(f"   أعداد أولية حقيقية: {len(true_primes)}")
        print(f"   مرشحين من الغربال: {len(candidates)}")
        print(f"   وقت المعالجة: {processing_time:.2f} ثانية")
        
        print(f"\n📊 مقارنة الطرق:")
        for method, metrics in results.items():
            print(f"   {method.upper()}:")
            print(f"      الدقة: {metrics['accuracy']:.2f}%")
            print(f"      Precision: {metrics['precision']:.2f}%")
            print(f"      Recall: {metrics['recall']:.2f}%")
            print(f"      F1-Score: {metrics['f1_score']:.2f}%")
        
        return {
            'matrix_result': matrix_result,
            'models': models,
            'results': results,
            'processing_time': processing_time,
            'true_primes': true_primes,
            'candidates': candidates
        }

def main():
    """
    الدالة الرئيسية للنظام المتقدم
    """
    
    print("🚀 النظام الهجين المتقدم - التطوير الشامل")
    print("توسيع النطاق، تحسين الدقة، وتقليل الأخطاء")
    print("="*80)
    
    try:
        # إنشاء النظام المتقدم
        advanced_system = AdvancedHybridSystem()
        
        # تشغيل التقييم الشامل
        print("🎯 بدء التقييم الشامل المتقدم...")
        
        # اختبار على نطاقات مختلفة
        test_ranges = [200, 300, 500]
        
        all_results = {}
        
        for max_num in test_ranges:
            print(f"\n" + "="*60)
            print(f"🔍 اختبار النطاق: 2-{max_num}")
            print("="*60)
            
            result = advanced_system.comprehensive_evaluation_advanced(max_num)
            all_results[max_num] = result
            
            # عرض أفضل النتائج
            best_method = max(result['results'].keys(), 
                            key=lambda x: result['results'][x]['f1_score'])
            best_f1 = result['results'][best_method]['f1_score']
            
            print(f"🏆 أفضل أداء في النطاق {max_num}: {best_method.upper()}")
            print(f"   F1-Score: {best_f1:.2f}%")
        
        # ملخص النتائج النهائية
        print(f"\n" + "="*80)
        print(f"🎉 ملخص النتائج النهائية")
        print("="*80)
        
        for max_num, result in all_results.items():
            best_method = max(result['results'].keys(), 
                            key=lambda x: result['results'][x]['f1_score'])
            metrics = result['results'][best_method]
            
            print(f"📊 النطاق {max_num}:")
            print(f"   أفضل طريقة: {best_method.upper()}")
            print(f"   F1-Score: {metrics['f1_score']:.2f}%")
            print(f"   Precision: {metrics['precision']:.2f}%")
            print(f"   Recall: {metrics['recall']:.2f}%")
            print(f"   وقت المعالجة: {result['processing_time']:.2f}s")
        
        # حفظ النتائج
        summary = {
            'timestamp': datetime.now().isoformat(),
            'system_type': 'advanced_hybrid_system',
            'test_ranges': test_ranges,
            'results_summary': {
                str(max_num): {
                    'best_method': max(result['results'].keys(), 
                                     key=lambda x: result['results'][x]['f1_score']),
                    'best_f1': max(result['results'][x]['f1_score'] 
                                 for x in result['results'].keys()),
                    'processing_time': result['processing_time']
                }
                for max_num, result in all_results.items()
            }
        }
        
        with open('advanced_hybrid_results.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 تم حفظ النتائج في: advanced_hybrid_results.json")
        print(f"🌟 النظام المتقدم جاهز للمرحلة التالية!")
        
    except Exception as e:
        print(f"\n❌ خطأ في النظام المتقدم: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
