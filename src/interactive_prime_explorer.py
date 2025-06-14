#!/usr/bin/env python3
"""
مستكشف الأعداد الأولية التفاعلي
واجهة تفاعلية متقدمة لاستكشاف وتحليل الأعداد الأولية
باستخدام النظام الهجين المتقدم
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import sys
import os
from datetime import datetime
import threading
import time

# إضافة مسار المشروع
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_hybrid_system import AdvancedHybridSystem
    from enhanced_matrix_sieve import enhanced_matrix_sieve
    from adaptive_equations import AdaptiveGSEEquation
    print("✅ تم تحميل جميع المكونات بنجاح")
except ImportError as e:
    print(f"❌ خطأ في تحميل المكونات: {e}")
    sys.exit(1)

class InteractivePrimeExplorer:
    """
    مستكشف الأعداد الأولية التفاعلي
    """

    def __init__(self):
        self.hybrid_system = AdvancedHybridSystem()
        self.current_results = {}
        self.analysis_history = []

        # إعداد الواجهة الرئيسية
        self.setup_main_window()

        print("🚀 تم إنشاء مستكشف الأعداد الأولية التفاعلي")

    def setup_main_window(self):
        """
        إعداد النافذة الرئيسية
        """

        self.root = tk.Tk()
        self.root.title("مستكشف الأعداد الأولية التفاعلي - GSE Matrix Hybrid")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # إعداد الأنماط
        style = ttk.Style()
        style.theme_use('clam')

        # إنشاء الإطارات الرئيسية
        self.create_control_panel()
        self.create_results_panel()
        self.create_visualization_panel()
        self.create_analysis_panel()

        # شريط الحالة
        self.status_var = tk.StringVar()
        self.status_var.set("جاهز للاستكشاف")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_control_panel(self):
        """
        إنشاء لوحة التحكم
        """

        control_frame = ttk.LabelFrame(self.root, text="لوحة التحكم", padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # إعدادات النطاق
        range_frame = ttk.Frame(control_frame)
        range_frame.pack(fill=tk.X, pady=5)

        ttk.Label(range_frame, text="النطاق:").pack(side=tk.LEFT)

        ttk.Label(range_frame, text="من:").pack(side=tk.LEFT, padx=(10, 0))
        self.start_var = tk.StringVar(value="2")
        start_entry = ttk.Entry(range_frame, textvariable=self.start_var, width=10)
        start_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(range_frame, text="إلى:").pack(side=tk.LEFT, padx=(10, 0))
        self.end_var = tk.StringVar(value="100")
        end_entry = ttk.Entry(range_frame, textvariable=self.end_var, width=10)
        end_entry.pack(side=tk.LEFT, padx=5)

        # أزرار التحكم
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(fill=tk.X, pady=10)

        ttk.Button(buttons_frame, text="🔍 استكشاف الأعداد الأولية",
                  command=self.explore_primes).pack(side=tk.LEFT, padx=5)

        ttk.Button(buttons_frame, text="📊 تحليل متقدم",
                  command=self.advanced_analysis).pack(side=tk.LEFT, padx=5)

        ttk.Button(buttons_frame, text="🔮 تنبؤ بالتالي",
                  command=self.predict_next).pack(side=tk.LEFT, padx=5)

        ttk.Button(buttons_frame, text="💾 حفظ النتائج",
                  command=self.save_results).pack(side=tk.LEFT, padx=5)

        ttk.Button(buttons_frame, text="📂 تحميل النتائج",
                  command=self.load_results).pack(side=tk.LEFT, padx=5)

        # إعدادات النموذج
        model_frame = ttk.LabelFrame(control_frame, text="إعدادات النموذج")
        model_frame.pack(fill=tk.X, pady=5)

        # طريقة التنبؤ
        ttk.Label(model_frame, text="طريقة التنبؤ:").pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="weighted")
        method_combo = ttk.Combobox(model_frame, textvariable=self.method_var,
                                   values=["weighted", "majority", "best_model"], width=15)
        method_combo.pack(side=tk.LEFT, padx=5)

        # عدد النماذج
        ttk.Label(model_frame, text="عدد النماذج:").pack(side=tk.LEFT, padx=(20, 0))
        self.num_models_var = tk.StringVar(value="5")
        models_spin = ttk.Spinbox(model_frame, from_=3, to=10, textvariable=self.num_models_var, width=5)
        models_spin.pack(side=tk.LEFT, padx=5)

    def create_results_panel(self):
        """
        إنشاء لوحة النتائج
        """

        results_frame = ttk.LabelFrame(self.root, text="النتائج", padding="10")
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # جدول النتائج
        columns = ("العدد", "نوع", "ثقة النموذج", "ثقة المصفوفة", "القرار النهائي")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=15)

        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100)

        # شريط التمرير
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # إحصائيات سريعة
        stats_frame = ttk.LabelFrame(results_frame, text="إحصائيات سريعة")
        stats_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        self.stats_text = tk.Text(stats_frame, height=6, width=40)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

    def create_visualization_panel(self):
        """
        إنشاء لوحة التصور
        """

        viz_frame = ttk.LabelFrame(self.root, text="التصور التفاعلي", padding="10")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # أزرار التصور
        viz_buttons = ttk.Frame(viz_frame)
        viz_buttons.pack(fill=tk.X, pady=5)

        ttk.Button(viz_buttons, text="📈 رسم التوزيع",
                  command=self.plot_distribution).pack(side=tk.LEFT, padx=5)

        ttk.Button(viz_buttons, text="🔍 رسم المصفوفة",
                  command=self.plot_matrix).pack(side=tk.LEFT, padx=5)

        ttk.Button(viz_buttons, text="📊 مقارنة النماذج",
                  command=self.plot_model_comparison).pack(side=tk.LEFT, padx=5)

        # منطقة الرسم
        self.viz_text = tk.Text(viz_frame, height=20, width=50)
        self.viz_text.pack(fill=tk.BOTH, expand=True, pady=5)

        viz_scroll = ttk.Scrollbar(viz_frame, orient=tk.VERTICAL, command=self.viz_text.yview)
        self.viz_text.configure(yscrollcommand=viz_scroll.set)
        viz_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def create_analysis_panel(self):
        """
        إنشاء لوحة التحليل المتقدم
        """

        analysis_frame = ttk.LabelFrame(self.root, text="التحليل المتقدم", padding="10")
        analysis_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # خيارات التحليل
        analysis_options = ttk.Frame(analysis_frame)
        analysis_options.pack(fill=tk.X, pady=5)

        ttk.Button(analysis_options, text="🔬 تحليل الفجوات",
                  command=self.analyze_gaps).pack(side=tk.LEFT, padx=5)

        ttk.Button(analysis_options, text="📈 تحليل الاتجاهات",
                  command=self.analyze_trends).pack(side=tk.LEFT, padx=5)

        ttk.Button(analysis_options, text="🎯 تحليل الدقة",
                  command=self.analyze_accuracy).pack(side=tk.LEFT, padx=5)

        ttk.Button(analysis_options, text="🧮 إحصائيات متقدمة",
                  command=self.advanced_statistics).pack(side=tk.LEFT, padx=5)

        # منطقة التحليل
        self.analysis_text = tk.Text(analysis_frame, height=8, width=100)
        self.analysis_text.pack(fill=tk.BOTH, expand=True, pady=5)

        analysis_scroll = ttk.Scrollbar(analysis_frame, orient=tk.VERTICAL, command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=analysis_scroll.set)
        analysis_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def explore_primes(self):
        """
        استكشاف الأعداد الأولية في النطاق المحدد
        """

        try:
            start = int(self.start_var.get())
            end = int(self.end_var.get())

            if start >= end or start < 2:
                messagebox.showerror("خطأ", "يرجى إدخال نطاق صحيح (البداية >= 2 والنهاية > البداية)")
                return

            self.status_var.set(f"جاري استكشاف النطاق {start}-{end}...")
            self.root.update()

            # تشغيل التحليل في خيط منفصل
            thread = threading.Thread(target=self._explore_primes_thread, args=(start, end))
            thread.daemon = True
            thread.start()

        except ValueError:
            messagebox.showerror("خطأ", "يرجى إدخال أرقام صحيحة")

    def _explore_primes_thread(self, start, end):
        """
        تشغيل استكشاف الأعداد الأولية في خيط منفصل
        """

        try:
            # تشغيل النظام المتقدم
            result = self.hybrid_system.comprehensive_evaluation_advanced(end)

            # تصفية النتائج للنطاق المطلوب
            candidates = [c for c in result['candidates'] if start <= c <= end]

            # الحصول على التنبؤات
            method = self.method_var.get()
            num_models = int(self.num_models_var.get())

            models = result['models'][:num_models]

            if method == "weighted":
                predictions, scores = self.hybrid_system.ensemble_prediction(models, np.array(candidates), 'weighted_voting')
            elif method == "majority":
                predictions, scores = self.hybrid_system.ensemble_prediction(models, np.array(candidates), 'majority_voting')
            else:  # best_model
                best_model = models[0]['model']
                raw_predictions = best_model.evaluate(np.array(candidates))
                threshold = models[0].get('optimal_threshold', 0.5)
                predictions = (raw_predictions > threshold).astype(int)
                scores = raw_predictions

            # حفظ النتائج
            self.current_results = {
                'range': (start, end),
                'candidates': candidates,
                'predictions': predictions,
                'scores': scores,
                'models': models,
                'method': method,
                'timestamp': datetime.now().isoformat()
            }

            # تحديث الواجهة
            self.root.after(0, self._update_results_display)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("خطأ", f"حدث خطأ أثناء الاستكشاف: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("حدث خطأ"))

    def _update_results_display(self):
        """
        تحديث عرض النتائج
        """

        # مسح النتائج السابقة
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        if not self.current_results:
            return

        candidates = self.current_results['candidates']
        predictions = self.current_results['predictions']
        scores = self.current_results['scores']

        # الأعداد الأولية الحقيقية للمقارنة
        true_primes = self._get_true_primes(max(candidates))

        # إضافة النتائج للجدول
        for i, candidate in enumerate(candidates):
            is_predicted = predictions[i] == 1
            is_true_prime = candidate in true_primes
            score = scores[i] if hasattr(scores, '__len__') else scores

            # تحديد النوع
            if is_true_prime and is_predicted:
                type_str = "✅ صحيح"
            elif is_true_prime and not is_predicted:
                type_str = "❌ مفقود"
            elif not is_true_prime and is_predicted:
                type_str = "⚠️ خاطئ"
            else:
                type_str = "✅ صحيح"

            # تحديد القرار
            decision = "أولي" if is_predicted else "غير أولي"

            self.results_tree.insert("", "end", values=(
                candidate, type_str, f"{score:.3f}", "متوسط", decision
            ))

        # تحديث الإحصائيات
        self._update_statistics()

        self.status_var.set(f"تم الانتهاء من استكشاف {len(candidates)} مرشح")

    def _update_statistics(self):
        """
        تحديث الإحصائيات السريعة
        """

        if not self.current_results:
            return

        candidates = self.current_results['candidates']
        predictions = self.current_results['predictions']

        # الأعداد الأولية الحقيقية
        true_primes = self._get_true_primes(max(candidates))
        true_in_range = [p for p in true_primes if p in candidates]

        # حساب المقاييس
        predicted_primes = [candidates[i] for i in range(len(candidates)) if predictions[i] == 1]

        tp = len([p for p in predicted_primes if p in true_in_range])
        fp = len([p for p in predicted_primes if p not in true_in_range])
        fn = len([p for p in true_in_range if p not in predicted_primes])

        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # عرض الإحصائيات
        stats_text = f"""📊 إحصائيات النطاق {self.current_results['range'][0]}-{self.current_results['range'][1]}:

🎯 أعداد أولية حقيقية: {len(true_in_range)}
🔮 أعداد متنبأ بها: {len(predicted_primes)}
✅ تنبؤات صحيحة: {tp}
❌ أعداد مفقودة: {fn}
⚠️ إيجابيات خاطئة: {fp}

📈 مقاييس الأداء:
   Precision: {precision:.2f}%
   Recall: {recall:.2f}%
   F1-Score: {f1:.2f}%

🕒 الطريقة: {self.current_results['method']}
📅 الوقت: {self.current_results['timestamp'][:19]}"""

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)

    def _get_true_primes(self, max_num):
        """
        الحصول على الأعداد الأولية الحقيقية
        """

        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        return [n for n in range(2, max_num + 1) if is_prime(n)]

    def advanced_analysis(self):
        """
        تحليل متقدم للنتائج الحالية
        """

        if not self.current_results:
            messagebox.showwarning("تحذير", "يرجى تشغيل الاستكشاف أولاً")
            return

        self.status_var.set("جاري التحليل المتقدم...")

        candidates = self.current_results['candidates']
        predictions = self.current_results['predictions']
        scores = self.current_results['scores']

        # تحليل توزيع النتائج
        predicted_primes = [candidates[i] for i in range(len(candidates)) if predictions[i] == 1]
        true_primes = self._get_true_primes(max(candidates))
        true_in_range = [p for p in true_primes if p in candidates]

        analysis_text = f"""🔬 التحليل المتقدم للنطاق {self.current_results['range'][0]}-{self.current_results['range'][1]}:

📊 توزيع النتائج:
   إجمالي المرشحين: {len(candidates)}
   أعداد أولية حقيقية: {len(true_in_range)}
   تنبؤات إيجابية: {len(predicted_primes)}
   نسبة التنبؤات الإيجابية: {len(predicted_primes)/len(candidates)*100:.2f}%

🎯 تحليل الدقة:
   متوسط نتائج النموذج: {np.mean(scores):.4f}
   انحراف معياري: {np.std(scores):.4f}
   أعلى نتيجة: {np.max(scores):.4f}
   أقل نتيجة: {np.min(scores):.4f}

🔍 تحليل الأخطاء:
   إيجابيات خاطئة: {len([p for p in predicted_primes if p not in true_in_range])}
   سلبيات خاطئة: {len([p for p in true_in_range if p not in predicted_primes])}

📈 أنماط التوزيع:
   كثافة الأعداد الأولية: {len(true_in_range)/(max(candidates)-min(candidates)+1)*100:.2f}%
   متوسط الفجوة بين الأعداد الأولية: {(max(true_in_range)-min(true_in_range))/(len(true_in_range)-1):.2f}

🧮 إحصائيات النموذج:
   عدد النماذج المستخدمة: {len(self.current_results['models'])}
   طريقة التجميع: {self.current_results['method']}
   متوسط دقة النماذج: {np.mean([m['accuracy'] for m in self.current_results['models']])*100:.2f}%
"""

        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(1.0, analysis_text)

        self.status_var.set("تم الانتهاء من التحليل المتقدم")

    def predict_next(self):
        """
        التنبؤ بالأعداد الأولية التالية
        """

        if not self.current_results:
            messagebox.showwarning("تحذير", "يرجى تشغيل الاستكشاف أولاً")
            return

        try:
            # الحصول على آخر عدد في النطاق
            last_num = self.current_results['range'][1]

            # البحث عن الأعداد التالية
            search_range = np.arange(last_num + 1, last_num + 101)  # البحث في 100 عدد تالي

            # استخدام أفضل نموذج للتنبؤ
            best_model = self.current_results['models'][0]['model']
            threshold = self.current_results['models'][0].get('optimal_threshold', 0.5)

            predictions = best_model.evaluate(search_range)
            candidates = search_range[predictions > threshold]
            candidate_scores = predictions[predictions > threshold]

            # ترتيب حسب النتيجة
            sorted_indices = np.argsort(candidate_scores)[::-1]
            top_candidates = candidates[sorted_indices][:10]  # أفضل 10
            top_scores = candidate_scores[sorted_indices][:10]

            # عرض النتائج
            prediction_text = f"""🔮 التنبؤ بالأعداد الأولية التالية بعد {last_num}:

🎯 أفضل 10 مرشحين:
"""

            for i, (candidate, score) in enumerate(zip(top_candidates, top_scores), 1):
                prediction_text += f"   {i:2d}. العدد {candidate}: نتيجة = {score:.4f}\n"

            # التحقق من الدقة
            true_primes = self._get_true_primes(last_num + 100)
            true_next = [p for p in true_primes if p > last_num][:10]

            prediction_text += f"\n✅ الأعداد الأولية الحقيقية التالية:\n"
            for i, prime in enumerate(true_next, 1):
                is_predicted = prime in top_candidates
                status = "✅" if is_predicted else "❌"
                prediction_text += f"   {i:2d}. العدد {prime} {status}\n"

            # حساب دقة التنبؤ
            correct_predictions = len([p for p in top_candidates if p in true_next])
            prediction_accuracy = correct_predictions / len(true_next) * 100 if true_next else 0

            prediction_text += f"\n📊 دقة التنبؤ: {prediction_accuracy:.2f}% ({correct_predictions}/{len(true_next)})"

            self.viz_text.delete(1.0, tk.END)
            self.viz_text.insert(1.0, prediction_text)

            self.status_var.set("تم الانتهاء من التنبؤ")

        except Exception as e:
            messagebox.showerror("خطأ", f"حدث خطأ أثناء التنبؤ: {str(e)}")

    def save_results(self):
        """
        حفظ النتائج الحالية
        """

        if not self.current_results:
            messagebox.showwarning("تحذير", "لا توجد نتائج للحفظ")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="حفظ النتائج"
            )

            if filename:
                # تحضير البيانات للحفظ
                save_data = {
                    'timestamp': self.current_results['timestamp'],
                    'range': self.current_results['range'],
                    'method': self.current_results['method'],
                    'candidates': self.current_results['candidates'],
                    'predictions': self.current_results['predictions'].tolist(),
                    'scores': self.current_results['scores'].tolist() if hasattr(self.current_results['scores'], 'tolist') else self.current_results['scores'],
                    'model_count': len(self.current_results['models']),
                    'model_accuracies': [m['accuracy'] for m in self.current_results['models']]
                }

                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)

                messagebox.showinfo("نجح", f"تم حفظ النتائج في:\n{filename}")

        except Exception as e:
            messagebox.showerror("خطأ", f"فشل في حفظ النتائج: {str(e)}")

    def load_results(self):
        """
        تحميل النتائج المحفوظة
        """

        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="تحميل النتائج"
            )

            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)

                # إعادة بناء النتائج (مبسطة)
                self.current_results = {
                    'range': tuple(loaded_data['range']),
                    'candidates': loaded_data['candidates'],
                    'predictions': np.array(loaded_data['predictions']),
                    'scores': np.array(loaded_data['scores']) if isinstance(loaded_data['scores'], list) else loaded_data['scores'],
                    'method': loaded_data['method'],
                    'timestamp': loaded_data['timestamp'],
                    'models': []  # لا يمكن استعادة النماذج الكاملة
                }

                # تحديث العرض
                self._update_results_display()

                messagebox.showinfo("نجح", f"تم تحميل النتائج من:\n{filename}")

        except Exception as e:
            messagebox.showerror("خطأ", f"فشل في تحميل النتائج: {str(e)}")

    def plot_distribution(self):
        """
        رسم توزيع الأعداد الأولية
        """

        if not self.current_results:
            messagebox.showwarning("تحذير", "يرجى تشغيل الاستكشاف أولاً")
            return

        try:
            candidates = self.current_results['candidates']
            predictions = self.current_results['predictions']
            scores = self.current_results['scores']

            # إنشاء الرسم
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'توزيع الأعداد الأولية - النطاق {self.current_results["range"][0]}-{self.current_results["range"][1]}', fontsize=14)

            # 1. توزيع النتائج
            predicted_primes = [candidates[i] for i in range(len(candidates)) if predictions[i] == 1]
            true_primes = self._get_true_primes(max(candidates))
            true_in_range = [p for p in true_primes if p in candidates]

            ax1.scatter(true_in_range, [1]*len(true_in_range), color='green', alpha=0.7, label='أعداد أولية حقيقية')
            ax1.scatter(predicted_primes, [0.5]*len(predicted_primes), color='red', alpha=0.7, label='تنبؤات النموذج')
            ax1.set_title('مقارنة التنبؤات مع الحقيقة')
            ax1.set_xlabel('العدد')
            ax1.set_ylabel('النوع')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. توزيع النتائج
            ax2.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(x=0.5, color='red', linestyle='--', label='عتبة افتراضية')
            ax2.set_title('توزيع نتائج النموذج')
            ax2.set_xlabel('النتيجة')
            ax2.set_ylabel('التكرار')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. الفجوات بين الأعداد الأولية
            if len(true_in_range) > 1:
                gaps = [true_in_range[i+1] - true_in_range[i] for i in range(len(true_in_range)-1)]
                ax3.plot(true_in_range[:-1], gaps, 'o-', color='purple')
                ax3.set_title('الفجوات بين الأعداد الأولية')
                ax3.set_xlabel('العدد الأولي')
                ax3.set_ylabel('حجم الفجوة')
                ax3.grid(True, alpha=0.3)

            # 4. دقة التنبؤ عبر النطاق
            chunk_size = max(1, len(candidates) // 10)
            chunk_accuracies = []
            chunk_centers = []

            for i in range(0, len(candidates), chunk_size):
                chunk_candidates = candidates[i:i+chunk_size]
                chunk_predictions = predictions[i:i+chunk_size]
                chunk_true = [c for c in chunk_candidates if c in true_in_range]
                chunk_predicted = [chunk_candidates[j] for j in range(len(chunk_candidates)) if chunk_predictions[j] == 1]

                if chunk_true:
                    accuracy = len([p for p in chunk_predicted if p in chunk_true]) / len(chunk_true)
                    chunk_accuracies.append(accuracy)
                    chunk_centers.append(np.mean(chunk_candidates))

            if chunk_accuracies:
                ax4.plot(chunk_centers, chunk_accuracies, 'o-', color='orange')
                ax4.set_title('دقة التنبؤ عبر النطاق')
                ax4.set_xlabel('موقع في النطاق')
                ax4.set_ylabel('الدقة')
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            self.status_var.set("تم عرض رسم التوزيع")

        except Exception as e:
            messagebox.showerror("خطأ", f"فشل في رسم التوزيع: {str(e)}")

    def run(self):
        """
        تشغيل الواجهة التفاعلية
        """

        print("🚀 بدء تشغيل مستكشف الأعداد الأولية التفاعلي...")
        self.root.mainloop()

def main():
    """
    الدالة الرئيسية
    """

    print("🎯 مستكشف الأعداد الأولية التفاعلي")
    print("واجهة متقدمة لاستكشاف وتحليل الأعداد الأولية")
    print("="*60)

    try:
        explorer = InteractivePrimeExplorer()
        explorer.run()

    except Exception as e:
        print(f"❌ خطأ في تشغيل المستكشف: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()