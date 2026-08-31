---
layout: default
title: تصنيف النصوص
nav_order: 6
---

# تصنيف النصوص
{: .fs-10 .no_toc }

## المحتويات
{: .no_toc .text-delta }

1. TOC
{:toc}

## المكتبات المستعملة
{: .no_toc .text-delta }
```python
import numpy as np
from b4msa.textmodel import TextModel
from EvoMSA.tests.test_base import TWEETS
from microtc.utils import tweet_iterator, load_model, save_model
from scipy.stats import norm, multinomial, multivariate_normal
from matplotlib import pylab as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from scipy.special import logsumexp
from sklearn.metrics import recall_score, precision_score, f1_score
from EvoMSA.utils import bootstrap_confidence_interval
from sklearn.naive_bayes import MultinomialNB
from os.path import join
```

## تثبيت المكتبات الخارجية
{: .no_toc .text-delta }

```bash
pip install b4msa
pip install evomsa
```

---

# المقدمة

تصنيف النصوص (Text Categorization) هو إحدى مهام معالجة اللغة الطبيعية (NLP) التي تتعامل مع إنشاء خوارزميات قادرة على تحديد فئة النص من بين مجموعة من الفئات المحددة مسبقًا. على سبيل المثال، ينتمي تحليل المشاعر (Sentiment Analysis) إلى هذه المهمة، والهدف منه هو فحص وتحديد قطبية النص (مثل: إيجابي، محايد، أو سلبي). علاوة على ذلك، فإن المهام المختلفة في NLP التي تبدو غير مرتبطة بهذه المشكلة في البداية يمكن صياغتها كمسألة تصنيف، مثل الإجابة عن الأسئلة (Question Answering) والاستلزام اللغوي (Sentence Entailment)، على سبيل المثال لا الحصر.

يمكن معالجة تصنيف النصوص من وجهات نظر مختلفة؛ والمنهج المتبع هنا هو التعامل معها كـ مسألة تعلم خاضع للإشراف (Supervised Learning). وكما هو الحال في أي مسألة تعلم خاضع للإشراف، فإن نقطة البداية هي مجموعة من الأزواج، حيث يكون العنصر الأول هو المدخل والعنصر الثاني يمثل المخرج. لنفرض أن $$\mathcal D = \{(\text{text}_i, y_i) \mid i=1,\ldots, N\}$$ حيث $$y \in \{c_1, \ldots c_K\}$$ و $$\text{text}_i$$ هو النص.

يمكن النظر إلى مسائل التعلم الخاضع للإشراف كـ إيجاد دالة تعيين من المدخلات إلى المخرجات. يمكن أن تكون الأداة عبارة عن خوارزمية [تحسين](/NLP-Course-Ar/topics/02Vocabulary.html#sec:optimization) قادرة على العثور على الدالة التي تقرر ودالة خسارة معينة، مثل $$L$$.

$$\min_{g \in \Omega} \sum_{(\mathbf x, y) \in \mathcal D} L(y, g(\mathbf x)),$$

حيث $$\Omega$$ هو فضاء البحث عن دوال التعيين الممكنة.

بالإضافة إلى ذلك، إذا كان المرء مهتمًا أيضًا بقياس درجة عدم اليقين، فإن المسار يعتمد على الاحتمالات. في هذا السيناريو الأخير، يتمثل أحد المناهج في فرض شكل الاحتمال الشرطي، أي $$\mathbb P(\mathcal Y=k \mid \mathcal X=x)=f_k(x)$$ حيث $$f_k$$ هي القيمة الرقمية $$k$$ لـ $$f: \mathcal X \rightarrow [0, 1]^K$$ والتي ترمز دالة كتل الاحتمال. بالنسبة لحالة مسألة التصنيف الثنائي، تكون الدالة $$f: \mathcal X \rightarrow [0, 1]$$. وكما يتضح، في هذا السيناريو، فإن التوزيع المناسب هو [برنولي (Bernoulli)](/NLP-Course-Ar/topics/03Collocations.html#sec:bernoulli)، حيث تأخذ الدالة $$f$$ مكان معلمة التوزيع، أي $$\mathcal Y \sim \textsf{Bernoulli}(f(\mathcal X))$$؛ ولعدد أكبر من التسميات، يمكن استخدام التوزيع الفئوي (Categorical distribution). من ناحية أخرى، يعتمد المسار المكمل على مبرهنة بايز (Bayes' theorem).

# مبرهنة بايز (Bayes' theorem)
{: #sec:bayes-theorem }

يمكن التعبير عن التوزيع ثنائي المتغير $$\mathbb P(\mathcal X, \mathcal Y)$$ باستخدام [الاحتمال الشرطي](/NLP-Course-Ar/topics/04NGramLM.html#sec:conditional-probability) كـ:

$$\begin{eqnarray}
\mathbb P(\mathcal X, \mathcal Y) &=& \mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)\\
\mathbb P(\mathcal X, \mathcal Y) &=& \mathbb P(\mathcal Y \mid \mathcal X) \mathbb P(\mathcal X).
\end{eqnarray}$$

يمكن دمج هذه العناصر للحصول على مبرهنة بايز باتباع الخطوات التالية:

$$\begin{eqnarray}
\mathbb P(\mathcal Y \mid \mathcal X) \mathbb P(\mathcal X) &=& \mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)\\
\mathbb P(\mathcal Y \mid \mathcal X)  &=& \frac{\mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)}{\mathbb P(\mathcal X)},
\end{eqnarray}$$

حيث $$\mathbb P(\mathcal Y \mid \mathcal X)$$ هو **الاحتمال البعدي (Posterior probability)**، و $$\mathbb P(\mathcal X \mid \mathcal Y)$$ يمثل **الاحتمالية / الإمكانية (Likelihood)**، و $$\mathbb P(\mathcal Y)$$ هو **الاحتمال القبلي (Prior)**، و $$\mathbb P(\mathcal X)$$ هو **الدليل (Evidence)**. يمكن التعبير عن الدليل كـ $$\mathbb P(\mathcal X) = \sum_y \mathbb P(\mathcal X \mid \mathcal y) \mathbb P(\mathcal y)$$ والذي يتوافق مع الهامش $$\mathbb P(\mathcal X)$$ باستخدام الاحتمال الشرطي؛ ويمكن ملاحظة أن هذا المصطلح يعمل كـ ثابت معايرة (Normalization constant).

تتمتع مبرهنة بايز بميزتين تجعلانها مناسبة لمعالجة مسائل التصنيف. الأولى هي أنها نموذج توليدي (Generative model)؛ فإلى جانب معالجة مسألة التصنيف، يمكن استخدام النموذج لتوليد البيانات، أي أن النموذج يتعلم توزيع مجموعة البيانات.

والسمة الثانية هي أن الاحتمالية (Likelihood) عبارة عن توزيع احتمالي لأي فئة. وبالتالي، تتمثل المسألة في تقدير $$K$$ من التوزيعات المختلفة باستخدام المجموعة الفرعية من مجموعة التدريب التي تنتمي إلى كل فئة مختلفة. من ناحية أخرى، فإن الاحتمال القبلي هو الاحتمال المقدر لكل فئة، ويمكن تقدير الدليل باستخدام القيمتين السابقين.

## التوزيع الطبيعي (Normal Distribution)

من أجل توضيح عملية حساب الاحتمال البعدي، يستخدم المثال التالي توزيعين طبيعيين، كل منهما يتوافق مع فئة مختلفة؛ الأحرف الحمراء للفئة السلبية، والأزرق لتمثيل الفئة الإيجابية.

```python
pos = norm(loc=3, scale=2.5)
neg = norm(loc=-0.5, scale=0.75)
```

![Two Normals](/NLP-Course-Ar/assets/images/two_normals.png)

يتم أخذ عينات من التوزيع الطبيعي المرتبط بالفئة السلبية 100 مرة؛ ومع ذلك، فإن العناصر التي تم أخذ عينات منها في الذيل والتي تتوافق مع كتلة أقل من 0.05 أو أعلى من 0.95 يتم التخلص منها. يتم أخذ عينات من توزيع الفئة الإيجابية 1000 مرة باستخدام قيد أن النقاط في الفترة الزمنية للفئة السلبية لا تؤخذ بعين الاعتبار.

```python
_min = neg.ppf(0.05)
_max = neg.ppf(0.95)
D = [(x, 0) for x in neg.rvs(100) if x >= _min and x <= _max]
D += [(x, 1) for x in pos.rvs(1000) if x < _min or x > _max]
```

توضح الصورة التالية توزيع الفئتين الإيجابية والسلبية؛ ويمكن ملاحظة أن الفئتين مفصولتان بالقيود المفروضة. ستُستخدم هذه النقاط لتوضيح إجراء تقدير التوزيع البعدي بمعلومية مجموعة البيانات؛ مجموعة البيانات هي $$\mathcal D=\{(x_i, y_i) \mid i=1, \ldots, N\}$$ حيث $$x_i \in \mathbb R$$ و $$y_i \in \{0, 1\}$$.

![Two Normal Samples](/NLP-Course-Ar/assets/images/two_normal_samples.png)

الخطوة الأولى هي تقدير الاحتمالية (Likelihood)، أي $$\mathbb P(\mathcal X \mid \mathcal Y)$$ حيث $$\mathcal Y=1$$ و $$\mathcal Y=0$$. يُفترض أن الاحتمالية موزعة طبيعيًا؛ وبالتالي، يتطلب الأمر تقدير المتوسط والإنحراف المعياري، وهو ما يمكن القيام به باستخدام الكود التالي.

```python
l_pos = norm(*norm.fit([x for x, k in D if k == 1]))
l_neg = norm(*norm.fit([x for x, k in D if k == 0]))
```

الخطوة الثانية هي حساب الاحتمال القبلي (Prior)، أي $$\mathbb P(\mathcal Y)$$ والذي يتوافق مع تقدير معلمات التوزيع الفئوي؛ يعتمد الكود التالي على استخدام `np.unique` لتقديرها.

```python
_, priors = np.unique([k for _, k in D], return_counts=True)
N = priors.sum()
prior_pos = priors[1] / N
prior_neg = priors[0] / N
```

تتطلب الخطوة التالية حساب الاحتمال البعدي غير المعاير $$\mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)$$; في المثال التالي، يتم حساب هذا المصطلح لجميع المدخلات في $$\mathcal D$$. يسترجع السطر الأول المدخلات، أي $$x$$. يحسب السطران الثاني والثالث $$\mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)$$ للفئة الإيجابية والسلبية.

```python
x = np.array([x for x, _ in D])
post_pos = l_pos.pdf(x) * prior_pos
post_neg = l_neg.pdf(x) * prior_neg
```

الخطوات النهائية هي حساب الدليل، $$\mathbb P(\mathcal X)$$ واستخدامه لمعايرة $$\mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)$$. يحسب السطر الأول الدليل، والسطران الثاني والأرابع يعايران الاحتمال البعدي.

```python
evidence = post_pos + post_neg
post_pos /= evidence
post_neg /= evidence
```

يقدم الشكل التالي الاحتمال البعدي لكل فئة؛ ويمكن ملاحظة أن الفئة الأكثر احتمالاً تتغير من الإيجابية إلى السلبية في تقاطع الخطين.

![Posterior of Two Classes](/NLP-Course-Ar/assets/images/two_classes_posterior.png)

بمجرد تقدير الاحتمال البعدي، يمكن استخدامه للتنبؤ بفئة $$x$$؛ بالنظر إلى أن الفئة الحقيقية لأي $$x$$ في $$\mathcal D$$ معروفة، فمن الممكن معرفة متى يرتكب المصنف خطأً. يوضح الشكل التالي البيانات في $$\mathcal D$$، مع تعليم النقاط باللون الأحمر حيث تختلف فئة المصنف عن الفئة الحقيقية. يمكن تنفيذ دالة التنبؤ بالفئة باستخدام الكود التالي؛ وتجدر الإشارة إلى أنه ليس من الضروري معايرة الاحتمال البعدي لأن الاهتمام ينصب فقط على الفئة.

```python
klass = lambda x: 1 if l_pos.pdf(x) * prior_pos > l_neg.pdf(x) * prior_neg else 0
```

![Posterior Errors](/NLP-Course-Ar/assets/images/two_classes_posterior_error.png)

## التوزيع الطبيعي متعدد المتغيرات (Multivariate Normal Distribution)

يمكن إجراء إجراء مكافئ للتوزيع الطبيعي متعدد المتغيرات. يوضح الشكل التالي مثالاً لتوزيعين متعدد المتغيرات؛ أحدهما يمثل فئة إيجابية (أزرق)، والآخر يتوافق مع الفئة السلبية (أحمر). توجد مجموعة البيانات التي تحتوي على الأزواج، $$(\mathbf x, y)$$، في المتغير `D`.

![Two Multivariate Normals](/NLP-Course-Ar/assets/images/two_classes_multivariate.png)

```python
D = load_model(join('dataset', 'two_classes_multivariate.gz'))
```

يمكن استخدام مجموعة البيانات $$\mathcal D$$ لتقدير التوزيع البعدي، حيث الخطوة الأولى هي تقدير معلمات الاحتمالية (Likelihood)، مجموعة واحدة من المعلمات لكل فئة. الخطوة الثانية هي حساب معلمات الاحتمال القبلي. يتوافق مجموع حاصل ضرب هذين المكونين مع الدليل، الذي يوفر جميع العناصر لحساب التوزيع البعدي.

يحسب الكود التالي الاحتمالية للفئة الإيجابية والسلبية.

```python
l_pos_m = np.mean(np.array([x for x, y in D if y == 1]), axis=0)
l_pos_cov = np.cov(np.array([x for x, y in D if y == 1]).T)
l_pos = multivariate_normal(mean=l_pos_m, cov=l_pos_cov)
l_neg_m = np.mean(np.array([x for x, y in D if y == 0]), axis=0)
l_neg_cov = np.cov(np.array([x for x, y in D if y == 0]).T)
l_neg = multivariate_normal(mean=l_neg_m, cov=l_neg_cov)
```

بمجرد تقدير الاحتمالية، فمن المباشر تقدير الاحتمال القبلي والدليل، ومع ذلك، تكون قادرًا على حساب الاحتمال البعدي. يمكن استخدام التوزيع البعدي للتنبؤ بفئة كل نقطة في $$\mathcal D$$. يوضح الشكل التالي باللون الأحمر النقاط في $$\mathcal D$$ حيث تختلف الفئة الحقيقية عن التوزيع البعدي المتنبأ به.

![Classification Errors in Two Multivariate Normals](/NLP-Course-Ar/assets/images/two_classes_multivariate_error.png)

## التوزيع الفئوي (Categorical Distribution)
{: #sec:categorical-distribution }

تستمر صياغة مبرهنة بايز بمثال للتوزيع الفئوي. يمكن للتوزيع الفئوي محاكاة سحب $$K$$ من الأحداث التي يمكن ترميزها كأحرف، ويمكن تمثيل تكرارات $$\ell$$ كتسلسل من الأحرف. وبالتالي، يمكن للتوزيع توضيح متواليات التوليد المرتبطة بفئات مختلفة، مثل الإيجابية أو السلبية.

الخطوة الأولى هي إنشاء مجموعة البيانات. كما تم عمله سابقًا، يتم تعريف توزيعين، واحد لكل فئة؛ ويمكن ملاحظة أن كل توزيع له معلمات مختلفة. الخطوة الثانية هي أخذ عينات من هذه التوزيعات؛ يتم أخذ عينات من التوزيعات 1000 مرة بالإجراء التالي. في كل مرة، يتم سحب متغير عشوائي يمثل عدد المخرجات المأخوذة من كل توزيع من توزيع طبيعي $$\mathcal N(15, 3)$$ ويخزن في المتغير `length.` يشير المتغير العشوائي إلى عدد المخرجات لكل توزيع فئوي؛ وتتحول النتائج إلى تسلسل، مرتبط بالتسمية المقابلة للفئة الإيجابية والسلبية، وتخزن في القائمة `D.`

```python
pos = multinomial(1, [0.20, 0.20, 0.35, 0.25])
neg = multinomial(1, [0.35, 0.20, 0.25, 0.20])
length = norm(loc=15, scale=3)
D = []
m = {k: chr(122 - k) for k in range(4)}
id2w = lambda x: " ".join([m[_] for _ in x.argmax(axis=1)])
for l in length.rvs(size=1000):
    D.append((id2w(pos.rvs(round(l))), 1))
    D.append((id2w(neg.rvs(round(l))), 0))
```

يوضح الجدول التالي أربعة أمثلة لهذه العملية؛ يحتوي العمود الأول على التسلسل، والعمود الثاني على التسمية المرتبطة.

|النص (Text)          |التسمية (Label)    |
|--------------|---------|
|x w x x z w y | positive       |
|y w z z z x w | negative       |
|z x x x z x z w x w | positive |
|x w z w y z z z z w | negative |

كما تم عمله سابقًا، فإن الخطوة الأولى هي حساب الاحتمالية بمعلومية مجموعة البيانات؛ بالنظر إلى أن البيانات تأتي من توزيع فئوي، فإن إجراء تقدير المعلمات يماثل الإجراءات المستخدمة لتقدير الاحتمال القبلي. يقدر الكود التالي معلمات البيانات المقابلة للفئة الإيجابية. يمكن ملاحظة أن المعلمات المقدرة تشبه المعلمات المستخدمة لتوليد مجموعة البيانات.

```python
D_pos = []
[D_pos.extend(data.split()) for data, k in D if k == 1]
words, l_pos = np.unique(D_pos, return_counts=True)
w2id = {v: k for k, v in enumerate(words)}
l_pos = l_pos / l_pos.sum()
l_pos
array([0.25489421, 0.33854064, 0.20773186, 0.1988333 ])
```

يتم إجراء إجراء مكافئ لحساب احتمالية الفئة السلبية.

```python
D_neg = []
[D_neg.extend(data.split()) for data, k in D if k == 0]
_, l_neg = np.unique(D_neg, return_counts=True)
l_neg = l_neg / l_neg.sum()
```

يتم تقدير الاحتمال القبلي بالكود التالي، وهو مكافئ للكود المستخدم في جميع الأمثلة التي شوهدت حتى الآن.

```python
_, priors = np.unique([k for _, k in D], return_counts=True)
N = priors.sum()
prior_pos = priors[1] / N
prior_neg = priors[0] / N
```

بمجرد تحديد المعلمات، يمكن استخدامها للتنبؤ بفئة تسلسل معين. الخطوة الأولى هي حساب الاحتمالية، على سبيل المثال $$\mathbb P($$w w x z$$\mid \mathcal Y)$$. يمكن ملاحظة أن التسلسل بحاجة إلى التحول إلى رموز وهو ما يمكن القيام به باستخدام الدالة `split`. ثم يتم تحويل الرمز إلى فهرس باستخدام التعيين `w2id`؛ وبمجرد استرجاع الفهرس، يمكن استخدامه للحصول على المعلمة المرتبطة بالكلمة. الاحتمالية هي حاصل ضرب جميع الاحتمالات؛ ومع ذلك، يتم حساب حاصل الضرب هذا في الفضاء اللوغاريتمي.

```python
def likelihood(params, txt):
    params = np.log(params)
    _ = [params[w2id[x]] for x in txt.split()]
    tot = sum(_)
    return np.exp(tot)
```

تنتج الاحتمالية المدمجة مع الاحتمال القبلي لجميع الفئات الدليل، والذي يُستخدم لاحقًا لحساب التوزيع البعدي. ثم يُستخدم الاحتمال البعدي للتنبؤ بالفئة لجميع المتواليات في $$\mathcal D$$. يتم تخزين التنبؤات في المتغير `hy`.

```python
post_pos = [likelihood(l_pos, x) * prior_pos for x, _ in D]
post_neg = [likelihood(l_neg, x) * prior_neg for x, _ in D]
evidence = np.vstack([post_pos, post_neg]).sum(axis=0)
post_pos /= evidence
post_neg /= evidence
hy = np.where(post_pos > post_neg, 1, 0)
```

# الدقة (Accuracy)

في الأمثلة السابقة، تم استخدام الأشكال لتصوير أخطاء التصنيف؛ ومع ذلك، ليس من العملي الاعتماد على شكل لتقييم أداء المصنف. بدلاً من ذلك، يمكن استخدام مقياس الأداء لتقييم جودة المصنف. مقياس الأداء الأول الذي تم مراجعته هو الدقة (Accuracy). الدقة هي نسبة التنبؤات الصحيحة.

تتم حساب دقة المصنف المدرب سابقًا بالكود التالي، حيث يحتوي المتغير `hy` على التنبؤات ويحتوي `y` على الفئات المأخوذة من $$\mathcal D$$.

```python
y = np.array([y for _, y in D])
(hy == y).mean()
0.761
```

# فترة الثقة (Confidence Interval)

مثل أي مقياس أداء آخر يُطبق في هذا المجال، يمكن أن تتغير الدقة عند تكرار التجربة؛ فإن أخذ عينات من التوزيعات وإنشاء مجموعة بيانات جديدة من شأنه أن ينتج دقة مختلفة. لذلك، للحصول على صورة كاملة لأداء المصنف، يلزم تقدير القيم المختلفة التي يمكن أن يتخذها هذا المقياس تحت نفس الظروف. أحد المناهج هو حساب فترة الثقة (Confidence interval) لمقياس الأداء المستخدم. تفترض الطريقة القياسية لحساب فترة الثقة أنها موزعة طبيعيًا عندما يؤول حجم $$\mathcal D$$ إلى المالانهاية؛ في هذه الحالة، تكون فترة الثقة هي $$(\hat \theta - z_{\frac{\alpha}{2}}\hat{\textsf{se}}, \hat \theta + z_{\frac{\alpha}{2}}\hat{\textsf{se}})$$, حيث $$\hat \theta$$ هي نقطة التقدير، مثل الدقة، و $$z_{\frac{\alpha}{2}}$$ هي النقطة التي تكون فيها الكتلة $$1-\frac{\alpha}{2}$$, و $$\hat{\textsf{se}} = \sqrt{\mathbb V(\hat \theta)}$$ هو الخطأ المعياري.

الدقة هي مجموع محاولات برنولي $$N$$، وبالتالي فإن $$\sqrt{\mathbb V(\hat \theta)}$$ هي $$\sqrt{\frac{p(1-p)}{N}}$$ حيث $$p$$ هي الدقة. وباستخدام هذه العناصر، تتم حساب فترة الثقة للدقة كما يلي.

```python
p = (hy == y).mean()
se = np.sqrt(p * (1 - p) / y.shape[0]) 
coef = norm.ppf(0.975)
ci = (p - coef * se, p + coef * se)
ci
(0.7423093514177674, 0.7796906485822326)
```

يمكن اشتقاق الخطأ المعياري للدقة باستخدام المطابقة $$\mathbb V(\sum_i a_i \mathcal X_i) = \sum_i a_i^2 \mathbb V(\mathcal X_i)$$ حيث تكون المتغيرات العشوائية $$\mathcal X_i$$ مستقلة و $$a_i$$ ثابت. من ناحية أخرى، يمكن النظر إلى الدقة على أنها مخرج لمتغير عشوائي حيث تشير $$1$$ إلى التنبؤ الصحيح و $$0$$ تمثل خطأً، ثم تكون الدقة هي مجموع هذه المتغيرات العشوائية. لندع $$\mathcal X_i$$ يمثل مخرج التنبؤ $$i$$، فتكون الدقة هي $$\frac{1}{N} \sum_i^N X_i$$. والتباين هو $$\mathbb V(\frac{1}{N} \sum_i^N X_i) = \sum_i \frac{1}{N^2} \mathbb V(\mathcal X_i)$$; وتباين توزيع برنولي بالمعلمة $$p$$ هو $$p(1-p)$$, وبالتالي فإن $$\sum_i \frac{1}{N^2} \mathbb V(\mathcal X_i) = \frac{1}{N^2} \sum_i p(1-p) = \frac{1}{N}p(1-p)$$, وهو ما يكمل الاشتقاق.

هناك مقاييس أداء يكون من الصعب أو غير العملي الحصول على $$\sqrt{\mathbb V(\hat \theta)}$$ لها تحليلياً، وفي هذه الحالات، يمكن استخدام طريقة التمهيد (Bootstrapping) لتقديرها، ويظهر الكود التالي استخدام طريقة تنفذ فترة النسبة المئوية للتمهيد عندما يكون مقياس الأداء هو الدقة. ومع ذلك، يمكن ملاحظة أن المقياس هو معلمة بالطريقة، لذا فهو يعمل مع أي مقياس أداء.

```python
ci = bootstrap_confidence_interval(y, hy, alpha=0.05,
                                  metric=lambda a, b: (a == b).mean())
ci                                  
(0.7415, 0.7797625)
```

# تصنيف النصوص - التوزيع الفئوي (Text Categorization - Categorical Distribution)
{: #sec:tc-categorical }

المنهج المتبع في تصنيف النصوص هو التعامل معها كمسألة تعلم خاضع للإشراف حيث نقطة البداية هي مجموعة بيانات $$\mathcal D = \{(\text{text}_i, y_i) \mid i=1,\ldots, N\}$$ حيث $$y \in \{c_1, \ldots c_K\}$$ و $$\text{text}_i$$ هو النص. على سبيل المثال، يستخدم الكود التالي مجموعة بيانات توضيحية لتحليل المشاعر بأربع فئات: سلبي (N)، محايد (NEU)، غياب القطبية (NONE)، وإيجابي (P).

```python
D = [(x['text'], x['klass']) for x in tweet_iterator(TWEETS)]
```

كما يمكن ملاحظته، فإن $$\mathcal D$$ تكافئ تلك المستخدمة في مثال [التوزيع الفئوي](#sec:categorical-distribution). الفرق هو أن تسلسل الأحرف يتغير بجملة. ومع ذلك، فإن المنهج الممكن هو الحصول على الرموز باستخدام الدالة `split`. ومنهج آخر هو استرجاع الرموز باستخدام المحلل اللفظي (Tokenizer)، كما غطينا في قسم [توحيد النصوص](/NLP-Course-Ar/topics/05TextNormalization.html).

يستخدم الكود التالي الفئة `TextModel` لتقطيع النص باستخدام الكلمات كمحلل لفظي؛ ويتم تخزين النص المقطع في المتغير `D.`

```python
tm = TextModel(token_list=[-1])
tok = tm.tokenize
D = [(tok(x), y) for x, y in D]
```

قبل تقدير معلمات الاحتمالية، يتطلب الأمر ترميز الرموز باستخدام فهرس؛ وبتنفيذ ذلك، من الممكن تخزين المعلمات في مصفوفة وحساب كل شيء بعمليات `numpy`. يرمز الكود التالي كل رمز بفهرس فريد؛ التعيين موجود في القاموس `w2id`.

```python
words = set()
[words.update(x) for x, y in D]
w2id = {v: k for k, v in enumerate(words)}
```

سابقًا، تم تمثيل الفئات باستخدام الأعداد الطبيعية. ارتبطت الفئة الإيجابية بالرقم $$1$$، بينما السلبية بـ $$0$$. ومع ذلك، في مجموعة البيانات هذه، الفئات عبارة عن سلاسل نصية. وتقرر ترميزها كأرقام لتسهيل العمليات اللاحقة. يمكن إجراء عملية الترميز بالتزامن مع تقدير الاحتمال القبلي لكل فئة. يُرجى ملاحظة أن الاحتمالات القبلية مخزنة باستخدام اللوغاريتم في المتغير `priors.`

```python
uniq_labels, priors = np.unique([k for _, k in D], return_counts=True)
priors = np.log(priors / priors.sum())
uniq_labels = {str(v): k for k, v in enumerate(uniq_labels)}
```

حان الوقت لتقدير معلمات الاحتمالية لكل من الفئات. يُفترض أن البيانات تأتي من توزيع فئوي وأن كل رمز مستقل. يمكن تخزين معلمات الاحتمالية في مصفوفة (المتغير `l_tokens`) بـ $$K$$ من الصفوف، يحتوي كل صف على معلمات الفئة، وعدد الأعمدة يتوافق مع حجم المفردات. الخطوة الأولى هي حساب تكرار كل رمز لكل فئة وهو ما يمكن القيام به باستخدام الكود التالي.

```python
l_tokens = np.zeros((len(uniq_labels), len(w2id)))
for x, y in D:
    w = l_tokens[uniq_labels[y]]
    cnt = Counter(x)
    for i, v in cnt.items():
        w[w2id[i]] += v
```

الخطوة التالية هي معايرة التكرار. ومع ذلك، قبل معايرتها، يتم استخدام تنعيم لابلاس بقيمة $$0.1$$. لذلك، يتم إضافة الثابت $$0.1$$ إلى جميع عناصر المصفوفة. الخطوة التالية هي المعايرة (السطر الثاني)، وأخيرًا، يتم تخزين المعلمات باستخدام اللوغاريتم.

```python
l_tokens += 0.1
l_tokens = l_tokens / np.atleast_2d(l_tokens.sum(axis=1)).T
l_tokens = np.log(l_tokens)
```

## التنبؤ (Prediction)

بمجرد تقدير جميع المعلمات، حان الوقت لاستخدام النموذج لتصنيف أي نص. تحسب الدالة التالية التوزيع البعدي. الخطوة الأولى هي تقطيع النص (السطر الثاني) وحساب تكرار كل رمز في النص. يتحول التكرار المخزن في القاموس `cnt` إلى المتجه `x` باستخدام دالة التعيين `w2id`. الخطوة النهائية هي حساب حاصل ضرب الاحتمالية والاحتمال القبلي. يتم حساب حاصل الضرب في الفضاء اللوغاريتمي؛ وبالتالي، يتم ذلك باستخدام مجموع الاحتمالية والاحتمال القبلي. الخطوة الأخيرة هي حساب الدليل ومعايرة النتيجة؛ يتم حساب الدليل بالدالة `logsumexp.`

```python
def posterior(txt):
    x = np.zeros(len(w2id))
    cnt = Counter(tm.tokenize(txt))
    for i, v in cnt.items():
        try:
            x[w2id[i]] += v
        except KeyError:
            continue
    _ = (x * l_tokens).sum(axis=1) + priors
    l = np.exp(_ - logsumexp(_))
    return l
```

يمكن لدالة الاحتمال البعدي التنبؤ بجميع النصوص في $$\mathcal D$$؛ وتُستخدم التنبؤات لحساب دقة النموذج. من أجل حساب الدقة، يلزم تحويل الفئات في $$\mathcal D$$ باستخدام تسمية مصفوفة الاحتمالية ومتجه الاحتمالات القبلية؛ ويتم ذلك باستخدام القاموس `uniq_labels` (السطر الثاني).

```python
hy = np.array([posterior(x).argmax() for x, _ in D])
y = np.array([uniq_labels[y] for _, y in D])
(y == hy).mean()
0.974
```

## التدريب (Training)

يتطلب حل مسائل التعلم الخاضع للإشراف مرحلتين؛ إحداهما هي مرحلة التدريب، والأخرى هي التنبؤ. تتعامل دالة الاحتمال البعدي مع المرحلة الأخيرة، ويتبقى تنظيم الكود الموصوف في دالة تدريب. يصف الكود التالي دالة التدريب؛ وتتطلب معلمات مجموعة البيانات ومثيل من `TextModel.`

```python
def training(D, tm):
    tok = tm.tokenize
    D =[(tok(x), y) for x, y in D]
    words = set()
    [words.update(x) for x, y in D]
    w2id = {v: k for k, v in enumerate(words)}
    uniq_labels, priors = np.unique([k for _, k in D], return_counts=True)
    priors = np.log(priors / priors.sum())
    uniq_labels = {str(v): k for k, v in enumerate(uniq_labels)}
    l_tokens = np.zeros((len(uniq_labels), len(w2id)))
    for x, y in D:
        w = l_tokens[uniq_labels[y]]
        cnt = Counter(x)
        for i, v in cnt.items():
            w[w2id[i]] += v
    l_tokens += 0.1
    l_tokens = l_tokens / np.atleast_2d(l_tokens.sum(axis=1)).T
    l_tokens = np.log(l_tokens)
    return w2id, uniq_labels, l_tokens, priors
```

# التقييم والأداء (Performance)

لا يمكن قياس أداء خوارزمية التعلم الخاضع للإشراف على نفس البيانات التي تدربت عليها. لتوضيح هذه المشكلة، تخيل خوارزمية تحفظ مجموعة البيانات، ولجميع المدخلات التي لم تُشاهد، تخرج الخوارزمية فئة عشوائية. وبالتالي، فإن هذه الخوارزمية غير مفيدة لأنه لا يمكن استخدامها للتنبؤ بمدخلات خارج مجموعة البيانات المستخدمة لتدريبها. ومع ذلك، فإن لها درجة مثالية، في أي مقياس أداء، في مجموعة البيانات المستخدمة لتقدير معلماتها.

تقليديًا، يتم التعامل مع هذه المشكلة عن طريق تقسيم مجموعة البيانات $$\mathcal D$$ إلى مجموعتين منفصلتين، $$\mathcal D = \mathcal T \cup \mathcal G$$ حيث $$\mathcal T \cap \mathcal G = \emptyset$$, أو ثلاث مجموعات $$\mathcal D = \mathcal T \cup \mathcal V \cup \mathcal G$$ حيث $$\mathcal T \cap \mathcal V \cap \mathcal G = \emptyset$$. تُستخدم المجموعة $$\mathcal T$$، المعروفة باسم **مجموعة التدريب (Training set)**، لتدريب الخوارزمية، أي لتقدير المعلمات، بينما تسمى المجموعة $$\mathcal G$$، **مجموعة الاختبار (Test set)** أو **المجموعة الذهبية (Gold set)**، وتُستخدم لقياس أدائها. وتستخدم المجموعة $$\mathcal V$$، المعروفة باسم **مجموعة التحقق (Validation set)**، لتحسين المعلمات الفائقة (Hyperparameters) للخوارزمية؛ على سبيل المثال، فإن المعلمة الفائقة في نموذج اللغة n-gram هي قيمة $$n$$.

## التجميع الطبقي KFold و StratifiedKFold

هناك سيناريوهات يمنع فيها حجم $$\mathcal D$$ تقسيمها إلى مجموعات تدريب وتحقق واختبار؛ في هذه الحالة، يكون المنهج البديل هو تقسيم $$\mathcal D$$ عدة مرات باختيار مختلف في كل مرة. وتُعرف هذه العملية باسم التحقق المتقاطع k-fold (k-fold cross-validation) عندما تُستخدم جميع العناصر في $$\mathcal D$$ مرة واحدة في مجموعة التحقق. على سبيل المثال، يتوافق التحقق المتقاطع k-fold عندما تكون $$k=3$$ مع العملية التالية. يتم تقسيم المجموعة $$\mathcal D$$ إلى ثلاث مجموعات منفصلة $$\mathcal D_1, \mathcal D_2, \mathcal D_3$$ حيث $$\mathcal D=\mathcal D_1 \cup \mathcal D_2 \cup \mathcal D_3;$$ بالخاصية التي تجعل جميع المجموعات الفرعية ذات أصل مماثل. وتُستخدم مجموعات البيانات هذه لإنشاء ثلاث مجموعات تدريب وتحقق، أي $$\mathcal T_1=\mathcal D_2 \cup \mathcal D_3$$, و $$\mathcal V_1=\mathcal D_1$$, و $$\mathcal T_2=\mathcal D_1 \cup \mathcal D_3$$, و $$\mathcal V_2=\mathcal D_2$$, و $$\mathcal T_3=\mathcal D_1 \cup \mathcal D_2$$, و $$\mathcal V_3=\mathcal D_3.$$ وكما يمكن ملاحظته، تُستخدم جميع العناصر في $$\mathcal D$$ مرة واحدة في مجموعة التحقق.

القيود الوحيدة المفروضة في تقاطع k-fold هي أن جميع مجموعات التحقق لها أصل مماثل وأن جميع العناصر في $$\mathcal D$$ تظهر مرة واحدة. بالنسبة للمسائل التي تكون فيها نسبة الفئات المختلفة غير متوازنة، يكون من المفيد إضافة قيد آخر في عملية الاختيار؛ والقيد هو إجبار توزيع الفئات على البقاء متشابهاً في جميع مجموعات التحقق. وتُعرف هذه العملية الأخيرة باسم التحقق المتقاطع الطبقي k-fold (Stratified k-fold cross-validation).

ينفذ الكود التالي التحقق المتقاطع الطبقي k-fold، ويتنبأ بجميع العناصر في $$\mathcal D$$، ويخزن التنبؤات في المتغير `hy.`

```python
D = [(x['text'], x['klass']) for x in tweet_iterator(TWEETS)]
tm = TextModel(token_list=[-1])
folds = StratifiedKFold(shuffle=True, random_state=0)
hy = np.empty(len(D))
for tr, val in folds.split(D, y):
    _ = [D[x] for x in tr]
    w2id, uniq_labels, l_tokens, priors = training(_, tm)
    hy[val] = [posterior(D[x][0]).argmax() for x in val]
```

بمجرد التنبؤ بالفئات في $$\mathcal D$$، يمكن حساب دقة المصنف بالكود التالي. يمكن ملاحظة أن الدقة أقل من تلك التي تم الحصول عليها عندما تم قياسها على نفس البيانات المستخدمة لتقدير المعلمات.

```python
y = np.array([uniq_labels[y] for _, y in D])
(y == hy).mean()
0.618
```

تتم حساب فترة الثقة للدقة كما يلي.

```python
p = (hy == y).mean()
s = np.sqrt(p * (1 - p) / y.shape[0]) 
coef = norm.ppf(0.975)
ci = (p - coef * s, p + coef * s)
ci
(0.5878856141926456, 0.6481143858073544)
```

## الضبط، الاستدعاء، ومقياس F1 (Precision, Recall, and F1-score)

الدقة (Accuracy) مقياس أداء شائع؛ ومع ذلك، لها عيوب. على سبيل المثال، في مجموعة بيانات تكون فيها إحدى الفئات أكثر تكرارًا من الأخرى، مثل وجود 99 مثالاً للفئة السلبية ومثال واحد فقط للفئة الإيجابية، فإن دقة المصنف الذي يتنبأ دائمًا بالسلبية تكون .99. يمكن النظر إلى هذا الأداء على أنه كافٍ؛ ومع ذلك، فإن المصنف المحدد ثابت، ولا ينظر حتى في المدخلات.

المقاييس الشهيرة الأخرى المستخدمة في التصنيف هي الضبط (Precision)، والاستدعاء (Recall)، ومقياس $$f_1$$. يتم تعريف مقاييس الأداء هذه في مسائل التصنيف الثنائي؛ ومع ذلك، يمكن ترميز مسائل التصنيف k-class كـ $$k$$ من المسائل الثنائية؛ في كل من هذه المسائل، تكون الفئة الإيجابية إحدى الفئات، وتكون الفئة السلبية هي اتحاد بقية الفئات.

الضبط (Precision) هو نسبة التصنيف الصحيح للكائنات الإيجابية، أي $$\textsf{precision}(\mathbf y, \hat{\mathbf y}) = \frac{\sum_i \mathbb 1(\mathbf y_i = 1) \mathbb 1(\mathbf{\hat y_i} = 1) }{\sum_i \mathbb 1(\mathbf{ \hat y_i = 1})}.$$ من ناحية أخرى، فإن الاستدعاء (Recall) هو $$\textsf{recall}(\mathbf y, \hat{\mathbf y}) = \frac{\sum_i \mathbb 1(\mathbf y_i = 1) \mathbb 1(\mathbf{\hat y_i} = 1) }{\sum_i \mathbb 1(\mathbf{y_i = 1})}.$$ يستخدم الكود التالي دالتين من `sklearn` لحساب الضبط والاستدعاء.

```python
p = precision_score(y, hy, average=None)
r = recall_score(y, hy, average=None)
```

علاوة على ذلك، يتم تعريف مقياس $$f_1$$ بمصطلحي الضبط والاستدعاء؛ وهو المتوسط التوافقي $$f_1 = 2 \frac{\textsf{precision} \cdot \textsf{recall}}{\textsf{precision} + \textsf{recall}}$$.

```python
f1_score(y, hy, average=None)
```

يتم تعريف الضبط، والاستدعاء، ومقياس $$f_1$$ في مسائل التصنيف الثنائي، وتتم حساب هذه المقاييس في معظم الأوقات للفئة الإيجابية؛ ومع ذلك، لا يوجد ما يمنع حسابها للفئة الأخرى؛ في الكود السابق، يلزم فقط تغيير $$1$$ إلى $$0$$. بالإضافة إلى ذلك، من الممكن حساب هذه المقاييس لجميع الفئات في مسألة تصنيف $$K$$، وتكون النتيجة الحصول على مقياس واحد لكل فئة؛ ومتوسط هذه القيم يُعرف باسم *Macro*. يحسب الكود التالي فترة الثقة لـ macro-recall المحصول عليها من التنبؤات في $$\mathcal D$$ باستخدام التحقق المتقاطع الطبقي k-fold.

```python
metric = lambda a, b: recall_score(a, b, average='macro')
ci = bootstrap_confidence_interval(y, hy,
                                   metric=metric)
ci
(0.4006296585578326, 0.4415247767402683)
```

# المحلل اللفظي (Tokenizer)

أحد المكونات الحسمية في خوارزمية تصنيف النصوص هو الطريقة المستخدمة لتقطيع النص؛ ويؤثر تغيير معلمات المحلل اللفظي على أداء المصنف، على سبيل المثال، يظهر الكود التالي مثالاً حيث استخدم المحلل اللفظي المعلمات الافتراضية. يمكن ملاحظة زيادة في الأداء، ولكن كلتا فترتي الثقة تتداخلان، لذلك لا يمكن الاستنتاج بأن مجموعة معينة من المعلمات أفضل من الأخرى.

```python
tm = TextModel()
hy = np.empty(len(D))
for tr, val in folds.split(D, y):
    T = [D[x] for x in tr]
    w2id, uniq_labels, l_tokens, priors = training(T, tm)
    assert np.all(np.isfinite([posterior(D[x][0]) for x in val]))
    hy[val] = [posterior(D[x][0]).argmax() for x in val]

y = np.array([uniq_labels[y] for _, y in D])
metric = lambda a, b: recall_score(a, b, average='macro')
ci = bootstrap_confidence_interval(y, hy,
                                   metric=metric)
ci
(0.4326075670113598, 0.47454989683417365)
```
