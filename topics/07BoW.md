---
layout: default
title: حقيبة الكلمات
nav_order: 7
---

# حقيبة الكلمات
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
from microtc.utils import tweet_iterator
from EvoMSA.utils import LabelEncoderWrapper, bootstrap_confidence_interval
from EvoMSA.model import Multinomial
from EvoMSA import BoW
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
```

## تثبيت المكتبات الخارجية
{: .no_toc .text-delta }

```bash
pip install b4msa
pip install evomsa
```

---

# المقدمة

يمكن معالجة مسألة تصنيف النصوص مباشرة لنمذجة الاحتمال الشرطي، أي $$\mathbb P(\mathcal Y=k \mid \mathcal X=x)=f_k(x)$$ حيث $$f_k$$ هي القيمة $$k$$ لـ $$f: \mathcal X \rightarrow [0, 1]^K$$ والتي ترمز دالة كتل الاحتمال. بالنسبة لحالة فئات $$K$$ فإن التوزيع المناسب هو [الفئوي (Categorical)](/NLP-Course-Ar/topics/03Collocations.html#sec:categorical)، حيث تأخذ الدالة $$f$$ مكان معلمة التوزيع، أي $$\mathcal Y \sim \textsf{Categorical}(f(\mathcal X))$$.

يمكن تحديد معلمات الدالة $$f$$ باستخدام مقدر الاحتمالية العظمى؛ وهذا الإجراء يعادل الإجراء المستخدم للمعلمة $$\mathbf p$$ للتوزيع [الفئوي (Categorical)](/NLP-Course-Ar/topics/03Collocations.html#sec:categorical).

# مقدر الاحتمالية العظمى (Maximum Likelihood Estimator)

يتم تعريف مقدر اللوغاريتم للاحتمالية كما يلي، حيث $$\mathcal D$$ هي مجموعة البيانات المستخدمة لتقدير المعلمات، و $$f_\mathcal Y$$ هي دالة كتل الاحتمال التي تتوافق مع التوزيع الفئوي، وتأخذ $$f(x)$$ مكان معلمة التوزيع؛ وكما يمكن ملاحظته، فهي دالة في المدخلات.

$$\begin{eqnarray}
l_{f_\mathcal Y}(f) &=& \log \prod_{(x, y) \in \mathcal D} f_\mathcal Y(y \mid f(x)) \\
&=& \log \prod_{(x, y) \in \mathcal D} \prod_{k=1}^K f_k(x)^{\mathbb 1(k=y)}\\
\end{eqnarray}$$

باقتراض أن الدالة $$f$$ تحتوي على معلمة $$w_j$$، فإن إجراء تقدير المعلمة هو حساب المشتقة الجزئية للوغاريتم الاحتمالية بالنسبة لـ $$w_j$$ وحلها عندما تسوي الصفر.

$$\begin{eqnarray}
\frac{\partial}{\partial w_j} l_{f_\mathcal Y}(f) &=& \frac{\partial}{\partial w_j} \log \prod_{(x, y) \in \mathcal D} \prod_{k=1}^K f_k(x)^{\mathbb 1(k=y)}\\
&=& \frac{\partial}{\partial w_j} \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K \mathbb 1(k=y) \log f_k(x) = 0
\end{eqnarray}$$

## تقليل الانتروبيا المتقاطعة (Minimizing Cross-entropy)

قبل حل لوغاريتم الاحتمالية، من الأساسي ربط هذا المفهوم بـ الإنتروبيا المتقاطعة (Cross-entropy). أولاً، يتم حساب القيمة المتوقعة لـ $$h(\mathcal X)$$ كـ $$\sum_{k \in \mathcal X} h(k) f(k)$$ حيث $$f$$ هي دالة الكتلة، ويمكن التعبير عن ذلك كـ $$\mathbb E_f[h(\mathcal X)]$$. من ناحية أخرى، فإن المحتوى المعلوماتي لحدث ما عبارة عن دالة متناقصة يكون صفرها عندما يكون للحدث أقصى احتمال، مما يعني أنه لا توجد معلومات محفوظة في حدث يحدث دائمًا. يمكن نمذجة محتوى المعلومات بالدالة $$I_f(e) = \log(\frac{1}{f(e)})=-\log(f(e)).$$ تقيس **الإنتروبيا (Entropy)** القيمة المتوقعة للمحتوى المعلوماتي وهي $$\mathbb E_f[I_f(\mathcal X)]=-\sum_{k \in \mathcal X} f(k) \log(f(k)).$$ أخيراً، تُعرف **الإنتروبيا المتقاطعة (Cross-entropy)** بين التوزيع $$p$$ و $$q$$ كـ $$H(p, q) = \mathbb E_p[I_q(\mathcal X)] = -\sum_{k \in \mathcal X} p(k) \log(q(k)).$$

يمكن ملاحظة أن سالب لوغاريتم الاحتمالية يجمع الإنتروبيا المتقاطعة لجميع العناصر في مجموعة البيانات $$\mathcal D$$، أي أن الاحتمال $$p(k)$$ هو $$\mathbb 1(y=k)$$، و $$q(k)=f_k(x)$$ باستخدام التعريف السابق للإنتروبيا المتقاطعة $$H(p, q)$$. والخاصية التي يجب ملاحظتها هي أن $$y$$ و $$x$$ ثوابت في المجموع الداخلي، والمتغير $$k$$ يمر على جميع الفئات. لذلك، فإن تقليل لوغاريتم الاحتمالية هو تقليل للإنتروبيا المتقاطعة، والتي تعمل كدالة الخسارة $$L$$ في [مسألة التحسين.](/NLP-Course-Ar/topics/02Vocabulary.html#eq:supervised-learning-optimization)

$$\begin{eqnarray}
-\frac{\partial}{\partial w_j}  l_{f_\mathcal Y}(f) &=& \frac{\partial}{\partial w_j} \sum_{(x, y) \in \mathcal D} \overbrace{- \sum_{k=1}^K \mathbb 1(k=y) \log f_k(x)}^{cross-entropy} \\
&=& -\sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \mathbb 1(k=y) \frac{\partial}{\partial w_j} \log f_k(x)\\
&=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\mathbb 1(k=y)}{f_k(x)} \frac{\partial}{\partial w_j} f_k(x)
\end{eqnarray}$$

الدالة $$f_k$$ لها قيد نظراً لأنها تأخذ مكان المعلمة $$\mathbf p$$ للتوزيع الفئوي؛ وهذا القيد هو $$\sum_k^K f_k(x) = 1$$ والذي يمكن الامتثال له بتقسيمها على عامل المعايرة كـ:

$$f_k(x) = \frac{h_k(w_k(x))}{\sum_{\ell=1}^K h_\ell(w_\ell(x))}$$

الخطوة التالية هي حساب المشتقة الجزئية بالنسبة لـ $$w_j$$

$$\begin{eqnarray}
\frac{\partial}{\partial w_j} f_k(x) &=& \frac{\partial}{\partial w_j} \frac{h_k(w_k(x))}{\sum_{\ell=1}^K h_\ell(w_\ell(x))}\\
&=& \frac{\sum_{\ell=1}^K h_\ell(w_\ell(x)) \frac{\partial}{\partial w_j} h_k(w_k(x)) - h_k(w_k(x)) \frac{\partial}{\partial w_j} \sum_{\ell=1}^K h_\ell(w_\ell(x))}{(\sum_{\ell=1}^K h_\ell(w_\ell(x)))^2}\\
&=& \frac{\sum_{\ell=1}^K h_\ell(w_\ell(x)) \frac{\partial}{\partial w_j} h_k(w_k(x)) - h_k(w_k(x)) \frac{\partial}{\partial w_j} h_j(w_j(x))}{(\sum_{\ell=1}^K h_\ell(w_\ell(x)))^2}.
\end{eqnarray}$$

وبالتعويض بالمشتقة الجزئية لـ $$f_k$$ في سالب لوغاريتم الاحتمالية نحصل على:

$$\begin{eqnarray}
-\frac{\partial}{\partial w_j}  l_{f_\mathcal Y}(f) &=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\mathbb 1(k=y)}{f_k(x)} \frac{\partial}{\partial w_j} f_k(x)\\
&=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K \frac{\mathbb 1(k=y)}{h_k(w_k(x))}   \frac{\sum_{\ell=1}^K h_\ell(w_\ell(x)) \frac{\partial}{\partial w_j} h_k(w_k(x)) - h_k(w_k(x)) \frac{\partial}{\partial w_j} h_j(w_j(x))}{\sum_{\ell=1}^K h_\ell(w_\ell(x))}
\end{eqnarray}$$

## الانحدار اللوجستي متعدد الحدود (Multinomial Logistic Regression)

من أجل التقدم في الاشتقاق، يلزم إجراء بعض الافتراضات؛ والافتراض الذي ينتج خوارزمية الانحدار اللوجستي متعدد الحدود هو أن $$h_k$$ عبارة عن أس، أي $$h_k(x)= \exp(x)$$ و $$f_k$$ الناتجة هي دالة softmax.

$$\begin{eqnarray}
h_k(w_k(x)) &=& \exp(w_k(x))\\
f_k(w_k(x)) &=& \frac{\exp(w_k(x))}{\sum_\ell \exp(w_\ell(x))}
\end{eqnarray}$$

إن استخدام $$f_k$$ كدالة softmax في سالب لوغاريتم الاحتمالية ينتج التالي:

$$\begin{eqnarray}
-\frac{\partial}{\partial w_j}  l_{f_\mathcal Y}(f) &=&  - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\mathbb 1(k=y)}{\exp(w_k(x))} \frac{\sum_{\ell=1}^K \exp(w_\ell(x)) \frac{\partial}{\partial w_j} \exp(w_k(x)) - \exp(w_k(x)) \exp(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)}{\sum_{\ell=1}^K \exp(w_\ell(x))}\\
&=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\mathbb 1(k=y)}{\exp(w_k(x))} \frac{\sum_{\ell=1}^K \exp(w_\ell(x)) \exp(w_k(x)) \frac{\partial}{\partial w_j} w_k(x) - \exp(w_k(x)) \exp(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)}{\sum_{\ell=1}^K \exp(w_\ell(x))}\\
&=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \mathbb 1(k=y) \frac{\sum_{\ell=1}^K \exp(w_\ell(x))  \frac{\partial}{\partial w_j} w_k(x) - \exp(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)}{\sum_{\ell=1}^K \exp(w_\ell(x))}\\
&=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \mathbb 1(k=y) \left[ \frac{\partial}{\partial w_j} w_k(x) - f_j(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)\right]\\
&=& - \sum_{(x, y) \in \mathcal D}  \mathbb 1(j=y) \left[ \frac{\partial}{\partial w_j} w_j(x) - f_j(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)\right] + \sum_{k \neq j}^K  \mathbb 1(k=y) \left[ \frac{\partial}{\partial w_j} w_k(x) - f_j(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)\right]\\
&=& - \sum_{(x, y) \in \mathcal D}  \mathbb 1(j=y) \left[ 1 - f_j(w_j(x)) \right] \frac{\partial}{\partial w_j} w_j(x) + \sum_{k \neq j}^K  -\mathbb 1(k=y) f_j(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)\\
&=& - \sum_{(x, y) \in \mathcal D}  \left( \mathbb 1(j=y) - f_j(w_j(x)) \right) \frac{\partial}{\partial w_j} w_j(x)\\
&=& \sum_{(x, y) \in \mathcal D}  \left( f_j(w_j(x)) - \mathbb 1(j=y) \right) \frac{\partial}{\partial w_j} w_j(x).
\end{eqnarray}$$

## الانحدار اللوجستي (Logistic Regression)

من ناحية أخرى، يتم الحصول على خوارزمية الانحدار اللوجستي عندما يفتُرض أن $$f_1$$ هي الدالة السجمية (Sigmoid function) وهناك فئتان؛ علاوة على ذلك، فإن هذا الافتراض يجعل من الممكن التعبير عن $$f_2$$ بدلالة $$f_1$$ كما يلي:

$$\begin{eqnarray}
f(x) &=& \frac{1}{1 + \exp(-x)}\\
f_1(x) &=& f(w(x)) \\
f_2(x) &=& 1 - f_1(x).
\end{eqnarray}$$

إن استخدام $$f_1$$ و $$f_2$$ والدالة السجمية في سالب لوغاريتم الاحتمالية ينتج التالي:

$$\begin{eqnarray}
-\frac{\partial}{\partial w_j}  l_{f_\mathcal Y}(f) &=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\mathbb 1(k=y)}{f_k(x)} \frac{\partial}{\partial w_j} f_k(x)\\
&=& - \sum_{(x, y) \in \mathcal D} \frac{\mathbb 1(1=y)}{f_1(x)} \frac{\partial}{\partial w_j} f_1(x) + \frac{\mathbb 1(2=y)}{1 - f_1(x)} \frac{\partial}{\partial w_j} \left(  1 - f_1(x) \right)\\
&=& - \sum_{(x, y) \in \mathcal D} \frac{\mathbb 1(1=y)}{f(w(x))} \frac{\partial}{\partial w_j} f(w(x)) - \frac{\mathbb 1(2=y)}{1 - f(w(x))} \frac{\partial}{\partial w_j} f(w(x))\\
&=& - \sum_{(x, y) \in \mathcal D} \left[ \frac{\mathbb 1(1=y)}{f(w(x))} - \frac{\mathbb 1(2=y)}{1 - f(w(x))} \right] \frac{\partial}{\partial w_j} f(w(x))\\
&=& - \sum_{(x, y) \in \mathcal D} \left[ \frac{\mathbb 1(1=y)}{f(w(x))} - \frac{\mathbb 1(2=y)}{1 - f(w(x))} \right] (1 - f(w(x))) f(w(x)) \frac{\partial}{\partial w_j} w(x)\\
&=& - \sum_{(x, y) \in \mathcal D} \left[ (1 - f(w(x)))\mathbb 1(1=y) - f(w(x))\mathbb 1(2=y) \right] \frac{\partial}{\partial w_j} w(x)\\
&=& - \sum_{(x, y) \in \mathcal D} (\mathbb 1(1=y) - f(w(x))) \frac{\partial}{\partial w_j} w(x)\\
&=& \sum_{(x, y) \in \mathcal D} (f(w(x)) - \mathbb 1(1=y)) \frac{\partial}{\partial w_j} w(x) 
\end{eqnarray}.$$

يمكن ملاحظة أن شكل سالب لوغاريتم الاحتمالية للانحدار اللوجستي متعدد الحدود والانحدار اللوجستي متطابقان؛ الفرق الوحيد هو وجود دالة $$w$$ لكل فئة في الحالة متعددة الحدود، ودالة واحدة فقط للانحدار اللوجستي.

# تصنيف النصوص - الانحدار اللوجستي (Logistic Regression)

بالإضافة إلى ذلك، لم يكن هناك افتراض يتعلق بشكل $$w(x)$$; وبالنظر إلى أن المسألة هي تصنيف النصوص، فإن المتغير $$x$$ يتوافق مع النص. ومع ذلك، فإن التعريف القياسي للانحدار اللوجستي متعدد الحدود والانحدار اللوجستي هو أن الدالة $$w$$ عبارة عن دالة خطية، أي $$w(x) = \mathbf w \cdot \mathbf x + w_0$$ حيث $$\mathbf w \in \mathbb R^d$$، و $$\mathbf x \in \mathbb R^d$$، و $$w_0 \in \mathbb R$$. وبالتالي، يتطلب الأمر تعريف دالة $$m: text \rightarrow \mathbb R^d$$ بحيث $$m(x) \in \mathbb R^d$$; وينتج عن ذلك أن الانحدار اللوجستي متعدد الحدود في هذه المسألة يكون:

$$\mathbb P(\mathcal Y=k \mid \mathcal X=x) = \frac{\exp(\mathbf w_k m(x) + w_{k_0})}{\sum_{j=1}^K \exp(\mathbf w_j m(x) + w_{k_0})}.$$

يعمل المقام في المعادلة السابقة كعامل معايرة، وتكون الفئة المتنبأ بها ثابتة لهذا العامل. بالإضافة إلى ذلك، فإن لوغاريتم $$\mathbb P(\mathcal Y=k \mid \mathcal X=x)$$ لا يؤثر على فئة التنبؤ بالقاعدة $$\textsf{class(x)} = \textsf{arg max}_k \mathbb P(\mathcal Y=k \mid \mathcal X=x).$$ وبالنظر إلى هذه العوامل، يتم التنبؤ بالفئة كـ:

$$\textsf{class(x)} = \textsf{arg max}_k \mathbf w_k m(x) + w_{k_0}.$$

كان [النهج الأول](/NLP-Course-Ar/topics/06TextCategorization.html#sec:tc-categorical) المتبع لمعالجة مسألة تصنيف النصوص هو استخدام مبرهنة بايز $$(\mathbb P(\mathcal Y \mid \mathcal X) = \frac{\mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)}{\mathbb P(\mathcal X)})$$ حيث تفُترض الاحتمالية ($$\mathbb P(\mathcal X \mid \mathcal Y)$$) لتكون توزيعاً فئويًا. يُعرّف التوزيع الفئوي بمتجه $$\mathbf p \in \mathbb R^d$$ حيث $$d$$ هي المخرجات المختلفة للتوزيع. الاحتمالية هي توزيع فئوي معطى الفئة $$\mathcal Y$$، لذلك هناك معلمة $$\mathbf p$$ لكل فئة، والتي يمكن تحديدها بدليل فرعي $$k$$، مثلًا $$\mathbf p_k$$ هي المعلمة المقابلة للفئة $$k$$. وتُقدر المعلمات بافتراض الاستقلالية، أي $$\mathbb P(\mathcal X=w_1,w_2,\ldots,w_\ell \mid \mathcal Y) = \prod_i^\ell \mathbb P(w_i \mid \mathcal Y),$$ حيث $$w_i$$ هو الرمز $$i$$ في النص.

الدليل $$\mathbb P(\mathcal X)$$ هو عامل معايرة في مبرهنة بايز، لذلك لا يؤثر على الفئة المتنبأ بها؛ علاوة على ذلك، فإن اللوغاريتم لا يغير قيمة التنبؤ. وبإدراج هذه التحويلات في مبرهنة بايز، نحصل على:

$$\log \mathbb P(\mathcal Y \mid \mathcal X=w_1,w_2,\ldots,w_\ell) \propto \sum_i^\ell \log \mathbb P(w_i \mid \mathcal Y) + \log \mathbb P(\mathcal Y);$$

والتي يمكن التعبير عنها باستخدام المعلمة، $$\mathbf p_k$$، للتوزيع الفئوي، وتكرار كل رمز كما يلي:

$$\log \mathbb P(\mathcal Y=k \mid \mathcal X=x) \propto \sum_i^d \log(\mathbf p_{k_i}) \textsf{freq}_i(x) + \log \mathbb P(\mathcal Y),$$

حيث تحسب $$\textsf{freq}_i(x)$$ تكرار الرمز المحدد بالفهرس $$i$$ في النص $$x$$. ولتوضيح التشابه بين المعادلة السابقة والمعادلة التي تم الحصول عليها مع الانحدار اللوجستي متعدد الحدود، فمن المناسب التعبير عنها باستخدام المتجهات، أي $$\log \mathbb P(\mathcal Y=k \mid \mathcal X=x) \propto \log(\mathbf p_k) \textsf{freq}(x) +  \log P(\mathcal Y),$$ حيث $$\log(\mathbf p_k) \in \mathbb R^d$$، و $$\textsf{freq}(x) \in \mathbb R^d$$، و $$\log \mathbb P(\mathcal Y=k) \in \mathbb R.$$ لذلك، فإن المعلمتين $$\log(\mathbf p_k)$$ و $$\log \mathbb P(\mathcal Y=k)$$ تكافآن $$\mathbf w$$ و $$w_0$$ في الانحدار اللوجستي (متعدد الحدود)، و $$m(x)$$ هو التكرار، $$\textsf{freq}(x)$$، في منهج التوزيع الفئوي.

## خوارزمية الانحدار التدرجي (Gradient Descent Algorithm)

خوارزمية الانحدار التدرجي خيار لتقدير المعلمتين $$\mathbf w$$ و $$w_0$$ في الانحدار اللوجستي (متعدد الحدود) وبشكل عام أي خوارزمية تستخدم دالة Sigmoid أو Softmax كخطوة أخيرة. الخطوة المفقودة هي تعريف الدالة $$m$$؛ والمنهج الذي تم استخدامه في التوزيع الفئوي هو استخدام التكرار كـ $$m$$.

يعتمد الكود التالي على الفئة `TextModel` لتمثيل النص في فضاء متجهي، حيث يكون مكون المتجه هو تكرار كل رمز من المدخلات.

يقرأ الكود التالي مجموعة بيانات ويرمز الفئات إلى معرفات فريدة؛ الأول هو الصفر.

```python
D = [(x['text'], x['klass']) for x in tweet_iterator(TWEETS)]
y = [y for _, y in D]
le = LabelEncoderWrapper().fit(y)
y = le.transform(y)
```

بمجرد وجود مجموعة البيانات في $$\mathcal D$$، يمكن استخدامها لتدريب الفئة `TextModel,` وهو ما يمكن القيام به بالطريقة `fit.` والتدريب هو عملية ربط كل رمز بمعرف وتحديد حجم المفردات في $$\mathcal D$$.

```python
tm = TextModel(token_list=[-1], 
               weighting='tf').fit([x for x, _ in D])
```

يمكن استخدام المثيل `tm` لتمثيل أي نص في فضاء المتجهات؛ بالنظر إلى أنه متجه مبعثر (Sparse vector)، فإنه يخرج الأبعاد فقط حيث تختلف القيمة عن الصفر، على سبيل المثال، النص *buenos dias dias* يتم تمثيله كـ:

```python
vec = tm['buenos dias dias']
vec
[(263, 0.3333333333333333), (87, 0.6666666666666666)]
```

حيث المكون المحدد كـ 263 (*buenos*) له القيمة $$0.33$$ والمكون 87 (*dias*) له $$0.66$$؛ والخاصية المحددة للطريقة هي أن التكرار تمت معايرته.

تحتوي `TextModel` على الطريقة `transform` التي يمكنها تحويل قائمة من النصوص إلى فضاء المتجهات؛ مخرج الطريقة عبارة عن مصفوفة مبعثرة. يمكن استخدام الطريقة كما يلي.

```python
X = tm.transform(['buenos dias dias'])
X.shape
(1, 3291)
```

يمكن دمج طريقة التحويل مع تنفيذ الانحدار اللوجستي متعدد الحدود لـ sklearn واختبارها تحت التحقق المتقاطع k-fold لحساب أدائها في $$\mathcal D$$.

```python
folds = StratifiedKFold(shuffle=True, random_state=0)
hy = np.empty(len(D))
for tr, val in folds.split(D, y):
    _ = [D[x][0] for x in tr]
    X = tm.transform(_)
    m = LogisticRegression(multi_class='multinomial').fit(X, y[tr])
    _ = [D[x][0] for x in val]
    hy[val] = m.predict(tm.transform(_))
ci = bootstrap_confidence_interval(y, hy)
ci
(0.2839760475399691, 0.30881116416736665)
```

توضح سحابة الكلمات التالية التكرار في [Semeval-2017 Task 4](https://aclanthology.org/S17-2088.pdf). يمكن ملاحظة أن الكلمات الأكثر تكرارًا هي رموز يمكن اعتبارها كلمات توقف.

![Term Frequency Semeval 2017 Task 4](/NLP-Course-Ar/assets/images/semeval2017_tf.png)

## تكرار المصطلح - مقلوب تكرار المستند (TF-IDF)

تكرار المصطلح ليس مخطط الأوزان الوحيد المتاح؛ حيث يمكن تطوير إجراءات مختلفة يمكن استخدامها لتحويل النص إلى فضاء متجهي. وإحدى الأكثر شعبية هي تكرار المصطلح - مقلوب تكرار المستند (*tf-idf*). يُعرّف tf-idf بأنه حاصل ضرب تكرار المصطلح، $$\textsf{tf}_i(x),$$ ومقلوب تكرار المستند $$\textsf{idf}_i(x).$$ يُعرّف تكرار المصطلح كـ:

$$\textsf{tf}_i(x) = \frac{\sum_{w \in x} \mathbb 1(w = i)}{\mid x \mid},$$

حيث $$x$$ هو النص المقطع الممثل كمجموعة متعددة، باستخدام المعرفات بدلاً من الرمز. من ناحية أخرى، فإن مقلوب تكرار المستند هو

$$\textsf{idf}_i(\mathcal D) = \log \frac{\mid \mathcal D \mid}{\sum_{x \in \mathcal D} \mathbb 1(i \in x) },$$

حيث $$\mathcal D$$ هي مجموعة البيانات المستخدمة لتدريب الخوارزمية و $$x$$ هو النص المقطع الممثل كمجموعة متعددة.

تستخدم الفئة `TextModel` كمخطط ترجيح افتراضي tf-idf، والذي يمكن اختباره بالكود التالي.

```python
tm = TextModel(token_list=[-1]).fit([x for x, _ in D])
vec = tm['buenos dias']
vec
[(263, 0.7489370345067511), (87, 0.6626411686155892)]
```

يمكن تقدير أداء مخطط الترجيح هذا على $$\mathcal D$$ بالكود التالي.

```python
hy = np.empty(len(D))
for tr, val in folds.split(D, y):
    _ = [D[x][0] for x in tr]
    X = tm.transform(_)
    m = LogisticRegression(multi_class='multinomial').fit(X, y[tr])
    # m = LinearSVC().fit(X, y[tr])
    _ = [D[x][0] for x in val]
    hy[val] = m.predict(tm.transform(_))

ci = bootstrap_confidence_interval(y, hy)
ci
(0.31927898144547495, 0.34791512559623444)
```

يكتمل وصف مقلوب تكرار المستند بسحابة الكلمات المحصول عليها مع مجموعة بيانات [Semeval-2017 Task 4](https://aclanthology.org/S17-2088.pdf). يمكن ملاحظة أن الكلمات ذات الأوزان الأعلى تكون غير متكررة.

![Inverse Document Frequency Semeval 2017 Task 4](/NLP-Course-Ar/assets/images/semeval2017_idf.png)

# المختبر التطبيقي: تصنيف النصوص بتمثيل حقيبة الكلمات (Lab: Bag of Words Text Categorization)
{: #sec:lab-bow}

قدم القسم السابق وهذا القسم العناصر اللازمة لإنشاء مصنف نصوص باستخدام تمثيل حقيبة الكلمات (Bag of Words - BoW) (أي الدالة $$m$$ المحددة سابقًا). حان الوقت لإنهاء كل هذا ووصف كيفية استخدام فئة بايثون BoW التي تنفذ ذلك.

الخطوة الأولى هي تهيئة تمثيل BoW؛ يمكن القيام بذلك بطريقتين مختلفتين. الأولى هي استخدام تمثيل BoW مدرب مسبقًا والذي تم تدريبه على نصف مليون تغريدة في كل لغة متاحة. والثانية هي تهيئة النموذج بالبيانات المستخدمة لملائمة مصنف النص.

يمكن استدعاء تمثيل BoW المدرب مسبقًا، كما هو موضح في الإرشادات التالية.

```python
bow = BoW(lang='en')
```

يمكن استخدام المثيل `bow` فورًا لأنه يستخدم نموذجًا مدربًا مسبقًا. تستقبل الطريقة `transform` قائمة نصوص ليتم تحويلها في تمثيل BoW؛ على سبيل المثال، يحول الكود التالي النص _good morning_ إلى التمثيل.

```python
bow.transform(['good morning'])
<1x16384 sparse matrix of type '<class 'numpy.float64'>'
	with 35 stored elements in Compressed Sparse Row format>
```

يُلاحظ أن المصفوفة لها أبعاد $$1 \times 16384$$، والتي تتوافق مع نص واحد، والفضاء في $$\mathbb R^{16384}$$ ، مما يعني أن هناك $$16384$$ من الرموز في المفردات. المصفوفة بتدوين مبعثر لأن الرموز التي تظهر في النص فقط هي التي تحتوي على قيمة تختلف عن الصفر.

يمكن رؤية المعاملات في الخاصية `data` الموضحة أدناه.

```python
X = bow.transform(['good morning'])
X.data
array([0.0418059 , 0.04938095, 0.06065429, 0.06554808, 0.07118364,
       0.07193482, 0.07464737, 0.07582207, 0.07998881, 0.08907673,
       0.1158131 , 0.1199745 , 0.12253494, 0.12477663, 0.13641194,
       0.13701824, 0.14673512, 0.17245292, 0.17909476, 0.18174817,
       0.18521041, 0.19004645, 0.19137817, 0.19414765, 0.19702447,
       0.20182253, 0.20188162, 0.20262674, 0.21067728, 0.24080574,
       0.24246672, 0.24981461, 0.26030563, 0.26209842, 0.27140846])
```

يمكن ملاحظة أن 35 رمزًا فقط تحتوي على قيمة تختلف عن الصفر. وتتوافق هذه الرموز مع تلك المحصول عليها من النص والموجودة أيضًا في مفردات التمثيل. يستخدم BoW مخطط ترجيح TF-IDF؛ ومع ذلك، يتم معايرة جميع المتجهات لتكون متجه وحدة، كما يمكن التحقق من ذلك في المثال السابق.

تُخزن الرموز غير الصفرية في الخاصية `indices;` ومع ذلك، فإنه تمثل الفهرس في فضاء المتجهات. تحتوي الخاصية `BoW.names` على الرمز الفعلي المرتب مكافئًا لفضاء المتجهات. يوضح الكود التالي الرموز المستخدمة لتمثيل النص _good morning_.

```python
' '.join([bow.names[x] for x in X.indices])
'q:in q:d~ q:~m q:ng q:or q:g~ q:ing q:ng~ q:ing~ q:~g q:oo q:ni q:mo q:go q:~go q:od q:~mo q:rn q:od~ q:ood q:mor q:nin q:ood~ q:ning q:goo q:~goo q:good q:~mor good q:orn q:rni q:rnin q:orni q:morn morning'
```

الإجراء الثاني لتهيئة الفئة BoW هو استخدام نفس مجموعة البيانات لتدريب المصنف؛ ويتم توضيح هذا النهج باستخدام مجموعة بيانات اصطناعية موجودة في حزمة EvoMSA. مسار مجموعة البيانات في المتغير TWEETS ويمكن قراءته باستخدام الدالة `tweet_iterator`، كما هو موضح في الإرشادات التالية.

```python
D = list(tweet_iterator(TWEETS))
```

تم إنشاء الفئة BoW بـ المعلمة `pretrain` معينة على false، كما هو موضح بعد ذلك.

```python
bow = BoW(pretrain=False)
```

يمكن التحقق من أنه، في هذا الوقت، من المستحيل تحويل أي نص إلى تمثيلات BoW لأن نموذج BoW لم يتم تدريبه بعد.

مجموعة البيانات في `D` عبارة عن مجموعة قطبية باللغة الإسبانية بأربعة تسميات: إيجابي، سلبي، محايد، ولا شيء. النص في الكلمة المفتاحية `text,` والتسمية المرتبطة بها في الكلمة المفتاحية `klass`؛ يمكن تغيير هذا وضبطه على القيم المناسبة للمشتركات `key` و `labe_key` لمنشئ BoW.

تحتوي D على جميع المكونات لتدريب مصنف نصوص؛ ويمكن القيام بذلك بالطريقة `fit.` داخليًا، ستستدعي الطريقة `fit` الطريقة `b4msa_fit` لتقدير معلمات تمثيل BoW قبل تدريب المصنف. يوضح الكود التالي كيفية ملائمة مصنف النص.

```python
bow.fit(D)
```

يمكن استخدام المتغير `bow` للتنبؤ بقطبية نص معين؛ ويمكن القيام بذلك بالطريقة `predict.` على سبيل المثال، يتنبأ الكود التالي بالنص _buenos días_ (صباح الخير).

```python
bow.predict(['buenos dias'])
array(['P'], dtype='<U4')
```

تستقبل الطريقة `predict` قائمة نصوص، ويمكن ملاحظة أن النص _buenos días_ يتنبأ به كـ P، والتي تتوافق مع الفئة الإيجابية.
