---
layout: default
title: نموذج اللغة N-Gram
nav_order: 4
---

# نموذج اللغة N-Gram
{: .fs-10 .no_toc }

## المحتويات
{: .no_toc .text-delta }

1. TOC
{:toc}

## المكتبات المستعملة
{: .no_toc .text-delta }
```python
import numpy as np
from matplotlib import pylab as plt
from microtc.utils import tweet_iterator
from os.path import join
from collections import Counter, defaultdict
from wordcloud import WordCloud as WC
```

## تثبيت المكتبات الخارجية
{: .no_toc .text-delta }

```bash
pip install microtc
pip install evomsa
pip install text_models
```

---

# المقدمة

يخصص نموذج اللغة (Language Model - LM) احتمالات للكلمات (الرموز Tokens)، أو الجمل، أو المستندات. الاستخدام المباشر لـ LM هو تقدير احتمال ملاحظة نص معين، ويمكن استخدامه أيضًا لتوليد النصوص، وبشكل عام، فإنه ينمذج ديناميكيات اللغة. ومن المهم الإشارة إلى أن معظم مهام **فهم اللغة الطبيعية** (Natural Language Understanding - NLU) تعتمد على نموذج اللغة (LM).

يتعامل نموذج اللغة مع نمذجة الاحتمال متعدد المتغيرات $$\mathbb P(\mathcal X_1, \mathcal X_2, \ldots, \mathcal X_\ell)$$ لملاحظة $$\ell$$ من الكلمات (الرموز). وكما هو متوقع، فإن هذه المتغيرات العشوائية $$\ell$$ تعتمد على بعضها البعض (انظر تعريف المفهوم المكمل، أي [الاستقلالية](/NLP-Course-Ar/topics/03Collocations.html#sec:independence-marginal).)، وللعمل معها، يلزم تعريف مفهوم الاحتمال الشرطي.

# الاحتمال الشرطي (Conditional Probability)
{: #sec:conditional-probability }

يُعرّف الاحتمال الشرطي لمتغيرين عشوائيين $$\mathcal X$$ و $$\mathcal Y$$ كـ:

$$\mathbb P(\mathcal Y \mid \mathcal X) = \frac{\mathbb P(\mathcal X, \mathcal Y)}{\mathbb P(\mathcal X)},$$

إذا كان $$\mathbb P(\mathcal X) > 0.$$

يسمح التعريف بتحديد $$\mathbb P(\mathcal X, \mathcal Y) = \mathbb P(\mathcal Y \mid \mathcal X) \mathbb P(\mathcal X)،$$ وهو أمر مفيد للنصوص المكونة من كلمتين. وتتضمن الحالة العامة $$\ell$$ من الكلمات، والتي يمكن تعريفها باستخدام قاعدة السلسلة للاحتمالات (Probability chain rule).

$$\begin{eqnarray}
\mathbb P(\mathcal X_1, \ldots, \mathcal X_\ell) &=& \mathbb P(\mathcal X_\ell \mid \mathcal X_1, \ldots, \mathcal X_{\ell -1}) \mathbb P(\mathcal X_1, \ldots, \mathcal X_{\ell - 1})\\ 
\mathbb P(\mathcal X_1, \ldots, \mathcal X_{\ell - 1}) &=& \mathbb P(\mathcal X_{\ell - 1} \mid \mathcal X_1, \ldots, \mathcal X_{\ell - 2}) \mathbb P(\mathcal X_1, \ldots, \mathcal X_{\ell - 2}) \\
&\vdots& \\
\mathbb P(\mathcal X_1, \mathcal X_2) &=& \mathbb P(\mathcal X_2 \mid \mathcal X_1) \mathbb P(\mathcal X_1)
\end{eqnarray}.$$

تظهر المساواة الأولى في نظام المعادلة السابق خاصية مثيرة؛ حيث تحسب احتمال الكلمة التالية، $$\ell$$، بالنظر إلى تاريخ مكون من $$\ell -1 $$ من الكلمات؛ أي:

$$\mathbb P(\mathcal X_\ell \mid \mathcal X_1, \ldots, \mathcal X_{\ell -1}) = \frac{\mathbb P(\mathcal X_1, \ldots, \mathcal X_\ell)}{\mathbb P(\mathcal X_1, \ldots, \mathcal X_{\ell - 1})},$$ 

حيث $$\mathbb P(\mathcal X_1, \ldots, \mathcal X_{\ell - 1}) = \sum_x \mathbb P(\mathcal X_1, \ldots, \mathcal X_{\ell - 1}, \mathcal X_\ell = x)$$ هو التوزيع الهامشي (Marginal distribution).

# نموذج اللغة N-Gram (N-Gram Language Model)

تقليديًا، تكون التوزيعات الاحتمالية متعددة المتغيرات ثابتة فيما يتعلق بعدد المتغيرات؛ ومع ذلك، فإن عدد الكلمات في الجملة أو المستند متغير. ومع ذلك، فإن التبعية بين الكلمة الأخيرة والكلمة الأولى في نص طويل تكون مهملة.

حتى عندما يكون عدد المتغيرات ثابتًا، أي يتم الإبقاء على $$\ell$$ ثابتًا في النموذج، فإن النتيجة هي أنه لا يمكن تمثيل نص أطول من $$\ell$$ كلمة. وهناك قلق آخر يأتي من تقدير معلمات التوزيع متعدد المتغيرات. حيث يتبع $$\mathcal X$$ توزيعًا فئويًا (Categorical distribution) مع $$d$$ من المخرجات الممكنة. الحد الأدنى لعدد الأمثلة المطلوبة لملاحظة مخرجات $$d$$ هو $$d^1$$؛ وفي حالة متغيرين، أي $$\ell=2$$، يلزم على الأقل $$d^2$$، وبشكل عام، لـ $$\ell$$ من المتغيرات يلزم $$d^\ell$$ من الأمثلة.

والنتيجة هي أنه بالنسبة لـ $$\ell$$ صغيرة نسبياً، يحتاج المرء إلى مجموعة بيانات ضخمة للحصول على معلومات كافية لتقدير التوزيع متعدد المتغيرات. لقد رأينا هذا السلوك في [مصفوفة التوارد](/NLP-Course-Ar/topics/03Collocations.html#tab:co-occurrence) حيث معظم عناصر المصفوفة أصفار؛ فقط $$0.021$$% من المصفوفة يختلف عن الصفر.

يمكن التعامل مع القيود المذكورة أعلاه عن طريق التقريب للاحتمال الشرطي للكلمة التالية. بدلاً من استخدام $$\ell - 1$$ كلمة لقياس احتمال الحصول على الكلمة $$\ell$$، يتم تثبيت السجل/التاريخ ليشمل فقط آخر $$n$$ كلمات، أي:

$$\mathbb P(\mathcal X_\ell \mid \mathcal X_1, \ldots, \mathcal X_{\ell -1}) \approx \mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell - n + 1}, \ldots, \mathcal X_{\ell -1}).$$ 


# ثنائيات الكلمات (Bigrams)

يُعرف نموذج n-gram لـ $$n=2$$ باسم ثنائي الكلمات (Bigram)، وصياغته هي $$\mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell-1}).$$ لقد عملنا بشكل مكثف مع ثنائيات الكلمات، وإن كان ذلك في اتجاه مختلف: [البحث عن المتلازمات اللفظية](/NLP-Course-Ar/topics/03Collocations.html). بالإضافة إلى ذلك، فإن مجموعة البيانات التي تم الحصول عليها من مكتبة `text_models` لا تتبع تعريف Bigrams الخاص بنماذج اللغة؛ حيث يستخدم LM bigram كلمتين متتاليتين كمدخلات. وعلى العكس من ذلك، فإن الكلمتين في نماذج `text_models` لم يكن لهما ترتيب وكانا عبارة عن تركيب لجميع الأزواج في كل تغريدة.

يُعرّف نموذج Bigram كـ:

$$\mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell - 1}) = \frac{\mathbb P(\mathcal X_{\ell -1}, \mathcal X_\ell)}{\mathbb P(\mathcal X_{\ell - 1})}.$$

تم وصف إجراء تقدير قيم $$\mathbb P(\mathcal X_{\ell -1}, \mathcal X_\ell)$$ و $$\mathbb P(\mathcal X_{\ell - 1})$$ [سابقًا.](/NLP-Course-Ar/topics/03Collocations.html#sec:bivariate-distribution) وكان المثال الذي شوهد عبارة عن زوج من النرد حيث توجد تبعية بين $$\mathcal X_r=2$$ و $$\mathcal X_c=1$$. يمكن تعديل هذا المثال لربطه عن كثب بـ bigram LM بافتراض أن هناك لغة تحتوي فقط على أربع كلمات، متمثلة بـ $$\{0, 1, 2, 3\}$$. تُستخدم هذه الكلمات لتكوين ثنائيات كلمات تتبع توزيعًا فئويًا، حيث يكون الإجراء الذي يولد هذا التوزيع ثنائي المتغير هو التالي.

```python
d = 4
R = np.random.multinomial(1, [1/d] * d, size=10000).argmax(axis=1)
C = np.random.multinomial(1, [1/d] * d, size=10000).argmax(axis=1)
rand = np.random.rand
Z = [[r, 2 if r == 1 and rand() < 0.1 else c]
      for r, c in zip(R, C)
     if r != c or (r == c and rand() < 0.2)]
```

يمكن ملاحظة أن نقطة البداية هي المتغيرات $$\mathcal X_r$$ (أي $$\mathcal X_{\ell - 1}$$) و $$\mathcal X_c$$ (أي $$\mathcal X_\ell$$) التابعة لتوزيع فئوي مع $$\mathbf p_i = \frac{1}{d}$$ حيث $$d=4$$. ثم يتم أخذ عينات من التوزيع ثنائي المتغير على المتغير `Z` حيث يتم إسقاط 80% من الأمثلة التي يكون فيها $$\mathcal X_r=\mathcal X_c$$ ويتم استحثاث تبعية لـ $$\mathcal X_r=1$$ و $$\mathcal X_c=2$$.

القيم المقدرة لـ $$\mathcal P(\mathcal X_r, \mathcal X_c)$$ هي:
{: #bivarite-distribution-bigrams }

$$
\begin{pmatrix}
0.0149 & 0.0771 & 0.0791 & 0.0782 \\
0.0663 & 0.0131 & 0.0939 & 0.0730 \\
0.0810 & 0.0799 & 0.0146 & 0.0796 \\
0.0797 & 0.0742 & 0.0780 & 0.0172 \\
\end{pmatrix}.
$$

التوزيع الهامشي هو $$\mathbb P(\mathcal X_r) = (0.249374, 0.246370, 0.255133, 0.249124)$$ ويمكن الحصول عليه كما يلي:

```python
M_r = W.sum(axis=1)
```

حيث `W` تحتوي على التوزيع ثنائي المتغير المقدر. ليس من الضروري الحصول على التوزيع الهامشي $$\mathbb P(\mathcal X_c) = (0.241988, 0.244367, 0.265648, 0.247997)$$؛ ومع ذلك، فإنه يُلاحظ أن التبعية المستحثة تؤثر على هذا الهامش وليس الهامش السابق.

يمكن تقدير الاحتمال الشرطي $$\mathbb P(\mathcal X_c \mid \mathcal X_r)$$ كـ:

```python
p_l = (W / np.atleast_2d(M_r).T)
```

والنتيجة معروضة في المصفوفة التالية:

$$
\begin{pmatrix}
0.0597 & 0.3092 & 0.3173 & 0.3138 \\
0.2693 & 0.0534 & 0.3811 & 0.2962 \\
0.3175 & 0.3131 & 0.0574 & 0.3121 \\
0.3201 & 0.2980 & 0.3131 & 0.0688 \\
\end{pmatrix}.
$$

## توليد المتواليات النصية (Generating Sequences)
{: #sec:generating-sequences }

يمكن استخدام الاحتمال الشرطي $$\mathbb P(\mathcal X_c \mid \mathcal X_r)$$ (المتغير `p_l`) والاحتمال الهامشي $$\mathbb P(\mathcal X_r)$$ (المتغير `M_r`) لتوليد نص. سيكون المثال أكثر واقعية إذا استخدمنا الأحرف بدلاً من الفهارس؛ ويمكن القيام بذلك بربط الفهرس بالسلسلة النصية، كما يتضح أدناه.

```python
id2word = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
```

من المفيد تعريف دالة (`cat`) تتصرف مثل توزيع فئوي؛ ويمكن القيام بذلك باستخدام توزيع متعدد الحدود (Multinomial distribution) باستخدام المعلمات الموضحة في الكود التالي.

```python
cat = lambda x: np.random.multinomial(1, x, 1).argmax()
```

شرط استخدام الاحتمال الشرطي هو الحاجة إلى كلمة بداية؛ الاحتمال الشرطي $$\mathbb P(\mathcal X_c \mid \mathcal X_r)$$ بمجرد معرفة قيمة $$\mathcal X_r$$. يمكننا أن نفترض أنه يمكن محاكاة الكلمة الأولى باستخدام الهامش $$\mathbb P(\mathcal X_r)$$ كما يتضح فيما يلي.

```python
w1 = cat(M_r)
```

بمجرد الحصول على كلمة البداية، فإنه يلزم التكرار للعدد المرغوب من المرات باستخدام الاحتمال الشرطي لتوليد الرمز التالي؛ ويمكن ملاحظة ذلك في الكود التالي.

```python
l = 25
text = [cat(M_r)]
while len(text) < l:
    next = cat(p_l[text[-1]])
    text.append(next)
text = " ".join(map(lambda x: id2word[x], text))
```

يتم تنفيذ الكود السابق (بما في ذلك التوزيع الهامشي) ثلاث مرات، ويمكن ملاحظة النتيجة في الجدول التالي.

|النص (Text)                                            |
|-------------------------------------------------|
|d a d c b d c d c d a b a b c d a d b d a d b c b|
|c a b a c a c d b d d a d c d b d b d b a b a b c|
|b a c b a b a c b d a d c b c d b c d c a c b d c|


## استخدام متوالية لتقدير $$\mathbb P(\mathcal X_r, \mathcal X_c)$$

تهدف معالجة اللغة الطبيعية إلى العثور على النموذج الذي يمكن استخدامه لتوليد النص وتقديره؛ لذلك، ليس من الواقعي الحصول على $$\mathbb P(\mathcal X_{1}, \mathcal X_{2}, \ldots, \mathcal X_\ell)$$ بشكل مباشر؛ ومع ذلك، يمكننا تقديره باستخدام الأمثلة. بالنظر إلى أن لدينا طريقة لتوليد النص، يمكننا توليد تسلسل طويل وتقدير معلمات التوزيع ثنائي المتغير منه.

الخطوة الأولى هي الحصول على تعيين من الكلمات إلى الأرقام. الخطوة الثانية هي استرجاع الكلمات، والخطوة الثالثة هي إنشاء ثنائيات الكلمات. ويمكن ملاحظة ذلك في الكود التالي.

```python
w2id = {v: k for k, v in id2word.items()}
lst = [w2id[x] for x in text.split()]
Z = [[a, b] for a, b in zip(lst, lst[1:])]
```

بقية الكود رأيناها سابقًا؛ ومع ذلك فإنه معروض أدناه لتسهيل القراءة.

```python
d = len(w2id)
W = np.zeros((d, d))
for r, c in Z:
    W[r, c] += 1
W = W / W.sum()
```

التوزيع ثنائي المتغير المقدر من التسلسل معروض في المصفوفة التالية. يمكن ملاحظة أن قيم هذه المصفوفة تشبه [المصفوفة](/NLP-Course-Ar/topics/04NGramLM.html#bivarite-distribution-bigrams) المستخدمة لتوليد التسلسل.

$$
\begin{pmatrix}
0.0150 & 0.0750 & 0.0790 & 0.0781 \\
0.0655 & 0.0113 & 0.0930 & 0.0715 \\
0.0869 & 0.0808 & 0.0156 & 0.0810 \\
0.0797 & 0.0743 & 0.0767 & 0.0165 \\
\end{pmatrix}.
$$

## الاحتمال المشترك (Joint Probability)

لدينا جميع العناصر لحساب الاحتمال المشترك لتسلسل معين، على سبيل المثال احتمال ملاحظة التسلسل *a d b c* هو $$\mathbb P(\text{"a d b c"}) = 0.008885$$؛ هذا ترميز مبسط والترميز الكامل سيكون $$\mathbb P(\mathcal X_1=a, \mathcal X_2=d, \mathcal X_3=b, \mathcal X_4=c) \approx \mathbb P(\mathcal X_1) \prod_{i=2}^4 \mathbb P(\mathcal X_{i} \mid \mathcal X_{i-1}).$$ يستخدم الكود التالي قاعدة سلسلة الاحتمالات لتقدير الاحتمال المشترك.

```python
text = 'a d b c'
lst = [w2id[x] for x in text.split()]
p = M_r[lst[0]]
for a, b in zip(lst, lst[1:]):
    p *= p_l[a, b]
```

يتم إكمال المثال السابق بتسلسل معروف، حسب التعريف، أن له احتمال أقل، حيث يختلف التسلسل فقط في الكلمة الأخيرة، واحتماله هو $$\mathbb P(\text{"a d b d"}) = 0.006907.$$

يقدم الإجراء الموصوف عملية نمذجة اللغة من البداية؛ يبدأ بافتراض أن اللغة تتولد من خوارزمية معينة، ثم تُستخدم الخوارزمية لتقدير توزيع ثنائي المتغير، والذي يُستخدم لإنتاج تسلسل. التسلسل هو قياس لنص مكتوب باللغة الطبيعية، ثم يتم استخدام التسلسل لتقدير توزيع ثنائي المتغير، ويمكننا مقارنة التوزيعين لتوضيح أنه حتى في عملية بسيطة، فإنه من غير العملي الحصول على مصفوفتين بنفس القيم.

# تجاوز الافتراضات (Overcoming Assumptions)

ومع ذلك، فإن بعض مكونات الصياغة السابقة غير واقعية لنمذجة اللغة. الأول هو أن مجموع الاحتمالات للمتواليات الممكنة لطول معين يساوي واحدًا، على سبيل المثال $$\sum_{x,y,z} \mathbb P(\mathcal X_{\ell-2}=x, \mathcal X_{\ell-1}=y, \mathcal X_\ell=z) = 1$$. والنتيجة هي أن هناك توزيع احتمالي لكل طول، وهي ليست ميزة مرغوبة لنموذج اللغة لأن طول الجملة متغير.

والثاني هو أنه لا يمكن تقدير الكلمة الأولى باستخدام الهامش $$\mathbb P(\mathcal X_r)$$؛ حيث لا يأخذ هذا التوزيع في الاعتبار أن بعض الكلمات تُستخدم بشكل أكثر تكرارًا لبدء الجملة. يمكن تضمين هذا التأثير في النموذج باستخدام رمز البداية، على سبيل المثال التسلسل *a b d* سيكون *$$\epsilon$$ a b d*.

يمكن رؤية مجموع احتمالات جميع الجمل الممكنة المكونة من ثلاث كلمات ورمز بداية في المعادلة التالية.

$$
\begin{eqnarray}
&&\sum_{x, y, z} \mathbb P(\mathcal X_1=\epsilon, \mathcal X_2=x, \mathcal X_3=y, \mathcal X_4=z) \approx \\
&&\sum_{x, y, z} \mathbb P(\mathcal X_4=z \mid \mathcal X_3=y) \mathbb P(\mathcal X_3=y \mid \mathcal X_2=x) \mathbb P(\mathcal X_2=x \mid \mathcal X_1=\epsilon) \mathbb P(\mathcal X_1=\epsilon) =\\
&&\mathbb P(\mathcal X_1=\epsilon) \sum_{x, y, z} \mathbb P(\mathcal X_4=z \mid \mathcal X_3=y) \mathbb P(\mathcal X_3=y \mid \mathcal X_2=x) \mathbb P(\mathcal X_2=x \mid \mathcal X_1=\epsilon) =\\
&&\mathbb P(\mathcal X_1=\epsilon) \sum_{x, y}  \mathbb P(\mathcal X_3=y \mid \mathcal X_2=x) \mathbb P(\mathcal X_2=x \mid \mathcal X_1=\epsilon)\sum_z \mathbb P(\mathcal X_4=z \mid \mathcal X_3=y) =\\
&&\mathbb P(\mathcal X_1=\epsilon) \sum_{x}  \mathbb P(\mathcal X_2=x \mid \mathcal X_1=\epsilon) \sum_y  \mathbb P(\mathcal X_3=y \mid \mathcal X_2=x)=\\
&&\mathbb P(\mathcal X_1=\epsilon) \sum_{x}  \mathbb P(\mathcal X_2=x \mid \mathcal X_1=\epsilon) = \mathbb P(\mathcal X_1=\epsilon) = 1
\end{eqnarray}
$$

يمكن ملاحظة أن تضمين رمز البداية لا يحل مشكلة وجود توزيع احتمالي لكل طول.

المشكلة الثالثة التي تم اكتشافها هي أن طول الجملة هو معلمة يتحكم فيها الشخص الذي يولد التسلسل؛ ومع ذلك، فإن طول الجملة يعتمد على محتوى الجملة، لذا فهو أيضًا متغير عشوائي. النهج الممكن لتضمين هذا السلوك في النموذج هو إضافة رمز النهاية.

يمكن التعبير عن الاحتمال المتراكم لجميع الإمكانيات لجملة مكونة من ثلاث كلمات برمز بداية ورموز نهاية (أي $$\epsilon_s$$ و $$\epsilon_e$$، على التوالي)، كما يلي:


$$
\begin{eqnarray}
&&\sum_{x, y, z} \mathbb P(\mathcal X_1=\epsilon_s, \mathcal X_2=x, \mathcal X_3=y, \mathcal X_4=z, \mathcal X_5=\epsilon_e) \approx \\
&&\mathbb P(\mathcal X_1=\epsilon_s) \sum_{x, y, z} \mathbb P(\mathcal X_5=\epsilon_e \mid \mathcal X_4=z) P(\mathcal X_4=z \mid \mathcal X_3=y) \mathbb P(\mathcal X_3=y \mid \mathcal X_2=x) \mathbb P(\mathcal X_2=x \mid \mathcal X_1=\epsilon_s)
\end{eqnarray}
$$

كما يتضح، فإن الإجراء المستخدم في المعادلة ذات رمز البداية ليس ممكنًا في المعادلة الحالية؛ وبالتالي، يلزم تحليل هذه المعادلة بشكل أكبر لمعرفة كيفية المتابعة. السمة الأولى التي يجب ملاحظتها هي أن $$\mathbb P(\mathcal X_1=\epsilon_s)=1$$. والسمة الثانية هي أن الكلمة الأولى هي دائمًا $$\epsilon_s$$؛ وبالتالي، فإن $$\mathbb P(\mathcal X_2=x \mid \mathcal X_1=\epsilon_s) = \mathbb P(\mathcal X_2=x)$$؛ وباستخدام هذه العناصر، نحصل على:

$$
\begin{eqnarray}
&&\sum_{x, y, z} \mathbb P(\mathcal X_1=\epsilon_s, \mathcal X_2=x, \mathcal X_3=y, \mathcal X_4=z, \mathcal X_5=\epsilon_e) \approx \\
&&\sum_{x, y, z} \mathbb P(\mathcal X_5=\epsilon_e \mid \mathcal X_4=z) P(\mathcal X_4=z \mid \mathcal X_3=y) \mathbb P(\mathcal X_3=y \mid \mathcal X_2=x) \mathbb P(\mathcal X_2=x) = \\
&&\sum_{y, z} \mathbb P(\mathcal X_5=\epsilon_e \mid \mathcal X_4=z) P(\mathcal X_4=z \mid \mathcal X_3=y) \sum_x \mathbb P(\mathcal X_3=y, \mathcal X_2=x) =\\
&&\sum_{y, z} \mathbb P(\mathcal X_5=\epsilon_e \mid \mathcal X_4=z) P(\mathcal X_4=z \mid \mathcal X_3=y) \mathbb P(\mathcal X_3=y) =\\
&&\sum_{z} \mathbb P(\mathcal X_5=\epsilon_e \mid \mathcal X_4=z) \sum_y P(\mathcal X_4=z, \mathcal X_3=y) = P(\mathcal X_5 = \epsilon_e)\\
\end{eqnarray}
$$

كما يتضح، فإن الاحتمال الإجمالي لا يجمع إلى $$1$$؛ بل يعتمد على احتمال اختيار رمز النهاية.

# نموذج اللغة لثنائيات الكلمات من التغريدات (Bigram LM from Tweets)

أبسط نموذج يمكننا إنشاؤه هو bigram LM؛ نقطة البداية هي وجود مدونة نصية (Corpus). -المدونة النصية المستخدمة في هذا المثال هي مجموعة من 50,000 تغريدة مكتوبة باللغة الإنجليزية.- بمجرد الحصول على المدونة النصية، يمكننا استخدامها لتقدير التوزيع ثنائي المتغير $$\mathbb P(\mathcal X_{\ell-1}, \mathcal X_\ell)$$ واستخدام الاحتمال الشرطي للحصول على $$\mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell -1}).$$

هناك مسارات مختلفة لحساب $$\mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell -1})$$ أحدها هو استخدام التكرار الخام للكلمات كما يلي:

$$\begin{eqnarray}
\mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell -1}) &=& \frac{\mathbb P(\mathcal X_{\ell-1}, \mathcal X_\ell)}{ \mathbb P(\mathcal X_{\ell-1})}\\
&=& \frac{\mathbb P(\mathcal X_{\ell-1}, \mathcal X_\ell)}{\sum_i \mathbb P(\mathcal X_{\ell-1}, \mathcal X_i)}\\
&=& \frac{\frac{C(\mathcal X_{\ell-1}, \mathcal X_\ell)}{N}}{\frac{\sum_i C(\mathcal X_{\ell-1}, \mathcal X_i)}{N}} \\
&=& \frac{C(\mathcal X_{\ell-1}, \mathcal X_\ell)}{\sum_i C(\mathcal X_{\ell-1}, \mathcal X_i)}\\
\end{eqnarray},$$

حيث $$C$$ هي مصفوفة التوارد.

تم إنشاء مصفوفة التوارد (المتغير `bigrams`) بالكود التالي؛ وكما يمكن ملاحظته لكل تغريدة، فإنه يتضمن رمز بداية ونهاية.

```python
fname = join('dataset', 'tweets-2022-01-17.json.gz')
bigrams = Counter()
for text in tweet_iterator(fname):
    text = text['text']
    words = text.split()
    words.insert(0, '<s>')
    words.append('</s>')
    _ = [(a, b) for a, b in zip(words, words[1:])]
    bigrams.update(_)
```

المصطلح $$\sum_i C(\mathcal X_{\ell-1}, \mathcal X_i)$$ هو تكرار الكلمة $$\mathcal X_{\ell-1}$$، أي $$C(\mathcal X_{\ell-1}) = \sum_i C(\mathcal X_{\ell-1}, \mathcal X_i)$$ والذي يتوافق مع المتغير `prev` في الكود التالي

```python
prev = dict()
for (a, b), v in bigrams.items():
    try:
        prev[a] += v
    except KeyError:
        prev[a] = v
```

يمكننا تخزين $$\mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell -1})$$ في قاموس متداخل وهو المتغير `P` في الكود التالي.

```python
P = defaultdict(Counter)
for (a, b), v in bigrams.items():
    next = P[a]
    next[b] = v / prev[a]
```

يمكن استخدام الاحتمال الشرطي $$\mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell -1})$$ لتوضيح الكلمة الأكثر احتمالاً في بداية الجملة، كما هو موضح في الشكل التالي.

![Word cloud probability given starting symbol](/NLP-Course-Ar/assets/images/wordcloud_prob_start.png)
<details markdown="block">
  <summary>
    كود سحابة الكلمات (Word cloud code)
  </summary>

```python
wc = WC().generate_from_frequencies(P['<s>'])
plt.imshow(wc)
plt.axis('off')
plt.tight_layout()
```
</details>

قدم قسم [توليد المتواليات النصية](#sec:generating-sequences) خوارزمية لتوليد جملة بالنظر إلى $$\mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell -1})$$؛ يمكن تمديد تلك الخوارزمية لتوليد جملة بمراعاة رمزي البداية والنهاية كما يمكن ملاحظته في الكود التالي.

```python
sentence = ['<s>']
while sentence[-1] != '</s>':
    var = P[sentence[-1]]
    pos = var.most_common(20)
    index = np.random.randint(len(pos))
    sentence.append(pos[index][0])
```

فيما يلي مثال لجملة تم توليدها بالإجراء السابق: *$$\epsilon_s$$ What happened before the one idiot or a few things to me up $$\epsilon_e$$*.

كما وصفنا سابقًا، يمكن استخدام قاعدة سلسلة الاحتمالات لتقدير احتمال الجملة. على سبيل المثال، يحدد الكود التالي دالة لحساب الاحتمال المشترك، أي احتمال الجملة؛ الفرق بين التنفيذ التالي والتنفيذ السابق هو تضمين رمزي البداية والنهاية.

```python
def joint_prob(sentence):
    words = sentence.split()
    words.insert(0, '<s>')
    words.append('</s>')
    tot = 1
    for a, b in zip(words, words[1:]):
        tot *= P[a][b]
    return tot

joint_prob('I like to play football')
8.491041580185946e-12
```

# الأداء وتقييم النموذج (Performance)

تم تخصيص هذا القسم لوصف نموذج اللغة (LM) والإجراء الخاص بتطويره. يمتلك هذا النهج أساسًا رياضيًا صلبًا؛ ومع ذلك، في الوقت نفسه، ومن أجل جعله ممكنًا، تم إجراء بعض الافتراضات. وبالتالي، يتساءل المرء عما إذا كانت تلك القرارات تؤثر على جودة LM وإلى أي درجة. أفضل طريقة لقياس تأثير تلك القرارات هي اختبار LM في التطبيق النهائي حيث يتم استخدامه؛ أي استخدام المقاييس المكتشفة لاختبار التطبيق، وقياس التأثير بشكل غير مباشر على معقدات LM في ذلك السيناريو.

ليس من الممكن دائمًا تضمين نماذج لغة مختلفة في التطبيق النهائي واختبار أيهما أفضل؛ هناك نهج آخر يكمن في استخدام مقياس أداء معين لاختبار LM المطور. سيكون النهج المباشر هو حساب الاحتمال المشترك في مجموعة بيانات أخرى واستخدام ذلك القياس لمقارنة LMs المختلفة. ومع ذلك، في الممارسة العملية، لا يتم استخدام الاحتمال المشترك؛ بدلاً من ذلك، يتم استخدام **الحيرة (Perplexity - PP)** والمُعرّفة كـ:

$$PP(\mathcal X_1, \ldots, \mathcal X_N) = \sqrt[N]{\frac{1}{\mathbb P(\mathcal X_1, \ldots, \mathcal X_N)}}.$$

إن الحيرة (PP) لـ bigram LM هي $$PP(\mathcal X_1, \ldots, \mathcal X_N) = \sqrt[N]{\frac{1}{\mathbb P(\mathcal X_1=\epsilon_s) \prod_{\ell=2}^N \mathbb P(\mathcal X_{\ell} \mid \mathcal X_{\ell -1})}} = \sqrt[N]{\frac{1}{\prod_{\ell=2}^N \mathbb P(\mathcal X_{\ell} \mid \mathcal X_{\ell -1})}}.$$ للحظة، دعنا نفترض أن $$\mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell -1}) = c$$ ثابت لجميع ثنائيات الكلمات. تحت هذا الافتراض، تكون الحيرة هي $$\sqrt[N]{\frac{1}{c^{N-1}}}$$؛ ومع ذلك، إذا كانت $$N$$ لا تأخذ في الاعتبار رمز البداية الذي له احتمال $$1$$، فستكون الحيرة هي $$\sqrt[N-1]{\frac{1}{c^{N-1}}}=c$$ وهي أكثر قابلية للتفسير من المعادلة السابقة، وترتبط بعامل التفريع في اللغة. وبالتالي، لن يساهم رمز البداية في قيمة $$N$$ في حساب الحيرة.

تحسب الدالة التالية الحيرة بافتراض جملة أو قائمة جمل كمدخلات. يتم تحويل الجداء $$\prod \mathbb P(\mathcal X_{\ell} \mid \mathcal X_{\ell-1})$$ إلى مجموع باستخدام اللوغاريتم، وتستمر بقية العمليات على الفضاء اللوغاريتمي. الخطوة الأخيرة هي تغيير النتيجة باستخدام الأس.

```python
def PP(sentences,
       prob=lambda a, b: P[a][b]):
    if isinstance(sentences, str):
        sentences = [sentences]
    tot, N = 0, 0
    for sentence in sentences:
        words = sentence.split()
        words.insert(0, '<s>')
        words.append('</s>')
        for a, b in zip(words, words[1:]):
            tot += np.log(prob(a, b))
        N += (len(words) - 1)
    _ = - tot / N
    return np.exp(_)
```

على سبيل المثال، فإن الحيرة للجملة *I like to play football* هي:

```python
text = 'I like to play football'
PP(text)
70.01211090353188
```

والحيرة للمدونة النصية المستخدمة لتدريب LM هي:

```python
fname2 = join('dataset', 'tweets-2022-01-17.json.gz')
PP([x['text'] for x in tweet_iterator(fname2)])
66.8740729934466
```

مثال آخر يمكن أن يكون *I like to play soccer* والذي يتم حسابه كما يلي.

```python
PP('I like to play soccer')
```

ينتج عن هذا المثال خطأ قسمة على الصفر؛ المشكلة هي أن الكلمتين الثنائيتين *play soccer* لم تُشاهد في مجموعة التنسيق/التدريب. ومع ذلك، لا يزال المرء يرغب في حساب حيرة تلك الجملة، والأهم من ذلك، يجب أن ينمذج LM أي جملة حتى لو لم تتم مشاهدتها في مدونة التدريب.

# الكلمات خارج المفردات (Out of Vocabulary - OOV)

المشكلة الموضحة في المثال السابق تُعرف باسم **الكلمات خارج المفردات (Out of Vocabulary - OOV)**. كما نعلم، فإن معظم الكلمات غير متكررة، مما يتطلب تدريب النموذج على مدونة نصية ضخمة لجمع أكبر عدد ممكن من الكلمات؛ ومع ذلك، لن تكون هناك مجموعة بيانات كبيرة بما يكفي لجميع الحالات بالنظر إلى أن اللغة تتطور والقيود الفيزيائية لحساب LM مع مدونة نصية بهذا الحجم. وبالتالي، يجب التعامل مع مشكلة OOV بشكل مختلف.

تقليديًا، يكون النهج المتبع هو تقليل الكتلة الممنوحة لتلك الكلمات المسترجعة في مجموعة التدريب ثم استخدام تلك الكتلة في كلمات OOV. ويتم ذكر الكتلة لأن احتمال جميع الأحداث يجب أن يجمع إلى واحد، ففي العملية التي اتبعناها، يكون مجموع احتمالات جميع الكلمات واحدًا. لا يمكن أن يكون هذا المجموع واحدًا لأن هناك كلمات لم تُشاهد بعد.

## تنعيم لابلاس (Laplace Smoothing)

أحد المناهج هو زيادة تكرار جميع الكلمات في مدونة التدريب بمقدار واحد. الفكرة هي تعريف دالة $$C^\star$$ كما يلي $$C^\star(\ldots, \mathcal X_{\ell-1}, \mathcal X_{\ell}) = C(\ldots, \mathcal X_{\ell-1}, \mathcal X_{\ell}) + 1$$، وبالنسبة لحالة ثنائيات الكلمات تتوافق مع $$C^\star(\mathcal X_{\ell-1}, \mathcal X_{\ell}) = C(\mathcal X_{\ell-1}, \mathcal X_{\ell}) + 1$$، حيث $$C^\star(\mathcal X_{\ell-1}) = \sum_i C^\star(\mathcal X_{\ell-1}, \mathcal X_i) = C(\mathcal X_{\ell-1}) + V$$، حيث $$V$$ هو حجم المفردات مع حساب الكلمة غير المعروفة. يمكن تنفيذ الطريقة بالكود التالي.

```python
V = set()
[[V.add(x) for x in key] for key in bigrams.keys()]
V = len(V) + 1

prev_l = dict()
for (a, b), v in bigrams.items():
    try:
        prev_l[a] += v
    except KeyError:
        prev_l[a] = v

P_l = defaultdict(Counter)
for (a, b), v in bigrams.items():
    next = P_l[a]
    next[b] = (v + 1) / (prev_l[a] + V)
```

يقارن الجدول التالي الكلمات الأربع الأكثر احتمالاً بوجود رمز البداية باستخدام النهج الذي لا يتعامل مع OOV وباستخدام تنعيم لابلاس.

|الكلمة (Word)|خط الأساس (Baseline)|تنعيم لابلاس (Laplace) |
|----|--------|--------|
|I   |0.028640|0.004450|
|The |0.020600|0.003201|
|This|0.009020|0.001403|
|A   |0.006780|0.001056|

يمكن ملاحظة من الجدول أن الاحتمال باستخدام طريقة لابلاس قد ينخفض لنفس الكلمة الثنائية؛ ومن ناحية أخرى، فإن الكتلة المقابلة للكلمات غير المعروفة المعطاة لرمز البداية هي: $$1 - \sum \mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell - 1}=\epsilon_s) \approx 0.7541.$$

يمكن حساب الحيرة باستخدام الطريقة الموصوفة سابقًا؛ ومع ذلك، فإن تنعيم لابلاس يغير كيفية حساب الاحتمال الشرطي. تستخدم طريقة الحيرة دالة مساعدة وهي الاحتمال الشرطي في الحالة القياسية. في تنعيم لابلاس، تضع الدالة المساعدة في الاعتبار الحالة التي تكون فيها الكلمة التالية غير معروفة، وحتى عندما تكون كلا الكلمتين غير معروفين، يمكن ملاحظة ذلك في الكود التالي.

```python
def laplace(a, b):
    if a in P_l:
        next = P_l[a]
        if b in next:
            return next[b]
    if a in prev_l:
        return 1 / (prev_l[a] + V)
    return 1 / V
```

إن الحيرة للجملة *I like to play football* أعلى من تلك المحسوبة سابقًا. ومن ناحية أخرى، فإن الحيرة لـ *I like to play soccer* هي $$5342.2$$.

```python
PP('I like to play football', prob=laplace)
2954.067962071032
```

يجب قياس حيرة نموذج اللغة على مدونة نصية لم تُشاهد من قبل؛ على سبيل المثال، قيمتها للتغريدات المجمعة في 10 يناير 2022 هي:

```python
fname2 = join('dataset', 'tweets-2022-01-10.json.gz')
PP([x['text'] for x in tweet_iterator(fname2)],
    prob=laplace)
49563.71966143271
```

إن الحيرة لمدونة التنسيق المستخدمة لتقدير المعلمات هي $$30646.76$$، وهي أقل من الحيرة المقاسة على مجموعة الاختبار، وهو السلوك المتوقع.

# الأنشطة والتمارين (Activities)

يمكن تعديل تنعيم لابلاس لتغيير كتلة التخزين للرموز غير المشاهدة.

## تنعيم إضافة k (Add-k Smoothing)

الفكرة هي استبدال $$1$$ بثابت $$k$$ في $$C^\star$$؛ مع هذا التعديل يتم تعريف $$C^\star$$ كـ: $$C^\star(\ldots, \mathcal X_{\ell-1}, \mathcal X_{\ell}) = C(\ldots, \mathcal X_{\ell-1}, \mathcal X_{\ell}) + k$$. بالنسبة لحالة ثنائيات الكلمات، يتوافق تكرار الكلمات مع $$C^\star(\mathcal X_{\ell-1}) = \sum_i C^\star(\mathcal X_{\ell-1}, \mathcal X_i) = C(\mathcal X_{\ell-1}) + kV.$$

يكون الاحتمال الشرطي لثنائيات الكلمات هو:

$$\mathbb P(\mathcal X_\ell, \mid \mathcal X_{\ell -1}) =  \frac{C^\star (\mathcal X_{\ell -1}, \mathcal X_\ell)}{\sum_i C^\star (\mathcal X_{\ell - 1}, \mathcal X_i)} = \frac{C(\mathcal X_{\ell -1}, \mathcal X_\ell) + k}{C(\mathcal X_{\ell - 1}) + kV}, $$

والذي يمكن تنفيذه كما يلي؛ حيث تتوافق `ngrams` مع $$C$$ و `prev` هي $$C(\mathcal X_{\ell - 1}).$$

```python
def cond_prob(ngrams, prev):
    output = defaultdict(Counter)
    for (*a, b), v in ngrams.items():
        key = tuple(a)
        next = output[key]
        next[b] = (v + K) / (prev[key] + K * V)
    return output
```

تظهر أدناه طريقة حساب $$C(\mathcal X_{\ell} - 1)$$ من $$C(\mathcal X_{\ell-1}, \mathcal X_i)$$ حيث تتوافق `data` مع التكرار الأخير.

```python
def sum_last(data):
    output = Counter()
    for (*prev, last), v in data.items():
        key = tuple(prev)
        output.update({key: v})
    return output
```

يؤثر الثابت $$k$$ أيضًا على الإجراء المستخدم لحساب الاحتمال الشرطي؛ الفرق هو التبادل بين واحد و $$k$$، كما هو ملاحظ في الكود التالي.

```python
K = 1 
def laplace(a, b):
    if a in P_l:
        next = P_l[a]
        if b in next:
            return next[b]
    if a in prev_l:
        return K / (prev_l[a] + K * len(V))
    return K / (len(V) + K * len(V))
```

يمكن حساب الحيرة للجملة *I like to play soccer* بالكود التالي، والفرق هو أن المعلمة $$k$$ تعين على $$0.1$$.

```python
prev_l = sum_last(bigrams)
K = 0.1
P_l = cond_prob(bigrams, prev_l)
PP('I like to play soccer', 
   prob=lambda a, b: laplace((a, ), b))
1780.7548164607583  
```

## التنعيم الأقصى (Max Smoothing)

تم اقتراح تقنيات مختلفة للتعامل مع الكلمات المفقودة في LM؛ هناك طريقة مشابهة لـ $$k$$ smoothing وهي تعريف $$C^\star$$ باستخدام الحد الأقصى للتكرار أو المعلمة $$k$$، أي $$C^\star(\ldots, \mathcal X_{\ell-1}, \mathcal X_{\ell}) = \max(C(\ldots, \mathcal X_{\ell-1}, \mathcal X_{\ell}), k).$$

الاحتمال الشرطي لثنائيات الكلمات هو:

$$\mathbb P(\mathcal X_\ell, \mid \mathcal X_{\ell -1}) = \frac{\max(C(\mathcal X_{\ell -1}, \mathcal X_\ell), k)}{C(\mathcal X_{\ell - 1}) + k \sum_i \mathbb 1(C(\mathcal X_{\ell-1}, \mathcal X_i)=0)}, $$

حيث يمكن تنفيذ المقام كـ:

```python
def sum_last_max(data):
    tokens = Counter()
    output = Counter()
    for (*prev, last), v in data.items():
        key = tuple(prev)
        output.update({key: v})
        tokens.update({key: 1})
    for key, v in tokens.items():
        output.update({key: K * (V - v)})
    return output
```

ويتوافق الاحتمال الشرطي مع الكود التالي.

```python
def cond_prob_max(ngrams, prev):
    output = defaultdict(Counter)
    for (*a, b), v in ngrams.items():
        key = tuple(a)
        next = output[key]
        next[b] = v / prev[key]
    return output 
```

تتوافق الدالة المساعدة لطريقة الحيرة مع الدالة التالية.

```python
def prob_max(a, b):
    if a in P_l:
        next = P_l[a]
        if b in next:
            return next[b]
    if a in prev_l:
        return K / prev_l[a]
    return 1 / V
```

تم الحصول على حيرة الجملة *I like to play soccer* بالكود التالي؛ ويمكن ملاحظة أن قيمة الحيرة تتشابه باستخدام طريقة تنعيم $$k$$ وهذه الطريقة الأخيرة.

```python
K = 0.1
prev_l = sum_last_max(bigrams)
P_l = cond_prob_max(bigrams, prev_l)
PP('I like to play soccer', 
   prob=lambda a, b: prob_max((a, ), b))
1762.903955247848  
```

من منظور آخر، المعلمة $$k$$ في كلا الطريقتين تعدل كمية الكتلة المخزنة للأحداث غير المشاهدة، في حالة $$k=0.1$$ تكون الكتلة المخزنة للأحداث غير المشاهدة هي $$1 - \sum \mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell - 1}=\epsilon_s) \approx 0.3269$$ وهي أقل بكثير من تلك التي تم الحصول عليها عندما تكون $$k=1$$ وهو تنعيم لابلاس القياسي.

يمكن تغيير المعلمة $$k$$ لقيم مختلفة لتوضيح سلوك الحيرة لعوامل التنعيم المختلفة. تنقّل الصورة التالية الحيرة لقيم مختلفة من $$k$$ عندما تكون $$k$$ في النطاق من $$0.01$$ إلى $$1$$. القيم المعروضة هي الحيرة التي تم الحصول عليها في مجموعة التدريب لطريقة تنعيم لابلاس والتنعيم الأقصى والحيرة في مجموعة الاختبار لكلا الطريقتين.


![Max Smoothing](/NLP-Course-Ar/assets/images/laplace_max_smoothing.png)

## النماذج متعددة الكلمات (N-Gram)

كما هو متوقع، فإن إنشاء LM باستخدام ثنائيات الكلمات فقط ليس كافياً لنمذجة تعقيد اللغة؛ ومع ذلك، فإن تمديد هذا النموذج أمر مباشر عن طريق زيادة عدد الكلمات المأخوذة في الاعتبار. يمكن أن يكون النموذج عبارة عن trigram LM أو 4-gram وهكذا. ومع ذلك، في كل مرة يتم فيها زيادة عدد الكلمات، يقل عدد الأمثلة لتقدير الاحتمال المشترك، وحتى زيادة حجم مجموعة التدريب لا تكفي. لذلك، تغيرت LMs إلى تمثيل مستمر بدلاً من التمثيل المنفصل؛ وسيتم تغطية هذا الموضوع لاحقًا في المساق.

ينمذج trigram LM قيمة $$\mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell - 2}, \mathcal X_{\ell -1})$$؛ والخطوة الأولى هي تقدير هذه القيم من مدونة نصية. الإجراء يعادل ثنائيات الكلمات حيث الفرق الوحيد هو أنه يجب إضافة رمز بداية آخر. يمكن ملاحظة أنه يتم استخدام رموز بداية $$n-1$$ وأن n-grams يتم حسابها باستخدام الدالة `zip` وتدوين تجميع القوائم.

```python
def compute_ngrams(fname, n=3):
    ngrams = Counter()
    for text in tweet_iterator(fname):
        text = text['text']
        words = text.split()
        [words.insert(0, '<s>') for _ in range(n - 1)]
        words.append('</s>')
        _ = [a for a in zip(*(words[i:] for i in range(n)))]
        ngrams.update(_)
    return ngrams
```

تطلب الدالة المساعدة لـ Perplexity التحديث لحساب n-grams؛ الإجراء مشابه للإجراء المستخدم لإنشاء n-grams.

```python
def PP(sentences,
       prob=lambda a, b: P_l[a][b], n=3):
    if isinstance(sentences, str):
        sentences = [sentences]
    tot, N = 0, 0
    for sentence in sentences:
        words = sentence.split()
        [words.insert(0, '<s>') for _ in range(n-1)]
        words.append('</s>')
        tot = 0
        for *a, b in zip(*(words[i:] for i in range(n))):
            tot += np.log(1 / prob(tuple(a), b))
        N += (len(words) - (n - 1))
    _ = tot / (len(words) - (n - 1))
    return np.exp(_)
```

عند هذه النقطة، لدينا جميع العناصر لإنشاء LM بأي حجم؛ على سبيل المثال، ينشئ الكود التالي trigram LM باستخدام التنعيم الأقصى.

```python
fname = join('dataset', 'tweets-2022-01-17.json.gz')
ngrams = compute_ngrams(fname, n=3)
V = set()
_ = [[V.add(x) for x in key] for key in ngrams.keys()]
V = len(V) - 1
K = 0.1
prev_l = sum_last_max(ngrams)
P_l = cond_prob_max(ngrams, prev_l)
```

يمكن توضيح استخدام LM بإنشاء سحابة كلمات للاحتمال الشرطي
$$\mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell-2}=of, \mathcal X_{\ell-1}=the)$$ (أي `P_l[('of', 'the')]`) الموضح أدناه.

![Conditional on *of the*](/NLP-Course-Ar/assets/images/wordcloud_prob_of_the.png)