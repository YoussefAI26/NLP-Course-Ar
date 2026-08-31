---
layout: default
title: توحيد النصوص
nav_order: 5
---

# توحيد النصوص
{: .fs-10 .no_toc }

## المحتويات
{: .no_toc .text-delta }

1. TOC
{:toc}

## المكتبات المستعملة
{: .no_toc .text-delta }
```python
import numpy as np
from wordcloud import WordCloud as WC
from matplotlib import pylab as plt
from b4msa.textmodel import TextModel
from microtc.params import OPTION_GROUP, OPTION_DELETE, OPTION_NONE
from b4msa.lang_dependency import LangDependency
from nltk.stem.porter import PorterStemmer
import re
from microtc.textmodel import SKIP_SYMBOLS
import unicodedata
```

## تثبيت المكتبات الخارجية
{: .no_toc .text-delta }

```bash
pip install b4msa
pip install nltk
```

---

# المقدمة

في جميع المواضيع التي تم تغطيتها، يفترض الخوارزمية أن النص منسق جيدًا وأن المسافات تفصل بين الكلمات (الرموز Tokens) بشكل صحيح. ومع ذلك، ليست هذه هي الحالة العامة، وتؤثر الأخطاء الإملائية والإجراءات المستخدمة لتعريف الرموز بشكل قوي على أداء الخوارزمية. وبالتالي، يُخصص هذا الجزء من المساق لتقديم التقنيات القياسية المستخدمة لتوحيد النصوص (Text Normalization) وتحويل النص إلى رموز.

تعد معالجة توحيد النصوص الموصوفة هنا هي الرئيسية المستخدمة في الأبحاث التالية:

1. [An automated text categorization framework based on hyperparameter optimization](https://www.sciencedirect.com/science/article/pii/S0950705118301217)
2. [A simple approach to multilingual polarity classification in Twitter](https://www.sciencedirect.com/science/article/abs/pii/S0167865517301721)
3. [A case study of Spanish text transformations for twitter sentiment analysis](https://www.sciencedirect.com/science/article/abs/pii/S0957417417302312)

# الكيانات (Entities)

تبدأ رحلة توحيد النصوص بالتعامل مع الكيانات المختلفة داخل النص؛ قد تكون الكيانات إشارة إلى مستخدم في تغريدة، أو أرقام، أو روابط URL، على سبيل المثال لا الحصر. الإجراءات المتخذة على الكيانات الموجودة هي حذفها أو استبدالها برمز خاص.

## أسماء المستخدمين (Users)

تتمثل العملية الأولى في التعامل مع اسم المستخدم باتباع تنسيق تويتر. في التغريدة، يتم التعرف على الإشارة إلى المستخدم بسلسلة نصية تبدأ بالمحرف @. يمكن أن يكون الإجراءان هما حذف جميع الإشارات للمستخدمين أو تغييرها إلى وسم موحد.

يستخدم الإجراء التعبيرات النمطية (Regular Expressions) للعثور على الكيانات؛ على سبيل المثال، يمكن للكود التالي إزالة الإشارات للمستخدمين.

```python
text = 'Hi @xx, @mm is talking about you.'
re.sub(r"@\S+", "", text)
'Hi   is talking about you.'
```

من ناحية أخرى، يمكن تنفيذ استبدال اسم المستخدم بوسم مشترك بالكود التالي، حيث الوسم هو `_usr`

```python
text = 'Hi @xx, @mm is talking about you.'
re.sub(r"@\S+", "_usr", text)
'Hi _usr _usr is talking about you.'
```

## الروابط (URL)

يمكن تكييف الكود السابق للتعامل مع روابط URL؛ يحتاج المرء فقط إلى تعريف التعبير النمطي المراد استخدامه؛ انظر الكود التالي الذي يزيل جميع ظواهر URL.

```python
text = "go http://google.com, and find out"
re.sub(r"https?://\S+", "", text)
'go  and find out'
```

## الأرقام (Numbers)

يمكن تعديل الكود السابق للتعامل مع الأرقام واستبدال الرقم الموجود بوسم مشترك مثل `_num`.

```python
text = "we have won 10 M"
re.sub(r"\d\d*\.?\d*|\d*\.\d\d*", "_num", text)
'we have won _num M'
```

# الإملاء والهجاء (Spelling)

تعدل الكتلة التالية من توحيد النصوص كتابة النص، حيث تزيل المكونات التي يمكن لبعض التطبيقات تجاهلها لتقليل حجم المفردات، مما يؤثر على تعقيد الخوارزمية ويمكن أن ينعكس في تحسين الأداء.

## حساسية حالة الأحرف (Case Sensitive)

أول هذه التحويلات هو التحويل إلى أحرف صغيرة (Lower Case)؛ تحويل جميع الكلمات إلى أحرف صغيرة له نتيجة تقليل المفردات، على سبيل المثال، الكلمتان Mexico و mexico تعتبران نفس الرمز. يمكن تنفيذ هذه العملية باستخدام الدالة `lower` كما يلي.

```python
text = "Mexico"
text.lower()
'mexico'
```

## علامات الترقيم (Punctuation)

رموز الترقيم أساسية لفهم اللغة الطبيعية وتوليدها؛ ومع ذلك، بالنسبة للتطبيقات الأخرى، مثل تحليل المشاعر (Sentiment Analysis) أو تصنيف النصوص، فإن المساهمة تكون غير واضحة بسبب الزيادة في حجم المفردات. وبالتالي، فإن إزالتها تؤثر على حجم المفردات، مما ينتج عنه أحيانًا نتيجة إيجابية في الأداء.

يمكن إزالة هذه الرموز بالمرور عبر السلسلة النصية وتخطي علامات الترقيم.

```python
text = "Hi! good morning,"
output = ""
for x in text:
    if x in SKIP_SYMBOLS:
        continue
    output += x
output
'Hi good morning'
```

## التشكيل والحركات (Diacritics)

تستخدم اللغات المختلفة رموز تشكيل/حركات، على سبيل المثال، México؛ وكما هو متوقع، فإن هذا يؤدي إلى زيادة المفردات. من ناحية أخرى، في الكتابة غير الرسمية، يكثر الاستخدام الخاطئ لرموز التشكيل؛ وإحدى الطرق المحددة للتعامل مع هذه المشكلة هي إزالة رموز التشكيل ومعاملتها بنفس الكلمة، على سبيل المثال، سيتم استبدال México بـ Mexico.

```python
text = 'México'
output = ""
for x in unicodedata.normalize('NFD', text):
    o = ord(x)
    if 0x300 <= o and o <= 0x036F:
        continue
    output += x
output
'Mexico'
```

# التوحيد الدلالي (Semantic Normalizations)

تهدف المجموعة التالية من تقنيات التوحيد إلى تقليل حجم المفردات باستخدام معنى الكلمات لتعديلها أو إزالتها من النص.

## كلمات التوقف (Stop words)

كلمات التوقف هي الكلمات الأكثر تكرارًا في اللغة. هذه الكلمات أساسية للتواصل ولكنها ليست ذات أهمية كبيرة في المهام التي تهدف إلى التمييز بين النصوص وفقًا لمعناها.

يمكن تخزين كلمات التوقف في قاموس، ثم تتكون عملية إزالتها من المرور على جميع الرموز من النص وإزالة تلك الموجودة في القاموس. وتتم العملية مع إعطاء مثال بالكود التالي.

```python
lang = LangDependency('english')

text = 'Good morning! Today, we have a warm weather.'
output = []
for word in text.split():
    if word.lower() in lang.stopwords[len(word)]:
        continue
    output.append(word)
output = " ".join(output) 
output
'Good morning! Today, warm weather.'
```

## التجذير والترديد إلى الأصل (Stemming and Lemmatization)

فكرة التجذير (Stemming) والترديد إلى الأصل (Lemmatization)، كعملية توحيد، هي تجميع كلمات مختلفة بناءً على جذرها؛ على سبيل المثال، تربط العملية كلمات مثل *playing* و *player* و *plays* بالرمز *play*.

يعالج التجذير المشكلة بفيود أقل من الترديد/الرد إلى الأصل، مما يتسبب في أن الكلمة الشائعة المكتشفة قد لا تكون الجذر اللغوي للكلمات؛ بالإضافة إلى ذلك، لا تأخذ الخوارزميات في الاعتبار دور الكلمة التي تتم معالجتها في الجملة. من ناحية أخرى، تحصل خوارزمية الترديد/الرد إلى الأصل على جذر الكلمة مع مراعاة جزء الكلام (Part of Speech) للكلمة المعالجة.

```python
stemmer = PorterStemmer()

text = 'I like playing football'
output = []
for word in text.split():
    w = stemmer.stem(word)
    output.append(w)
output = " ".join(output) 
output
'i like play footbal'
```

# التقطيع والتحليل اللفظي (Tokenization)

بمجرد توحيد النص، حان الوقت لتحويله إلى عناصره الأساسية، والتي يمكن أن تكون كلمات، أو ثنائيات كلمات، أو n-grams، أو سلاسل فرعية، أو مدمجًا بينها؛ وتُعرف هذه العملية باسم التقطيع (Tokenization). يمكن تطبيق طرق مختلفة لتقطيع النص، والطريقة المستخدمة حتى الآن هي تحويل النص إلى قائمة كلمات حيث تفصل الكلمات مسافة أو محارف غير قابلة للطباعة. يعتمد القرار بشأن المحلل اللفظي الذي يجب استخدامه على التطبيق؛ على سبيل المثال، من أجل توليد النص، من المهم تعلم رموز الترقيم، وبالتالي فإن هذه الرموز تعتبر رموزًا مستقلة. من ناحية أخرى، في مسألة تصنيف النصوص، حيث تكون المهمة هي تصنيف النص، قد يكون من غير المهم الحفاظ على ترتيب الكلمات.

## متواليات الكلمات (n-grams)

تتوافق مراجعة المحلل اللفظي الأول مع تحويل النص إلى كلمات وثنائيات كلمات وبشكل عام n-grams. تكون حالة الكلمات مباشرة باستخدام الدالة `split`؛ وبمجرد الحصول على الكلمات، يمكن دمجها لتشكيل n-gram بأي حجم، كما هو موضح أدناه.

```python
text = 'I like playing football on Saturday'
words = text.split()
n = 3
n_grams = []
for a in zip(*[words[i:] for i in range(n)]):
    n_grams.append("~".join(a))
n_grams
['I~like~playing', 'like~playing~football',
 'playing~football~on', 'football~on~Saturday']
```

## متواليات المحارف (q-grams)

يكمل المحلل اللفظي q-gram ميزات n-grams؛ ويُعرف بالسلسلة النصية الفرعية ذات الطول $$q$$. تتمتع q-grams بميزتين هامتين؛ الأولى هي أنها مستقلة عن اللغة وبالتالي يمكن تطبيقها على أي لغة، والثانية هي أنها تعالج مشكلة الأخطاء الإملائية من منظور المطابقة التقريبية.

الكود يعادل الكود المستخدم لحساب n-grams، مع وجود الفارق في أن التكرار يكون على الأحرف بدلاً من الكلمات.

```python
text = 'I like playing'
q = 4
q_grams = []
for a in zip(*[text[i:] for i in range(q)]):
    q_grams.append("".join(a))
q_grams
['I li', ' lik',  'like', 'ike ', 'ke p', 'e pl',
 ' pla', 'play', 'layi', 'ayin', 'ying']
```

# نمذجة النصوص (TextModel)

تحتوي الفئة `TextModel` في مكتبة [B4MSA](https://b4msa.readthedocs.io/en/latest/) على توحيد النصوص والمحللات اللفظية الموصوفة ويمكن استخدامها كما يلي.

الخطوة الأولى هي إنشاء مثيل من الفئة بالنظر إلى المعلمات المطلوبة. تحتوي معلمات [الكيانات](#entity) على ثلاثة خيارات للحذف (`OPTION_DELETE`) للكيان، أو الاستبدال (`OPTION_GROUP`) برمز محدد سابقًا، أو عدم تطبيق هذه العملية (`OPTION_NONE`). هذه المعلمات هي:

* usr_option
* url_option
* num_option

تحتوي الفئة على ثلاثة تحويلات إضافية وهي:

* emo_option
* hashtag_option
* ent_option

يمكن تشغيل تحويلات [الإملاء والهجاء](#spelling) بالكلمات المفتاحية التالية:

* lc 
* del_punc
* del_diac

والتي تتوافق مع حالة الأحرف الصغيرة، وعلامات الترقيم، والتشكيل.

يتم إعداد التوحيد [الدلالي](#semantic-normalizations) بالمعلمات:

* stopwords
* stemming

أخيرًا، يتم تكوين المحلل اللفظي بـ المعلمة `token_list` التي تأخذ التنسيق التالي؛ تشير الأرقام السالبة إلى $$n$$-grams والأرقام الموجبة إلى $$q$$-grams.

على سبيل المثال، يدعو الكود التالي خوارزمية توحيد النص؛ الفارق الوحيد هو استبدال المسافات بـ `~`.

```python
text = 'I like playing football with @mgraffg'
tm = TextModel(token_list=[-1, 3], lang='english', 
               usr_option=OPTION_GROUP,
               stemming=True)
tm.text_transformations(text)
'~i~like~play~fotbal~with~_usr~'
```

من ناحية أخرى، يُستخدم المحلل اللفظي كما يلي.

```python
text = 'I like playing football with @mgraffg'
tm = TextModel(token_list=[-1, 5], lang='english', 
               usr_option=OPTION_GROUP,
               stemming=True)
tm.tokenize(text)               
['i', 'like', 'play', 'fotbal', 'with', '_usr',
 'q:~i~li', 'q:i~lik', 'q:~like', 'q:like~', 'q:ike~p',
 'q:ke~pl', 'q:e~pla', 'q:~play', 'q:play~', 'q:lay~f',
 'q:ay~fo', 'q:y~fot', 'q:~fotb', 'q:fotba', 'q:otbal',
 'q:tbal~', 'q:bal~w', 'q:al~wi', 'q:l~wit', 'q:~with',
 'q:with~', 'q:ith~_', 'q:th~_u', 'q:h~_us', 'q:~_usr',
 'q:_usr~']
```

 يمكن ملاحظة أن جميع $$q$$-grams تبدأ بالبادئة *q:*.
