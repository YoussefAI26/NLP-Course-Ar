---
layout: default
title: معالجة اللغة الطبيعية
nav_order: 1
permalink: /
---

# معالجة اللغة الطبيعية (NLP)
{: .fs-10 }

ماريو غراف Mario Graff (mgraffg en ieee.org)
{: .fs-7 }


---

# المقدمة (Introduction)

في حياتنا اليومية، نستخدم تطبيقات تستند إلى تقنيات معالجة اللغة الطبيعية (Natural Language Processing - NLP). فمن الشائع استخدام المساعدات الافتراضية مثل Siri أو Alexa، أو الاتصال بمراكز الخدمة الهاتفية حيث تُحدد الخيارات شفهيًا، أو حتى استخدام خدمة الإملاء الصوتي في الهواتف المحمولة. وعند زيارة دولة أجنبية أو تعلم لغة أخرى، من الشائع استخدام خدمات أنظمة الترجمة الفورية عبر الإنترنت، مثل ترجمة جوجل (Google Translate). قد لا تكون تطبيقات معالجة اللغة الطبيعية الأخرى بنفس الدرجة من الوضوح؛ على سبيل المثال، يقترح تويتر متابعة المحادثات بناءً على الموضوع. ويساعد Grammarly في تحسين أسلوب الكتابة، وهو ما يمكن القيام به أيضًا عبر خيارات Word وGoogle Docs. من ناحية أخرى، تم تطوير تطبيقات لـ NLP لاستخلاص المعلومات والآراء المعبَّر عنها في وسائل التواصل الاجتماعي أو الويب.

يهدف هذا المساق التعليمي إلى تعريف الطلاب بمجال معالجة اللغة الطبيعية (NLP). وسيتم شرح أهم تطبيقات معالجة اللغة الطبيعية من خلال المحاضرات، القراءات، والأنشطة البرمجية؛ وتشمل هذه التطبيقات المتلازمات اللفظية (Collocations)، نماذج اللغة (Language Models)، تصنيف النصوص (Text Categorization)، تضمينات الكلمات (Word Embeddings)، الإجابة عن الأسئلة (Question Answering)، والاستلزام اللغوي (Sentence Entailment)، من بين مواضيع أخرى.

# الترميز والرموز (Notation)

|الرمز            | المعنى                                                  |
|------------------|----------------------------------------------------------|
|$$x$$             | متغير يُستخدم عادة كـ مدخل (Input)                          |
|$$y$$             | متغير يُستخدم عادة كـ مخرج (Output)                         |
|$$w$$             | متغير يُستخدم غالبًا للكلمات (Tokens)                  |
|$$\mathbb R$$     | مجموعة الأعداد الحقيقية (Real numbers)                                         |
|$$\mathbf x$$     | متجه عمودي (Column vector) $$\mathbf x \in \mathbb R^d$$              |
|$$d$$             | البُعد (Dimension)                                                |
|$$\mathbf w^\intercal \cdot \mathbf x$$ | الضرب النقطي (Dot product) حيث $$\mathbf w$$ و $$\mathbf x \in \mathbb R^d$$ |
|$$\mathcal D$$    | مجموعة البيانات (Dataset) للأزواج $$\{(x_i, y_i) \mid i=1, \dots N\}$$    |
|$$\mathcal T$$  | مجموعة التدريب (Training set)| 
|$$\mathcal V$$| مجموعة التحقق (Validation set) |
|$$\mathcal G$$| مجموعة الاختبار (Test / Gold set) |
|$$N$$             | عدد الأمثلة (Number of examples)                                       | 
|$$K$$             | عدد الفئات أو التسميات (Number of classes or labels)                              |
|$$\mathbb P(\cdot)$$  | التوزيع الاحتمالي (Probability distribution)                             |
|$$\mathcal X, \mathcal Y$$    | متغيرات عشوائية (Random variables)                             |
|$$\mathcal N(\mu, \sigma^2)$$    | التوزيع الطبيعي (Normal distribution) بالمعلمتين $$\mu$$ و $$\sigma^2$$|
|$$f_{\mathcal X}$$| دالة كثافة الاحتمال (pdf) للمتغير $$\mathcal X$$      |
|$$\mathbb 1(e)$$     | دالة المؤشر (Indicator function)؛ وتساوي $$1$$ فقط إذا كان $$e$$ صحيحًا          |
|$$\Omega$$        | فضاء البحث (Search space)                                             |
|$$\mathbb V$$     | التباين (Variance)                                                 |
|$$\mathbb E$$     | القيمة المتوقعة (Expectation)                                                 |

# المتطلبات (Requirements)

## مكتبات بايثون (Python's Libraries)

- [NumPy](https://numpy.org)
- [scikit-learn](https://scikit-learn.org/stable/index.html)
- [spacy](https://spacy.io)
- [$$\mu$$TC](https://microtc.readthedocs.io/en/latest/)
- [EvoMSA](https://evomsa.readthedocs.io/en/latest/)
- [text_models](https://text-models.readthedocs.io/en/latest/)

# المراجع والمصادر (Bibliography)

- Speech and Language Processing. An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition. Third Edition draft. Daniel Jurafsky and James H. Martin. [pdf](https://web.stanford.edu/~jurafsky/slp3/ed3book_sep212021.pdf)
- Introduction to machine learning, Third Edition. Ethem Alpaydin. MIT Press
- All of Statistics. A Concise Course in Statistical Inference. Larry Wasserman. MIT Press.
