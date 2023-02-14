---
layout: lecture
title: "Data-Centric AI vs. Model-Centric AI"
date: 2023-01-17
ready: true
panopto: "https://mit.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=edd1be69-b3de-4302-ac2b-af85012d2b18"
video:
  aspect: 56.25
  id: ayzOzZGHZy4
slides:
  - /lectures/data-centric-model-centric/data-centric-model-centric.pdf
  - /lectures/data-centric-model-centric/data-centric-model-centric.pptx
---

When you learn Machine Learning in school, a dataset is given to you that is fairly clean & well-curated (e.g. dog/cat images), and your job is to produce the best model for this dataset. All techniques taught in most ML classes are centered around this aim, covering different types of: models (neural networks, decision trees, etc.), training techniques (regularization, optimizaton algorithms, loss functions, etc.), and model/hyperparameter selection (plus model ensembling). We call this paradigm **model-centric AI**.

When working on real-world ML, your company/users do not care what clever modeling tricks you know to produce accurate predictions on highly curated data. Contrary to the classroom, the **data are not fixed** in real-world applications! You are free to modify the dataset in order to get better modeling performance or even collect additional data as your budget allows. Real-world data tends to be highly messy and plagued with issues, such that improving the dataset is a prerequisite for producing an accurate model ("*garbage in, garbage out*"). Seasoned data scientists know it is more worthwhile to invest in exploring and fixing the data than tinkering with models, but this process can be cumbersome for large datasets. Improving data has been mostly done in an ad hoc manner via manual labor guided by human intuition/expertise.

In this class, you will instead learn how to *systematically engineer data* to build better AI systems. We call this paradigm **data-centric AI** [[G21](#G21)]. While manual *exploratory data analysis* is a key first step of understanding and improving any dataset, *data-centric AI* uses AI methods to more systematically diagnose and fix issues that commonly plague real-world datasets. Data-centric AI can take  one of two forms:

1. AI algorithms that understand data and use that information to improve models. *Curriculum learning* is an example of this, in which ML models are trained on 'easy data' first [[B09](#B09)].

2. AI algorithms that modify data to improve AI models. *Confident learning* is an example of this, in which ML models are trained on a filtered dataset where mislabeled data has been removed [[NJC21](#NJC21)].

In both examples above, determining which data is easy or mislabeled is estimated automatically via algorithms applied to the outputs of trained ML models.

# A practical recipe for supervised Machine Learning

To recap: **model-centric AI** is based on goal of *producing the best model for a given dataset*, whereas **data-centric AI** is based on the goal of *systematically & algorithmically producing the best dataset to feed a given ML model*. To deploy the best supervised learning systems in practice, one should do both. A *data-centric AI* pipeline can look like this:

1. Explore the data, fix fundamental issues, and transform it to be ML appropriate.

2. Train a baseline ML model on the properly formatted dataset.

3. Utilize this model to help you improve the dataset (*techniques taught in this class*).

4. Try different modeling techniques to improve the model on the improved dataset and obtain the best model.

Remember not to skip straight from Step 2 to Step 4 even though it is tempting! To deploy the best ML systems, one can iterate Steps 3 and 4 multiple times.

# Examples of data-centric AI

Methodologies that fall within the purview of this field include:
- Outlier detection and removal (handling abnormal examples in dataset)
- Error detection and correction (handling incorrect values/labels in dataset)
- Establishing consensus (determining truth from many crowdsourced annotations)
- Data augmentation (adding examples to data to encode prior knowledge)
- Feature engineering and selection (manipulating how data are represented)
- Active learning (selecting the most informative data to label next)
- Curriculum learning (ordering the examples in dataset from easiest to hardest)

Recent high-profile ML applications have clearly shown how the *reliability of ML models* deployed in the real-world depends on the *quality of their training data*.

OpenAI has openly stated that one of the biggest issues in training famous ML models like Dall-E, GPT-3, and ChatGPT were errors in the data and labels. Here are some stills from their [video demo of Dall-E 2](https://openai.com/dall-e-2/#demos):

![!Stills from Dalle-E 2 Video by OpenAI](/lectures/data-centric-model-centric/dalle.png)

Through relentless model-assisted dataset improvement (Step 3 above), Tesla could produce autonomous driving systems far more advanced than similar competitors. They point to their *Data Engine* as the key reason from this success, depicted in the following [slides from Andrej Karpathy](https://vimeo.com/274274744), the Tesla Director of AI (2021):

![Tesla Data Engine](/lectures/data-centric-model-centric/dataengine.png)

![Amount of sleep lost over data in PhD vs at Tesla](/lectures/data-centric-model-centric/teslasleep.png)

# Why we need data-centric AI

Bad data costs the U.S. alone around $3 Trillion every year [[R16](#R16)]. Data quality issues plague almost every industry and dealing with them manually imposes an immense burden. As datasets grow larger, it becomes infeasible to ensure their quality without the use of algorithms [[S22](#S22)]. Recent ML systems trained on massive datasets like [ChatGPT](https://openai.com/blog/chatgpt/) have relied on huge amounts of labor (human feedback) to try to overcome shortcomings arising from low-quality training data; however such efforts have been unable to fully overcome these shortcomings [[C23](#C23)]. Now more than ever, we need automated methods and systematic engineering principles to ensure ML models are being trained with clean  data. As ML becomes intertwined with our daily lives in healthcare, finance, transportation, it is imperative these systems are trained in a reliable manner.

Recent research is highlighting the value of data-centric AI for various applications. For image classification with noisily labeled data, a recent benchmark studied various methods to train models under increasing noise rates in the famous Cifar-10 dataset [[NJC21](#NJC21)]. The findings revealed that simple methods which adaptively change the dataset can lead to much more accurate models than methods which aim to account for the noise through sophisticated modeling strategies.

![Cifar10 image classification with noisy labels](/lectures/data-centric-model-centric/cifar10benchmarks.png)




Now that you've seen the importance of data-centric AI, this class will teach you techniques to improve *any* ML model by improving its data. The techniques taught here are applicable to most supervised ML models and training techniques, such that you will be able to continue applying them in the future even as better ML models and training techniques are invented. The better models of the future will better help you find and fix issues in datasets when combined with the techniques from this course, in order to help you produce even better versions of them!


# Lab

Lab assignments for each lecture in this course are available in the [dcai-lab](https://github.com/dcai-course/dcai-lab) GitHub repository. Get started with:

```console
$ git clone https://github.com/dcai-course/dcai-lab
```

Remember to run a `git pull` before doing every lab, to make sure you have the latest version of the labs.
Alternatively you can just download individual files yourself from [dcai-lab](https://github.com/dcai-course/dcai-lab).

The first lab assignment, in [`data_centric_model_centric/Lab - Data-Centric AI vs Model-Centric AI.ipynb`](https://github.com/dcai-course/dcai-lab/blob/master/data_centric_model_centric/Lab%20-%20Data-Centric%20AI%20vs%20Model-Centric%20AI.ipynb), walks you through an ML task of building a text classifier, and illustrates the power (and often simplicity) of data-centric approaches.


# References

<span id="G21"></span> [G21] Press, G. [Andrew Ng Launches A Campaign For Data-Centric AI](https://www.forbes.com/sites/gilpress/2021/06/16/andrew-ng-launches-a-campaign-for-data-centric-ai/?sh=664bf56374f5). *Forbes*, 2021.

<span id="B09"></span> [B09] Bengio, Y., et al. [Curriculum Learning](https://ronan.collobert.com/pub/2009_curriculum_icml.pdf). *ICML*, 2009.

<span id="NJC21"></span> [NJC21] Northcutt, C., Jiang, L., Chuang, I.L. [Confident Learning: Estimating Uncertainty in Dataset Labels](https://arxiv.org/abs/1911.00068). *Journal of Artifical Intelligence Research*, 2021.

<span id="R16"></span> [R16] Redman, T. [Bad Data Costs the U.S. $3 Trillion Per Year](https://hbr.org/2016/09/bad-data-costs-the-u-s-3-trillion-per-year
). *Harvard Business Review*, 2016.

<span id="S22"></span> [S22] Strickland, E. [Andrew Ng: Unbiggen AI](https://spectrum.ieee.org/andrew-ng-data-centric-ai). *IEEE Spectrum*, 2022.


<span id="C23"></span> [C23] Chiang, T. [ChatGPT is a Blurry JPEG of the Web](https://www.newyorker.com/tech/annals-of-technology/chatgpt-is-a-blurry-jpeg-of-the-web). *New Yorker*, 2023.

