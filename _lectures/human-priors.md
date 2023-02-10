---
layout: lecture
title: "Encoding Human Priors: Data Augmentation and Prompt Engineering"
date: 2023-01-26
ready: true
panopto: "https://mit.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=45670839-428a-449a-bdff-af85012d2c30"
video:
  aspect: 56.25
  id: z44vZ_9av-M
slides:
  - /lectures/files/lec8.pdf
  - /lectures/files/lec8.pptx
---

This lecture covers encoding human priors into machine learning models through data! Two popular ways are data augmentation for training data, and at test-time with large language models (LLMs) using prompt engineering.


This lecture is structured as follows:

1. [ML models fail in simple ways](#ml-models-fail-in-simple-ways): encoding human priors, overfitting/underfitting
2. [Human priors to augment training data](#human-priors-to-augment-training-data): data augmentation, types of augmentation
3. [Human priors at test-time (LLMs)](#human-priors-at-test-time-llms): prompt engineering, context matters, correcting input data


# ML Models Fail in Simple Ways

First, let’s start with why this all matters. ML models, if you played with them outside a perfectly manicured ML class environment, will fail in sometimes seemingly simple ways. They seem not just simple, but often silly or stupid to us at first glance. For example, a model that is trained on pictures of dogs that are upright, won’t recognize a sideways rotated dog!

![Model misclassifying a rotated dog](/lectures/files/human-priors/lec8.004.jpeg)

That seems ridiculous. But that’s because we know something that the model doesn’t. We know that dogs, while often upright, are sometimes rotated too. We didn’t gather the data because we didn’t think that would be a problem. But we took that for granted. That, in essence, is an example of a human prior.

**Human priors.** They are prior knowledge we have about the world, about the data, about the task. And we often take them for granted, like the rotated dog.

In the case of the rotated dog, it’s a special type of human prior that is particularly useful. That’s an invariance. Basically a change to the input data that doesn’t change its output. This is useful because we can find smart ways to encode them in the input training data without needing to gather more data.

**Encoding** — what does that mean? It just means finding a function to represent the invariance. So for rotating the image, it’s just a function for rotation.

Specifically, we’re looking at adapting the data today. It’s a very effective place to be doing this and much easier than making architectural or loss function adaptations. It’s a common technique that we ML researchers and practitioners all do.

![Model correctly classifying a rotated dog](/lectures/files/human-priors/lec8.009.jpeg)

So with rotated dogs, we get a model that can detect dogs when rotated too. It’s not as simple as just rotating a few dogs. You have to literally do a few, not all of them, and it’s more of an empirical task of figuring out how much of your data you should flip or not to get the desired result.

In more technical language, this touches on the problem of overfitting. When a model overfits, it overindexes on the features it has seen a lot of and starts to memorize patterns as opposed to learning and generalizing. This is also relevant to underfitting as underfitting often means a lack of data in the first place, so adding more data when collecting that data is hard can be very valuable.

![Overfitting vs underfitting](/lectures/files/human-priors/lec8.010.jpeg)

# Human Priors to Augment Training Data

Data augmentation comes from the observation that collecting enough data is often very difficult. Sometimes, you think you have a lot of data but you actually don't have enough coverage of certain classes. This is called class imbalance. Alternatively, you could have a lot of data, but since it's skewed, biased, or simulated, it doesn't actually generalize to your test set.

What data augmentation does is it enables you to add more data with the data you already have. This is an easy win to improve your model without needing to collect more data, when collecting more data is hard or expensive or time consuming. For example, if you are in a medical use case, you need labels from a doctor, that doctor might be extremely expensive.

As a concrete example, I had a case where we needed several board certified pathologists to be able to label our data and even then they didn't always agree with each other. Getting their time and accurate labels over and over again (since we couldn't necessarily get the labeling scheme right the first couple times) was really expensive and hard to coordinate, delaying the project for months if that wasn’t possible.

So what data augmentation can do is it enables you to encode your human priors over invariances that you know about in your data and you're able to augment your dataset further, such as using flip and rotation on those dog pictures that you saw previously.

Now, those are pretty simple. There are far more advanced methods, such as Mobius transformations ([Zhou et al., 2020](https://arxiv.org/abs/2002.02917)). If you have classes, you can also use an effective method called Mixup, where you can mix your different classes together to be used as interpolated examples in alpha space ([Zhang et al., 2017](https://arxiv.org/abs/1710.09412)). What does that mean? If you have dog pictures and cat pictures, you can overlay these images together (e.g. by varying the alpha or A parameter in RGBA). For example, you can change the alpha of a cat image to 60% and the dog image to 40%. You’d get a blended cat-dog, and as a human, you’d agree that there is a cat and dog in it. Then, you could actually change your class label to be 60% cat and 40% dog for your model to predict. You can vary this however you want across your data to produce more training examples with precise labels. This is actually a very effective technique and used pretty widely now.

![Mixup](/lectures/files/human-priors/lec8.015.jpeg)

Data augmentation can also be taken to the extreme of synthetic data augmentation. This means using the data you already have, you can even train a model to generate more of that kind or class of data. For this, you can train your own model or you can use [foundation models](https://arxiv.org/abs/2108.07258), such as [DALL-E](https://openai.com/dall-e-2/) or [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) in the image scenario, to generate more data from them. Just know that you have to think about how this impacts your test set, if the foundation model has been trained on samples in your test set.

Data augmentation can also be useful for robotics. Because it is so expensive to run experiments in the physical world, we often use simulation to train robotics algorithms. So, we can transfer styles from a simulated environment into the styles of a real environment using a generative model. In this case, this was work from Google on RetinaGAN ([Ho et al., 2020](https://arxiv.org/abs/2011.03148)).

![RetinaGAN](/lectures/files/human-priors/lec8.017.jpeg)

Data augmentation works across a lot of different modalities. It's not just on images. For text, there's a really interesting technique called back-translation, where you can take an English sentence, such as “I have no time”, and use a translation model into French (“je n’ai pas le temps”), then translate back into English for “I don’t have time.”  What's really interesting is now your translation back into English aren’t in the same words that you used before, but it has the same meaning. So you can use this new example as augmentation on your data set to help the model understand different ways of phrasing the same thing, and avoid overfitting on the original example. Of course, you can use this on any source and target languages.

# Human Priors at Test-Time (LLMs)

Now, that's encoding human priors into training data before you train your model. However, you can also encode human priors into your model at test time. One popular method is called prompt engineering. It is used for large [language models](https://en.wikipedia.org/wiki/Language_model) (LLMs). What this means is that you are changing the input at test time to elicit certain results at output time. For example, you can ask an LLM to write a letter of recommendation for a student. It'll write a letter of recommendation that's pretty average. But if you ask it to write a letter of recommendation for a student who gets into MIT, then it does much better because it assumes your letter will get into MIT. Interesting, right?

LLMs are special because they have an easy interface for humans to use, and that is language. Human language is something that we are all very comfortable using to prompt the model and to provide as input. This is somewhat taken for granted and is not actually a very known thing. In the past, research has really focused on understanding how to find those secret knobs inside of a model, called disentanglement. Entire PhDs have been completed around this method. but adding an LLM actually gives us an interface into the original models, such as an image model. The models themselves don't have to change, but adding the language model really does change how we interact with the model.

Prompt engineering really depends on the model. Different models have been trained to do different things. For example you can see here that GPT-3 ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)) has been trained to just predict the next thing and right here it is just assuming, for example, that you're in a form and you're writing different questions for a form. It's likely that it’s seen a lot of forms. Now, [GPT-3.5 (ChatGPT)](https://openai.com/blog/chatgpt/) acts very differently: it's able to take in commands, because it's been trained additionally on a lot of dialogue and commands data. So when asked a question, it actually answers it rather than propose a new question.

![GPT-3 vs GPT-3.5 (ChatGPT)](/lectures/files/human-priors/lec8.023.jpeg)

A very powerful method for adapting these models is giving them examples. Examples help nudge not only what you want but also provides context into what type of scenario the model should be operating under. For example, you can give GPT-3 some context that you are actually answering questions, as opposed to writing questions for a form. You can give examples of asking and answering questions to then be able to answer your questions now.

![Prompt engineering GPT-3](/lectures/files/human-priors/lec8.024.jpeg)

You’ll have the opportunity to explore this in the lab assignment for this lecture. There, you'll get to build a context template for scalability and reusability. You'll be able to add examples to boost the model’s performance. You'll get to just tweak the prompt to see how that changes the output.

This is really similar to how you use Google right now as you tweak the queries you give to Google to elicit results that you want. So, that's a wrap on how to encode human priors into models, starting with data augmentation — which is really manipulating the data space before we train the model — as well as prompt engineering — which is manipulating the data that goes into the model at test time. Taken together, data provides an extremely useful interface to encode human priors into your models to improve their performance.

# Lab

The lab assignments for the course are available in the [dcai-lab](https://github.com/dcai-course/dcai-lab) repository.

Remember to run a `git pull` before doing every lab, to make sure you have the latest version of the labs.

The lab assignment for this class is in [`prompt_engineering/Lab_Prompt_Engineering.ipynb`](https://github.com/dcai-course/dcai-lab/blob/master/prompt_engineering/Lab_Prompt_Engineering.ipynb) (also available on [Colab](https://colab.research.google.com/drive/1cipH-u6Jz0EH-6Cd9MPYgY4K0sJZwRJq)). This lab guides you through prompt engineering, crafting inputs for large language models (LLMs). With these large pre-trained models, even small amounts of data can make them very useful.
