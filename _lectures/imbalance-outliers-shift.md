---
layout: lecture
title: "Class Imbalance, Outliers, and Distribution Shift"
date: 2023-01-23
ready: true
video:
  aspect: 56.25
  # id: Wz50FvGG6xU
---

This lecture covers three common problems in real-world ML data: [class
imbalance](#class-imbalance), [outliers](#outliers), and [distribution shift](#distribution-shift).

# Class imbalance

{% include scaled_image.html alt="Class imbalance in credit card fraud" src="/lectures/files/imbalance-outliers-shift/fraud.svg" width="500" %}

Many real-world classification problems have the property that certain classes
are more prevalent than others. For example:

- COVID infection: among all patients, only [10% might have COVID](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge)
- Fraud detection: among all credit card transactions, fraud might [make up 0.2% of the transactions](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Manufacturing defect classification: different types of manufacturing defects might have [different prevalence](https://www.kaggle.com/c/severstal-steel-defect-detection)
- Self-driving car object detection: different types of objects have [different prevalence](https://github.com/udacity/self-driving-car/tree/master/annotations) (cars vs trucks vs pedestrians)

{% include lecnote.html content="Question: what is the difference between class imbalance and underperforming subpopulations, a topic covered in the previous lecture?" %}

## Evaluation metrics

If you're splitting a dataset into train/test splits, make sure to use [stratified data splitting](https://scikit-learn.org/stable/modules/cross_validation.html#stratification) to ensure that the train distribution matches the test distribution (otherwise, you're creating a [distribution shift](#distribution-shift)) problem.

With imbalanced data, standard metrics like accuracy might not make sense. For example, a classifier that always predicts "NOT FRAUD" would have 99.8% accuracy in detecting credit card fraud.

There is no one-size-fits-all solution for choosing an evaluation metric: the choice should depend on the problem. For example, an evaluation metric for credit card fraud detection might be a weighted average of the [precision and recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) scores (the [F-beta score](https://machinelearningmastery.com/fbeta-measure-for-machine-learning/)), with the weights determined by weighing the relative costs of failing to block a fraudulent transaction and incorrectly blocking a genuine transaction:

{% include lecnote.html content="precision = what proportion of positive identifications were actually positive = TP / (TP + FP). recall = what proportion of actual positives were identified correctly = TP / (TP + FN)." %}

\\[
F_\beta = \left(1 + \beta^2 \right) \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}{\beta^2 \cdot \mathrm{precision} + \mathrm{recall}}
\\]

{% include lecnote.html content="When beta=1, this turns into F1 score, the harmonic mean of precision and recall." %}

## Training models on imbalanced data

Once an evaluation metric has been chosen, you can try training a model in the standard way. If training a model on the true distribution works well, i.e., the model scores highly on the evaluation metric over a held-out test set that matches the real-world distribution, then you're done!

If not, there are techniques you can use to try to improve model performance on the minority classes.

**Sample weights.** Many models can be fit to a dataset with per-sample weights. Instead of optimizing an objective function that's a uniform average of per-datapoint losses, this optimizes a weighted average of losses, putting more emphasis on certain datapoints. While simple and conceptually appealing, this often does not work well in practice. For classifiers trained using mini-batches, using sample weights results in varying the effective learning rate between mini-batches, which can make learning unstable.

**Over-sampling.** Related to sample weights, you can simply replicate datapoints in the minority class, even multiple times, to make the dataset more balanced. In simpler settings (e.g., least-squares regression, this might be equivalent to sample weights), in other settings (e.g., training a neural network with mini-batch gradient descent), this is not equivalent and often performs better than sample weights. This solution is often unstable, and it can result in overfitting.

**Under-sampling.** Another way to balance a dataset is to _remove_ datapoints from the majority class. In some cases, it can work well, but in some situations, it can result in throwing away a lot of data when you have highly imbalanced datasets, resulting in poor performance.

**[SMOTE](https://arxiv.org/abs/1106.1813) (Synthetic Minority Oversampling TEchnique).** Rather than over-sampling by copying datapoints, you can use dataset augmentation to create new examples of minority classes by combining or perturbing minority examples. The SMOTE algorithm is sensible for certain data types, where interpolation in feature space makes sense, but doesn't make sense for certain other data types: averaging pixel values of one picture of a dog with another picture of a dog is unlikely to produce a picture of a dog. Depending on the application, other data augmentation methods could work better.

**[Balanced mini-batch training](https://ieeexplore.ieee.org/document/8665709).** For models trained with mini-batches, like neural networks, when assembling the random subset of data for each mini-batch, you can include datapoints from minority classes with higher probability, such that the mini-batch is balanced. This approach is similar to over-sampling, and it does not throw away data.

These techniques can be combined. For example, the SMOTE authors note that the combination of SMOTE and under-sampling performs better than plain under-sampling.

## References

- [imbalanced-learn Python package](https://imbalanced-learn.org)
- [SMOTE tutorial](https://towardsdatascience.com/smote-fdce2f605729)
- [Experimental perspectives on learning from imbalanced data (paper)](https://dl.acm.org/doi/10.1145/1273496.1273614)
- [Tour of evaluation metrics for imbalanced classification](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/)

# Outliers

{% include scaled_image.html alt="Outlier" src="/lectures/files/imbalance-outliers-shift/outlier.svg" width="350" %}

Outliers are datapoints that differ significantly from other datapoints. Causes include errors in measurement (e.g., a damaged air quality sensor), bad data collection (e.g., missing fields in a tabular dataset), malicious inputs (e.g., [adversarial examples](https://arxiv.org/abs/1312.6199)), and rare events (statistical outliers, e.g., an albino animal in an image classification dataset).

Outlier identification is of interest because outliers can cause issues during model training, at inference time, or when applying statistical techniques to a dataset. Outliers can harm model training, and certain machine learning models (e.g., vanilla SVM) can be particularly sensitive to outliers in the training set. A model, at deployment time, may not produce reasonable output if given outlier data as input (a form of [distribution shift](#distribution-shift)). If data has outliers, data analysis techniques might yield bad results.

{% include lecnote.html content="In 6.036, you learned some model-centric techniques to deal with outliers. For example, using L1 loss over L2 loss to be less sensitive to outliers. In this course, taking a data-centric view, we'll focus on identifying outliers." %}

Once found, what do you do with outliers? It depends. For example, if you find outliers in the training set, you don't want to blindly discard them: they might be rare events rather than invalid data points.

## Problem setup

Being a bit more formal with terminology, here are two tasks of interest:

**Outlier detection.** In this task, we are not given a clean dataset containing only in-distribution examples. Instead, we get a single un-labeled dataset, and the goal is to detect outliers in the dataset, datapoints that are unlike the others. This task comes up, for example, when cleaning a training dataset that is to be used for ML.

**Anomaly detection.** In this task, we are given an un-labled dataset of only in-distribution examples. Given a new datapoint _not_ in the dataset, the goal is to identify whether it belongs to the same distribution as the dataset. This task comes up, for example, when trying to identify whether a datapoint, at inference time, is drawn from the same distribution as a model's training set.

{% include lecnote.html content="Question: what makes anomaly different from a standard supervised learning classification problem (classify as anomaly or not)?" %}

## Identifying outliers

Outlier detection is a heavily studied field, with many algorithms and lots of published research. Here, we cover a couple selected techniques.

**[Tukey's fences](https://en.wikipedia.org/wiki/Outlier#Tukey's_fences).** A simple method for real-valued data. If $$Q_1$$ and $$Q_3$$ are the lower and upper quartiles, then this test says that any observation outside the following range is considered an outlier: $$[Q_1 - k(Q_1 - Q_3), Q_3 + k(Q_3 - Q_1)]$$. A multiplier of $$k=1.5$$ was proposed by John Tukey.

**Z-score.** For one-dimensional or low-dimensional data, Assuming a Gaussian distribution of data: calculate the Z-score as $$z_i = \frac{x_i - \mu}{\sigma}$$, where $$\mu$$ is the mean of all the data and $$\sigma$$ is the standard deviation. An outlier is a data point that has a high-magnitude Z-score, $$\| z_i \| > z_{thr}$$. A commonly used threshold is $$z_{thr} = 3$$. You can apply this technique to individual features as well.

**[Isolation forest](https://ieeexplore.ieee.org/document/4781136).** This technique is related to decision trees. Intuitively, the method creates a "random decision tree" and scores data points according to how many nodes are required to isolate them. The algorithm recursively divides (a subset of) a dataset by randomly selecting a feature and a split value until the subset has only one instance. The idea is that outlier data points will require fewer splits to become isolated.

**KNN distance.** In-distribution data is likely to be closer to its neighbors. You can use the mean distance (choosing an appropriate distance metric, like cosine distance) to a datapoint's k nearest neighbors as a score. For high-dimensional data like images, you can [use embeddings from a trained model](https://arxiv.org/abs/2207.03061) and do KNN in the embedding space.

{% include lecnote.html content="Suppose we want to use KNN for outlier detection. What's the setup? How does it change for anomaly detection?" %}

**Reconstruction-based methods.** [Autoencoders](https://en.wikipedia.org/wiki/Autoencoder) are generative models that are trained to compress high-dimensional data into a low-dimensional representation and then reconstruct the original data. If an autoencoder learns a data distribution, then it should be able to encode and then decode an in-distribution data point back into a data point that is close to the original input data. However, for out-of-distribution data, the reconstruction will be worse, so you can use reconstruction loss as a score for detecting outliers.

You'll notice that many outlier detection techniques involve computing a score for every datapoint and then thresholding to select outliers. Outlier detection methods can be evaluated by looking at the [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic), or if you want a single summary number to compare methods, looking at the AUROC.

## References

- [PyOD library](https://pyod.readthedocs.io/)
- [Tutorial: Outlier detection with autoencoders](https://towardsdatascience.com/outlier-detection-with-autoencoders-6c7ac3e2aa90)
- [Outlier detection in scikit-learn](https://scikit-learn.org/stable/modules/outlier_detection.html)

# Distribution shift

{% include scaled_image.html alt="Distribution shift" src="/lectures/files/imbalance-outliers-shift/distribution-shift.svg" width="450" %}

Distribution shift is a challenging problem that occurs when the joint distribution of inputs and outputs differs between training and test stages, i.e., $$p_\mathrm{train}(\mathbf{x}, y) \neq p_\mathrm{test}(\mathbf{x}, y)$$. This issue is present, to varying degrees, in nearly every practical ML application, in part because it is hard to perfectly reproduce testing conditions at training time.

## Types of distribution shift

### Covariate shift / data shift

Covariate shift occurs when $$p(\mathbf{x})$$ changes between train and test, but $$p(y \mid \mathbf{x})$$ does not. In other words, the distribution of inputs changes between train and test, but the relationship between inputs and outputs does not change.

{% include scaled_image.html alt="Covariate shift" src="/lectures/files/imbalance-outliers-shift/covariate-shift.svg" width="500" %}

Examples of covariate shift:

- Self-driving car trained on the sunny streets of San Francisco and deployed in the snowy streets of Boston
- Speech recognition model trained on native English speakers and then deployed for all English speakers
- Diabetes prediction model trained on hospital data from Boston and deployed in India

### Concept shift

Concept shift occurs when $$p(y \mid \mathbf{x})$$ changes between train and test, but $$p(\mathbf{x})$$ does not. In other words, the input distribution does not change, but the relationship between inputs and outputs does.

{% include scaled_image.html alt="Concept shift" src="/lectures/files/imbalance-outliers-shift/concept-shift.svg" width="500" %}

- Predicting a stock price based on company fundamentals, trained on data from 1975 and deployed in 2023
- Making product recommendations based on web browsing behavior, trained on pre-pandemic data and deployed in March 2020

### Prior probability shift / label shift

Prior probability shift appears only in $$y \rightarrow \mathbf{x}$$ problems (when we believe $$y$$ causes $$\mathbf{x}$$). It occurs when $$p(y)$$ changes between train and test, but $$p(\mathbf{x} \mid y)$$ does not. You can think of it as the converse of covariate shift.

To understand prior probability shift, consider the example of spam classification, where a commonly-used model is [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier). If the model is trained on a balanced dataset of 50% spam and 50% non-spam emails, and then it's deployed in a real-world setting where 90% of emails are spam, that is an example of prior probability shift.

Another example is when training a classifier to predict diagnoses given symptoms, as the relative prevalence of diseases is changing over time. Prior probability shift shift (rather than covariate shift) is the appropriate assumption to make here, because diseases cause symptoms.

## Detecting and addressing distribution shift

Some ways you can detect distribution shift in deployments:

- Monitor the performance of your model. Monitor accuracy, precision, statistical measures, or other evaluation metrics. If these change over time, it may be due to distribution shift.
- Monitor your data. You can detect data shift by comparing statistical properties of training data and data seen in a deployment.

At a high level, distribution shift can be addressed by fixing the data and re-training the model.

## References

- [Dataset Shift in Machine Learning (book)](https://direct.mit.edu/books/book/3841/Dataset-Shift-in-Machine-Learning)

# Lab

The lab assignment for this lecture is to implement and compare different methods for identifying outliers.

![Outlier meme](https://raw.githubusercontent.com/dcai-course/assets/master/outlier-meme.jpg)

For this lab, we've focused on anomaly detection. You are given a clean training dataset consisting of many pictures of dogs, and an evaluation dataset that contains outliers (non-dogs). Your task is to implement and compare various methods for detecting these outliers. You may implement some of the ideas presented in today's lecture, or you can look up other outlier detection algorithms in the linked references or online.

The lab assignments for the course are available in the [dcai-lab](https://github.com/dcai-course/dcai-lab) repository.

Remember to run a `git pull` before doing every lab, to make sure you have the latest version of the labs.

The lab assignment for this class is in [`outliers/Lab - Outliers.ipynb`](https://github.com/dcai-course/dcai-lab/blob/master/outliers/Lab%20-%20Outliers.ipynb).
