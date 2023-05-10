---
layout: lecture
title: "Label Errors and Confident Learning"
description: >
  Learn how to algorithmically identify mislabeled data.
thumbnail: /lectures/label-errors/thumbnail.png
date: 2023-01-18
ready: true
panopto: "https://mit.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=1773586c-361c-40d9-b972-af85012d2b78"
video:
  aspect: 56.25
  id: AzU-G1Vww3c
slides:
  - /lectures/label-errors/label-errors.pdf
  - /lectures/label-errors/label-errors.pptx
---


If you've ever used datasets like [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html), [MNIST](https://web.archive.org/web/20230219004208/http://yann.lecun.com/exdb/mnist/), [ImageNet](https://www.image-net.org/), or [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/), you likely assumed the class labels are correct. Surprise: **there are 100,000+ label issues in ImageNet.** In this lecture, we introduce a principled and theoretically grounded framework called confident learning (open-sourced in the [cleanlab](https://github.com/cleanlab/cleanlab) package) that can be used to identify label issues/errors, characterize label noise, and learn with noisy labels automatically for most classification datasets.

{% comment %}
**There are label errors in ImageNet**. First, let's take a look at a few of the label issues confident learning finds in the standard 2012 ILSVRC ImageNet training set.
{% endcomment %}



![Label issues in ImageNet training set](imagenet_train_label_errors_32.jpg)

<p class="small center">Top 32 label issues in the 2012 ILSVRC ImageNet train set identified using confident learning. Label Errors are boxed in red. Ontological issues in green. Multi-label images in blue.</p>


The figure above shows examples of label errors in the 2012 ILSVRC ImageNet training set found using confident learning. For interpretability, we group label issues found in ImageNet using CL into three categories:
1. *Multi-label images* (<span style="color:teal">blue</span>) have more than one label in the image.
2. *Ontological issues* (<span style="color:lime">green</span>) comprise *is-a* (*bathtub* labeled *tub*) or *has-a* (*oscilloscope* labeled *CRT screen*) relationships. In these cases, the dataset should include one of the classes.
3. *Label errors* (<span style="color:red">red</span>) occur when a class exists in the dataset that is more appropriate for an example than its given class label.

{% comment %}
Accurately finding label errors in a dataset like ImageNet is tricky because there are 1000 classes, 1.28 million images, and typical accuracy is around 73%.
{% endcomment %}

{% comment %}
The creators of ImageNet, Russakovsky et al. (2015), suggest label errors exist due to human error, no attempt has been made to find them in the training set, characterize them, and re-train without them.
{% endcomment %}

Using confident learning, we can find label errors in any dataset using any appropriate model for that dataset. Here are three other real-world examples in common datasets.

![Three label errors from different datasets.](three_label_errors_example.png)

<p class="small center">Examples of label errors that currently exist in <a href="https://jmcauley.ucsd.edu/data/amazon/">Amazon Reviews</a>, <a href="https://web.archive.org/web/20230219004208/http://yann.lecun.com/exdb/mnist/">MNIST</a>, and <a href="https://github.com/googlecreativelab/quickdraw-dataset">Quickdraw</a> datasets identified using confident learning for varying data modalities and models.</p>



# What is Confident Learning?

Confident learning (CL) has emerged as a subfield within [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) and [weak-supervision](https://en.wikipedia.org/wiki/Weak_supervision) to:
* characterize label noise
* find label errors
* learn with noisy labels
* find ontological issues


CL is based on the principles of [**pruning** noisy data](https://arxiv.org/abs/1705.01936) (as opposed to [fixing label errors](https://en.wikipedia.org/wiki/Error_correction_code) or [modifying the loss function](https://papers.nips.cc/paper/5073-learning-with-noisy-labels)), [**counting** to estimate noise](https://www.semanticscholar.org/paper/Quantifying-Counts-%2C-Costs-%2C-and-Trends-Accurately-Forman/b00ce7c4d8a29071dd70ef9d944bdc73e53d4f78) (as opposed to [jointly learning noise rates](https://arxiv.org/abs/1406.2080) during training), and [**ranking** examples](https://en.wikipedia.org/wiki/Learning_to_rank) to train with confidence (as opposed to [weighting by exact probabilities](https://www.semanticscholar.org/paper/Training-deep-neural-networks-using-a-noise-layer-Goldberger-Ben-Reuven/bc550ee45f4194f86c52152c10d302965c3563ca)). Here, we generalize CL, building on the assumption of [Angluin and Laird's classification noise process](https://homepages.math.uic.edu/~lreyzin/papers/angluin88b.pdf) , to directly estimate the joint distribution between noisy (given) labels and uncorrupted (unknown) labels.



![CL Diagram](confident_learning_digram_final.jpg)

<p class="small center">The confident learning process and examples of the confident joint and estimated joint distribution between noisy (given) labels and uncorrupted (unknown) labels. \(\tilde{y}\) denotes an observed noisy label and \(y^*\) denotes a latent uncorrupted label.</p>


From the figure above, we see that CL requires two inputs:
* out-of-sample predicted probabilities (matrix size: # of examples by # of classes)
* noisy labels (vector length: number of examples)

For the purpose of [weak supervision](https://en.wikipedia.org/wiki/Weak_supervision), CL consists of three steps:
1. Estimate the joint distribution of given, noisy labels and latent (unknown) uncorrupted labels to fully characterize class-conditional label noise.
2. Find and prune noisy examples with label issues.
3. Train with errors removed, re-weighting examples by the estimated latent prior.

## Benefits of Confident Learning

Unlike most machine learning approaches, confident learning requires no hyperparameters. We use cross-validation to obtain predicted probabilities out-of-sample. Confident learning features a number of other benefits. CL

* directly estimates the joint distribution of noisy and true labels
* works for multi-class datasets
* finds the label errors (errors are ordered from most likely to least likely)
* is non-iterative (finding training label errors in ImageNet takes 3 minutes)
* is theoretically justified (realistic conditions exactly find label errors and consistent estimation of the joint distribution)
* does not assume randomly uniform label noise (often unrealistic in practice)
* only requires predicted probabilities and noisy labels (any model can be used)
* does not require any true (guaranteed uncorrupted) labels
* extends naturally to multi-label datasets
* is free and open-sourced as the [`cleanlab` Python package](https://github.com/cgnorthcutt/cleanlab) for characterizing, finding, and learning with label errors.

{% comment %}
A summary of these features compared with recent common benchmarks in noisy labels is shown in the table below.

{% include image-caption.html imageurl="table.png"
title="related_works" caption="Comparison of commonly benchmarked approaches for learning with noisy labels. Comparison is limited to model-agnostic approaches that do not require a subset of true labels." %}
{% endcomment %}






## The Principles of Confident Learning

CL builds on principles developed across the literature dealing with noisy labels:

* **Prune** to search for label errors, e.g. following the example of [Natarajan et al. (2013)](https://papers.nips.cc/paper/5073-learning-with-noisy-labels.pdf); [van Rooyen et al. (2015)](https://arxiv.org/abs/1505.07634); [Patrini et al. (2017)](https://arxiv.org/abs/1609.03683), using soft-pruning via loss-reweighting, to avoid the convergence pitfalls of iterative re-labeling.
* **Count** to train on clean data, avoiding error-propagation in learned model weights from reweighting the loss [(Natarajan et al., 2017)](https://www.jmlr.org/papers/volume18/15-226/15-226.pdf) with imperfect predicted probabilities, generalizing seminal work [Forman (2005, 2008)](https://dl.acm.org/citation.cfm?id=1403849); [Lipton et al. (2018)](https://arxiv.org/abs/1802.03916).
* **Rank** which examples to use during training, to allow learning with unnormalized probabilities or SVM decision boundary distances, building on well-known robustness findings of [PageRank (Page et al., 1997)](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf) and ideas of curriculum learning in [MentorNet (Jiang et al.,2018)](https://arxiv.org/abs/1712.05055).


# Theoretical Findings in Confident Learning

For full coverage of CL algorithms, theory, and proofs, please read [our paper](https://arxiv.org/abs/1911.00068). Here, I summarize the main ideas.

Theoretically, we show realistic conditions where CL (Theorem 2: *General Per-Example Robustness*) **exactly finds label errors** and **consistently estimates the joint distribution of noisy and true labels**. Our conditions allow for error in predicted probabilities for every example and every class.

# How does Confident Learning Work?

To understand how CL works, let's imagine we have a dataset with images of dogs, foxes, and cows. CL works by estimating the joint distribution of noisy and true labels (the Q matrix on the right in the figure below).

![Example matrices](example_matrices.png)
<p class="small center">Left: Example of confident counting examples. This is an unnormalized estimate of the joint. Right: Example joint distribution of noisy and true labels for a dataset with three classes.</p>


Continuing with our example, CL counts 100 images labeled *dog* with high probability of belonging to class *dog*, shown by the C matrix in the left of the figure above. CL also counts 56 images labeled *fox* with high probability of belonging to class *dog* and 32 images labeled *cow* with high probability of belonging to class *dog*.

For the mathematically curious, this counting process takes the following form.

![Confident Joint Equation](cj.png)

For an in-depth explanation of the notation, check out [the CL paper](https://arxiv.org/abs/1911.00068). The central idea is that when the predicted probability of an example is greater than a per-class-threshold, we *confidently count* that example as actually belonging to that threshold's class. The thresholds for each class are the average predicted probability of examples in that class. This form of thresholding generalizes [well-known robustness results in PU Learning (Elkan & Noto, 2008)](https://cseweb.ucsd.edu/~elkan/posonly.pdf) to multi-class weak supervision.

## Find label issues using the joint distribution of label noise

From the matrix on the right in the figure above, to estimate label issues:
1. Multiply the joint distribution matrix by the number of examples. Let's assume 100 examples in our dataset. So, by the figure above (*Q* matrix on the right), there are 10 images labeled *dog* that are actually images of *foxes*.
2. Mark the 10 images labeled dog with *largest* probability of belonging to class *fox* as label issues.
3. Repeat for all non-diagonal entries in the matrix.

Note: this simplifies the methods used in [our paper](https://arxiv.org/abs/1911.00068), but captures the essence.


# Practical Applications of Confident Learning




### CL Improves State-of-the-Art in Learning with Noisy Labels by over 10% on average and by over 30% in high noise and high sparsity regimes

![Confident Joint Equation](benchmarks.png)

The table above shows a comparison of CL versus recent state-of-the-art approaches for multiclass learning with noisy labels on CIFAR-10. At high sparsity (see next paragraph) and 40% and 70% label noise, CL outperforms [Google's](https://ai.google/research/pubs/pub47110/) top-performing [MentorNet](https://github.com/google/mentornet), [Co-Teaching](https://github.com/bhanML/Co-teaching), and [Facebook Research's](https://research.fb.com/downloads/mixup-cifar10/) [Mix-up](https://github.com/facebookresearch/mixup-cifar10) by over 30%. Prior to confident learning, improvements on this benchmark were significantly smaller (on the order of a few percentage points).

*Sparsity* (the fraction of zeros in *Q*) encapsulates the notion that real-world datasets like ImageNet have classes that are unlikely to be mislabeled as other classes, e.g. p(tiger,oscilloscope) ~ 0 in *Q*. Shown by the highlighted cells in the table above, CL exhibits significantly increased robustness to sparsity compared to state-of-the-art methods like [Mixup](https://github.com/facebookresearch/mixup-cifar10), [MentorNet](https://github.com/google/mentornet), [SCE-loss](https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels), and [Co-Teaching](https://github.com/bhanML/Co-teaching). This robustness comes from directly modeling *Q*, the joint distribution of noisy and true labels.

{% comment %}
{% include image-caption.html imageurl="benchmarks.png"
title="related_works" caption=" Comparison of CL versus prior art for multiclass learning with noisy labels in CIFAR-10. At 40% and 70% label noise, Cl outperforms the top-performing MentorNet by 30%." %}
{% endcomment %}

### Training on ImageNet cleaned with CL Improves ResNet Test Accuracy

![Improve ResNet training](imagenet.png)

In the figure above, each point on the line for each method, from left to right, depicts the accuracy of training with 20%, 40%..., 100% of estimated label errors removed. The black dotted line depicts accuracy when training with all examples. Observe increased ResNet validation accuracy using CL to train on a cleaned ImageNet train set (no synthetic noise added) when less than 100k training examples are removed. When over 100k training examples are removed, observe the relative improvement using CL versus random removal, shown by the red dash-dotted line.




{% comment %}
{% include image-caption.html imageurl="imagenet.png"
title="related_works" caption=" Increased ResNet validation accuracy using CL methods on ImageNet with original labels (no synthetic noise added). Each point on the line for each method, from left to right, depicts the accuracy of training with 20%, 40%..., 100% of estimated label errors removed. The red dash-dotted baseline captures when examples are removed uniformly randomly. The black dotted line depicts accuracy when training with all examples." %}
{% endcomment %}


### Good Characterization of Label Noise in CIFAR with Added Label Noise

![Accurate joint estimation](joint_estimation.png)

The figure above shows CL estimation of the joint distribution of label noise for CIFAR with 40% added label noise. Observe how close the CL estimate in (b) is to the true distribution in (a) and the low error of the absolute difference of every entry in the matrix in (c). Probabilities are scaled up by 100.

{% comment %}
{% include image-caption.html imageurl="joint_estimation.png"
title="related_works" caption="CL estimation of the joint distribution of label noise for CIFAR with 40% label noise and 60% sparsity. *Sparsity* is a proxy for the magnitude of non-uniformity of the label noise. Observe the similarity of values between (a) and (b) and the low error of the absolute difference of every entry in the matrix in (c). Probabilities are scaled up by 100." %}
{% endcomment %}


### Automatic Discovery of Onotological (Class-Naming) Issues in ImageNet

![Ontological issues](ontology.png)

CL automatically discovers ontological issues of classes in a dataset by estimating the joint distribution of label noise directly. In the table above, we show the largest off diagonals in our estimate of the joint distribution of label noise for ImageNet, a single-class dataset. Each row lists the noisy label, true label, image id, counts, and joint probability. Because these are off-diagonals, the noisy class and true class must be different, but in row 7, we see ImageNet actually has two **different** classes that are both called *maillot*. Also observe the existence of misnomers: *projectile* and *missile* in row 1, *is-a* relationships: *bathtub* is a *tub* in row 2, and issues caused by words with multiple definitions: *corn* and *ear* in row 9.



{% comment %}
{% include image-caption.html imageurl="ontology.png"
title="related_works" caption="ImageNet is supposed to only contain single-class images. Using CL, we automatically discover ontological issues with the classes in ImageNet. For example, the class maillot appears twice, the existence of is-a relationships like bathtub is a tub, misnomers like projectile and missile, and unanticipated issues caused by words with multiple definitions like corn and ear." %}
{% endcomment %}


# Final Thoughts

Our theoretical and experimental results emphasize the practical nature of confident learning, e.g. identifying numerous label issues in ImageNet and CIFAR and improving standard ResNet performance by training on a cleaned dataset. Confident learning motivates the need for further understanding of uncertainty estimation in dataset labels, methods to clean training and test sets, and approaches to identify ontological and label issues in datasets.


# Resources
* This lecture overviews the paper (JAIR 2021) [Confident Learning: Estimating Uncertainty in Dataset Labels](https://arxiv.org/abs/1911.00068)
* This lecture also covers the label errors paper (NeurIPS 2021): [Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks](https://arxiv.org/abs/2103.14749)
* Results of this method on the ten most commonly cited test sets in ML: [https://labelerrors.com](https://labelerrors.com)
* Try these methods yourself (open-sourced via cleanlab): [https://github.com/cleanlab/cleanlab](https://github.com/cleanlab/cleanlab).
* The [cleanlab](https://github.com/cleanlab/cleanlab) package is a data-centric AI package for improving ML models by improving datasets and supports things like training ML models and deep learning models with noisy labels, outliers, and data labeled by multiple annotators. Learn more in the [`cleanlab` documentation](https://docs.cleanlab.ai). [`cleanlab`](https://github.com/cgnorthcutt/cleanlab/tree/79035913dad2cc178eda5417f4a8bf03e011ddaf) + the [confidentlearning-reproduce](https://github.com/cgnorthcutt/confidentlearning-reproduce) repo reproduces results in the [CL paper](https://arxiv.org/abs/1911.00068).



# Lab

The lab assignments for the course are available in the [dcai-lab](https://github.com/dcai-course/dcai-lab) repository.

Remember to run a `git pull` before doing every lab, to make sure you have the latest version of the labs.

The lab assignment for this class is in [`label_errors/Lab - Label Errors.ipynb`](https://github.com/dcai-course/dcai-lab/blob/master/label_errors/Lab%20-%20Label%20Errors.ipynb). This lab guides you through writing your own implementation of automatic label error identification using Confident Learning, the technique taught in today's lecture.
