---
layout: page
title: Introduction to Data-Centric AI
subtitle: IAP 2023
---

Typical machine learning classes teach techniques to produce effective models for a given dataset. In real-world applications, data is messy and improving models is not the only way to get better performance. You can also improve the dataset itself rather than treating it as fixed. **Data-Centric AI** (DCAI) is an emerging science that studies techniques to improve datasets, which is often the best way to improve performance in practical ML applications. While good data scientists have long practiced this manually via ad hoc trial/error and intuition, DCAI considers the improvement of data as a systematic engineering discipline.

This is the first-ever course on DCAI. This class covers algorithms to find and fix common issues in ML data and to construct better datasets, concentrating on data used in supervised learning tasks like classification. All material taught in this course is highly practical, focused on impactful aspects of real-world ML applications, rather than mathematical details of how particular models work.  You can take this course to learn practical techniques not covered in most ML classes, which will help mitigate the “garbage in, garbage out” problem that plagues many real-world ML applications.

# Syllabus

<ul>
{% assign lectures = site['lectures'] | sort: 'date' %}
{% for lecture in lectures %}
    <li>
    <strong>{{ lecture.date | date: '%-m/%d/%y' }}</strong>:
    {% if lecture.ready %}
        <a href="{{ lecture.url }}">{{ lecture.title }}</a>
    {% else %}
        {{ lecture.title }}
    {% endif %}
    </li>
{% endfor %}
</ul>

Video recordings of the lectures are available <a href="https://www.youtube.com/@dcai-course">on YouTube</a>.

Each lecture has an accompanying [lab
assignment](https://github.com/dcai-course/dcai-lab), hands-on programming
exercises in Python / Jupyter Notebook. You can work on these on your own, in
groups, and/or in office hours.

**Dates**: Tuesday, January 17 -- Friday, January 27, 2023<br>
**Lecture**: [35-225](http://whereis.mit.edu/?go=35), 1pm--2pm<br>
**Office hours**: [2-132](http://whereis.mit.edu/?go=2), 3pm--5pm (every day, after lecture)

# Prerequisites

Anyone is welcome to take this course, regardless of background. However, to
get the most out of this course, we recommend that you:

- Completed an introductory course in machine learning (like [6.036 / 6.390](https://introml.mit.edu/)). To learn this on your own, check out [Andrew Ng's ML course](https://www.coursera.org/learn/machine-learning), [fast.ai's ML course](https://course.fast.ai/), or [dive into deep learning](https://d2l.ai/).
- Are familiar with Python and its basic data science ecosystem (pandas, NumPy, scikit-learn, and Jupyter Notebook). To learn this on your own, check out [Jupyter Notebook 101](https://github.com/fastai/fastbook/blob/master/app_jupyter.ipynb), [Introduction to Pandas](https://walkwithfastai.com/Pandas), and [Python for Data Analysis](https://www.coursera.org/projects/python-for-data-analysis-numpy).
