---
layout: page
title: "2024 Lectures"
permalink: /2024/
description: >
  Lecture videos for Introduction to Data-Centric AI, MIT IAP 2024.
thumbnail: /static/assets/thumbnail.png
phony: true
---

<ul class="double-spaced">
  {% assign lectures = site['2024'] | sort: 'date' %}
  {% for lecture in lectures %}
    {% if lecture.phony != true %}
      <li>
        <strong>{{ lecture.date | date: '%-m/%d/%y' }}</strong>:
        {% if lecture.ready %}
          <a href="{{ lecture.url }}">{{ lecture.title }}</a>
        {% elsif lecture.last_year %}
          {{ lecture.title }} (<a href="{{ lecture.last_year }}">see last year's version</a>)
        {% else %}
          {{ lecture.title }} (coming soon)
        {% endif %}
        {% if lecture.details %}
          <br>
          ({{ lecture.details }})
        {% endif %}
      </li>
    {% endif %}
  {% endfor %}
</ul>

<br>
**Lecture**: [2-190](https://whereis.mit.edu/?go=6), 12pm--1pm<br>
**Questions**: Post in the `#dcai-course` channel on [this community Slack team](https://cleanlab.ai/slack) (preferred) or email us at [dcai@mit.edu](mailto:dcai@mit.edu).

---

# Previous year's lectures

You can find lecture notes and videos from [last year's version of the class](/2023/). Each year's lectures are fully self-contained, and we recommend following the most recent version of the material (i.e. the 2024 lectures). There is slight variation in the topics covered, so we continue to host notes and videos for earlier versions of this course.
