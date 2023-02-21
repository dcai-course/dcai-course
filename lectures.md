---
layout: page
title: "Lectures"
description: >
  Lecture videos for Introduction to Data-Centric AI, MIT IAP 2023.
thumbnail: /static/assets/thumbnail.png
---

<ul class="double-spaced">
  {% assign lectures = site['lectures'] | sort: 'date' %}
  {% for lecture in lectures %}
    <li>
      <strong>{{ lecture.date | date: '%-m/%d/%y' }}</strong>:
      {% if lecture.ready %}
        <a href="{{ lecture.url }}">{{ lecture.title }}</a>
      {% else %}
        {{ lecture.title }} [coming soon]
      {% endif %}
      {% if lecture.details %}
        <br>
        ({{ lecture.details }})
      {% endif %}
    </li>
  {% endfor %}
</ul>

{% comment %}
**Lecture**: [6-120](https://whereis.mit.edu/?go=6), 1pm--2pm<br>
**Office hours**: [2-132](https://whereis.mit.edu/?go=2), 3pm--5pm (every day, after lecture)
{% endcomment %}

Video recordings of the lectures are available <a href="https://www.youtube.com/watch?v=ayzOzZGHZy4&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5">on YouTube</a>.

If you have questions about the lecture material or the labs, please ask in
`#dcai-course` on this [Slack](https://cleanlab.ai/slack/).
