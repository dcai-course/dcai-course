---
layout: page
title: "2023 Lectures"
permalink: /2023/
description: >
  Lecture videos for Introduction to Data-Centric AI, MIT IAP 2023.
thumbnail: /static/assets/thumbnail.png
phony: true
---

<ul class="double-spaced">
  {% assign lectures = site['2023'] | sort: 'date' %}
  {% for lecture in lectures %}
    {% if lecture.phony != true %}
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
    {% endif %}
  {% endfor %}
</ul>

{% comment %}
**Lecture**: [6-120](https://whereis.mit.edu/?go=6), 1pm--2pm<br>
**Office hours**: [2-132](https://whereis.mit.edu/?go=2), 3pm--5pm (every day, after lecture)
{% endcomment %}

Video recordings of the lectures are available <a href="https://www.youtube.com/watch?v=ayzOzZGHZy4&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5">on YouTube</a>.

If you have questions about the lecture material or the labs, please ask in
`#dcai-course` on this [Slack](https://cleanlab.ai/slack/).

# Beyond MIT

We've also shared this class beyond MIT in the hopes that others may
benefit from these resources. You can find posts and discussions on:

- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/comments/1194wm0/p_mit_introduction_to_datacentric_ai/)
- [r/learnmachinelearning](https://www.reddit.com/r/learnmachinelearning/comments/1194vsn/mit_introduction_to_datacentric_ai/)
- [Hacker News](https://news.ycombinator.com/item?id=34906593)
- [Lobsters](https://lobste.rs/s/qtaba8/mit_introduction_data_centric_ai)
- [LinkedIn](https://www.linkedin.com/pulse/teaching-first-data-centric-ai-course-mit-curtis-northcutt/)
- [Twitter](https://twitter.com/anishathalye/status/1628437244464992256)
