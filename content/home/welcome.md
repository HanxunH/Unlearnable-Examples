---
# A Demo section created with the Blank widget.
# Any elements can be added in the body: https://wowchemy.com/docs/writing-markdown-latex/
# Add more sections by duplicating this file and customizing to your requirements.

widget: hero  # See https://wowchemy.com/docs/page-builder/
headless: true  # This file represents a page section.
weight: 10  # Order that this section will appear.
title: "Unlearnable Examples: Making Personal Data Unexploitable"
subtitle: ""
<!-- hero_media: welcome.jpg -->
design:
  # Choose how many columns the section has. Valid values: 1 or 2.
  columns: '1'
advanced:
  css_style:
  css_class:
---

---
### Team Member:
- [Hanxun Huang](http://hanxunh.github.io/) PhD Student <sup>1</sup>
- [Xingjun Ma](http://xingjunma.com/) Lecturer <sup>2</sup>
- [Sarah Erfani](https://people.eng.unimelb.edu.au/smonazam/) Senior Lecturer <sup>1</sup>
- [James Bailey](https://people.eng.unimelb.edu.au/baileyj/) Professor <sup>1</sup>
- [Yisen Wang](https://sites.google.com/site/csyisenwang/) Assistant Professor <sup>3</sup>
- <small> 1 The University of Melbourne </small>
- <small> 2 Deakin University </small>
- <small> 3 Peking University </small>
---
### Abstract
<small>
The volume of "free" data on the internet has been key to the current success of deep learning. However, it also raises privacy concerns about the unauthorized exploitation of personal data for training commercial models. It is thus crucial to develop methods to prevent unauthorized data exploitation. This paper raises the question: can data be made unlearnable for deep learning models? We present a type of error-minimizing noise that can indeed make training examples unlearnable. Error-minimizing noise is intentionally generated to reduce the error of one or more of the training example(s) close to zero, which can trick the model into believing there is "nothing" to learn from these example(s). The noise is restricted to be imperceptible to human eyes, and thus does not affect normal data utility. We empirically verify the effectiveness of error-minimizing noise in both sample-wise and class-wise forms. We also demonstrate its flexibility under extensive experimental settings and practicability in a case study of face recognition. Our work establishes an important ﬁrst step towards making personal data unexploitable to deep learning models.
</small>
