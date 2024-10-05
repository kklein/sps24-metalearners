---
theme: base
paginate: true
header: "![w:50px](imgs/quantco.svg)"
footer: Francesc Martí Escofet [@fmartiescofet](https://twitter.com/fmartiescofet), Kevin Klein [@kevkle](https://twitter.com/kevkle)
---

<!-- _color: "black" -->
<!-- _footer: ''-->
<!-- _header: ''-->

<!-- _paginate: skip -->

# Learning From Experiments With Causal Machine Learning

## A case study using ``metalearners``

Francesc Martí Escofet (@fmartiescofet)
Kevin Klein (@kevkle)

---

TODO

---

<!-- _footer: ''-->
<!-- _header: ''-->

![bg left 80%](imgs/qr-metalearners.svg)

## Please leave feedback on GitHub! :)

[github.com/QuantCo/metalearners](https://github.com/QuantCo/metalearners)

[github.com/kklein/pdp24-metalearners](https://github.com/kklein/pdp24-metalearners)

---

<!-- _footer: ''-->
<!-- _header: ''-->

## Would you like to work on such topics, too?

Join us!
[quantco.com](https://www.quantco.com)

![bg left 70%](imgs/quantco_black.png)

---

<!-- _footer: ''-->
<!-- _header: ''-->

![bg 90%](imgs/jobs-dl-eng.png)
![bg 90%](imgs/jobs-ds.png)

---

# Backup

---

## Conventional assumptions for estimating CATEs

- Positivity/overlap
- Conditional ignorability/unconfoundedness
- Stable Unit Treatment Value (SUTVA)

A randomized control trial usually gives us the first two for free.

For more information see e.g. [Athey and Imbens, 2016](https://arxiv.org/pdf/1607.00698.pdf).

---

## Python implementations of MetaLearners

|                                           | `metalearners` | `causalml` | `econml` |
| ----------------------------------------- | :------------: | :--------: | :------: |
| MetaLearner implementations               |       ✔️       |     ✔️     |    ✔️    |
| Support\* for `pandas`, `scipy`, `polars` |       ✔️       |     ❌     |    ❌    |
| HPO integration                           |       ✔️       |     ❌     |    ❌    |
| Concurrency across base models            |       ✔️       |     ❌     |    ❌    |
| >2 treatment variants                     |       ✔️       |     ✔️     |    ❌    |
| Classification\*                          |       ✔️       |     ❌     |    ✔️    |
| Other Causal Inference methods            |       ❌       |     ✔️     |    ✔️    |
