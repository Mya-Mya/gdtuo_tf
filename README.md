# GDTUO
This repository is based on paper, "Gradient Descent: The Ultimate Optimizer".

# Get Started
### Install & Import
```
pip install git+https://git@github.com/Mya-Mya/gdtuo_tf.git
```

```python
import gdtuo_tf
```

### Example
Create 2D Rosenbrock function. This optimal solution is $(1,1)$.
```python
def rosenbrock(x,y):
  return 100.*(y-x**2.)**2.+(1.-x)**2.
```

Build an `Adam/SGD` hyper optimizer.
```python
adamsgd = gdtuo_tf.Hyperoptimizer(
    gdtuo_tf.Adam(),
    gdtuo_tf.SGD(alpha=1e-4)
)
```

Starting from the initial variable $v_0=(0,-1), execute a loop of 100 epochs.$

```python
v = tf.Variable([0., -1.])
adamsgd_history = [v]

for epoch in range(100):
  fv = rosenbrock(v[0], v[1])
  v = adamsgd.step(fv, v)
  
  adamsgd_history.append(v)
adamsgd_history = np.array(adamsgd_history).T
```

### How to build a single optimizer?
`Dummy` optimizer does not optimize anything, be optimized anything. So, it is useful for when you want to run single optimizer.

For example, build a single `Adam` optimizer.
```python
adam = gdtuo_tf.Hyperoptimizer(
    gdtuo_tf.Adam(),
    gdtuo_tf.Dummy()
)
```

And executee a loop.
```python
v = tf.Variable([0., -1.])
adam_history = [v]

for epoch in range(num_epoch):
  fv = rosenbrock(v[0], v[1])
  v = adam.step(fv, v)

  adam_history.append(v)
adam_history = np.array(adam_history).T
```

### Results
Draw the function, and the trajectory of each training.
```python
X, Y = np.meshgrid(np.linspace(-1.5, 1.5), np.linspace(-1.5, 1.5))
Z = rosenbrock(X, Y)
plt.figure(figsize=(8,8), facecolor="white")
plt.contour(X, Y, Z, levels=[1, 10, 100, 1000], colors="gray")
plt.scatter(1, 1, marker="x", color="black")
plt.plot(adamsgd_history[0], adamsgd_history[1], label="Adam/SGD")
plt.plot(adam_history[0], adam_history[1], label="Adam")

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()
```

![results_example](https://raw.githubusercontent.com/Mya-Mya/gdtuo_tf/main/results-example.png)

# Reference
```
@inproceedings{
chandra2022gradient,
title={Gradient Descent: The Ultimate Optimizer},
author={Kartik Chandra and Audrey Xie and Jonathan Ragan-Kelley and Erik Meijer},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=-Qp-3L-5ZdI}
}
```
