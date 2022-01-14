import matplotlib.pyplot as plt

a = (0, 0, 0.5, 1)
b = (0, 0, 1, 0.25)
arrow_a = plt.arrow(a[0], a[1], a[2], a[3])
arrow_b = plt.arrow(b[0], b[1], b[2], b[3])
result = plt.arrow(
    a[0] + b[0],
    a[1] + b[1],
    a[2] + b[2],
    a[3] + b[3],
    ec='green',
    head_width=0.02, )
plt.legend([result, arrow_a, arrow_b], ['result', 'a', 'b'])
plt.show()
