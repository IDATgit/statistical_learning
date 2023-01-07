import matplotlib.pyplot as plt


def d(p, n):
    return (1 - 0.5 ** (1/n)) ** (1 / p)




p_list = [3, 5, 10, 20, 50, 100]
n_list = [100, 5000, 100000]

for n in n_list:
    d_vals = [0]*len(p_list)
    for (p_idx, p) in enumerate(p_list):
        d_val = d(p, n)
        d_vals[p_idx] = d_val
    plt.plot(p_list, d_vals)

legend_strings = list(map(lambda x:f'n = {x}', n_list))
print(legend_strings)
plt.legend(legend_strings)
plt.title("expected median distance to origin nearest neighbor vs p (dimension size)")
plt.xlabel("p (ball dimension)")
plt.ylabel("d(p, n)")
plt.grid()
plt.show()
