
def convolution(a, b):
    b = b[::-1]
    c = []
    for i in range(len(a) + len(b) - 1):
        conv = 0
        if i < len(a):
            for j in range(len(b) - len(a) + i + 1):
                conv += a[i] * b[j]
        c.append(conv)
    return c

a = [1, 2, 3, 4, 5]
b = [6, 7, 8, 9, 10]
result = convolution(a, b)
print(result)
