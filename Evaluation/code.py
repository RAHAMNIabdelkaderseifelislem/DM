"""
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


actual = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]
predicted = [1, 1, 0, 1, 0, 0, 2, 2, 1, 2, 3, 3]
accuracy = accuracy(actual, predicted)
print('Accuracy: %.2f%%' % accuracy)
"""
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

actual = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]

predicted1 = [1, 1, 0, 1, 0, 0, 2, 2, 1, 2, 3, 3]
accuracy1 = accuracy(actual, predicted1)

predicted2 = [1, 1, 0, 1, 0, 0, 2, 2, 1, 3, 3, 3]
accuracy2 = accuracy(actual, predicted2)

predicted3 = [1, 1, 0, 1, 0, 0, 2, 2, 1, 3, 2, 2]
accuracy3 = accuracy(actual, predicted3)

print('Accuracy1: %.2f%%' % accuracy1)
print('Accuracy2: %.2f%%' % accuracy2)
print('Accuracy3: %.2f%%' % accuracy3)

import matplotlib.pyplot as plt

# plot a bar chart labeled with the accuracy values and the algorithm names
plt.bar([1, 2, 3], [accuracy1, accuracy2, accuracy3], tick_label=['Algorithm 1', 'Algorithm 2', 'Algorithm 3'])
plt.show()
