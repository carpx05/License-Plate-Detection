import cv2
import matplotlib.pyplot as plt

def display(img, title=''):
    # Your existing code for displaying images goes here.
    # Create a function, such as `display`, and place the code in it.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10,6))
    ax = plt.subplot(111)
    ax.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()

img = cv2.imread('E:/LPR/NumberPlate.jpg')
display(img, 'input image')
