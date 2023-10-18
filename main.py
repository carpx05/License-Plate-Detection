import cv2
import matplotlib.pyplot as plt
from src.plate_detection import detect_plate
from src.image_display import display
from src.character_segmentation import segment_characters
from Model.model import model

# Load the input image
img = cv2.imread('E:/LPR/NumberPlate.jpg')

# Display the input image
display(img, 'Input Image')

print("done...")
# Getting plate from the processed image
output_img, plate = detect_plate(img) #may be an error, check no of input parameters

# Display the detected license plate in the input image
display(output_img, 'Detected License Plate in the Input Image')

# Display the extracted license plate from the image
display(plate, 'Extracted License Plate from the Image')

# Segment characters
char = segment_characters(plate)

# Display segmented characters
for i in range(len(char)):
    plt.subplot(1, len(char), i + 1)
    plt.imshow(char[i], cmap='gray')
    plt.axis('off')

plt.show()

# Create and train the deep learning model
model = model()

# Predict the characters on the license plate
predicted_plate = model.predict(char)

# Display the predicted license plate characters
predicted_plate_str = ''.join(predicted_plate)
display(plate, 'Predicted License Plate: ' + predicted_plate_str)

# Save the final output image with the predicted license plate
cv2.imwrite('E:/LPR/Output/output_license_plate.jpg', plate)

print('Predicted License Plate:', predicted_plate_str)
