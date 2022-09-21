from PIL import Image
 
image = input("Enter the image file name: ")
im = Image.open(image)
print('Image resolution is (w, h): ' + str(im.size))