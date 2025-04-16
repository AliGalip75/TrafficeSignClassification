from PIL import Image

img = Image.open("your_image")
img_resized = img.resize((32, 32))
img_resized.save("your_image.png")