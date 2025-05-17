from PIL import Image

img = Image.open("C:\\Users\galip\Downloads\\120kmh.png")
img_resized = img.resize((32, 32))
img_resized.save("your_image.png")