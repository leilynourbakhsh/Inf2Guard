from PIL import Image
import math

# Create a list to store the image objects
images = []

# Loop through the image files and open them
for i in range(1, 25):
    filename = f"{i}_atk.png"
    img = Image.open(filename)
    images.append(img)

# Determine the size of the final stitched image
max_width = max(img.width for img in images)
max_height = max(img.height for img in images)
total_width = max_width * 3
total_height = max_height * 8

# Create a new image with the determined size
stitched_image = Image.new('RGB', (total_width, total_height))

# Paste each image into the stitched image
x_offset, y_offset = 0, 0
for img in images:
    stitched_image.paste(img, (x_offset, y_offset))
    x_offset += max_width
    if x_offset >= total_width:
        x_offset = 0
        y_offset += max_height

# Save the stitched image
stitched_image.save("atk.png")

# Close all the image objects
for img in images:
    img.close()

# Create a list to store the image objects
images = []

# Loop through the image files and open them
for i in range(1, 25):
    filename = f"{i}_ori.png"
    img = Image.open(filename)
    images.append(img)

# Determine the size of the final stitched image
max_width = max(img.width for img in images)
max_height = max(img.height for img in images)
total_width = max_width * 3
total_height = max_height * 8

# Create a new image with the determined size
stitched_image = Image.new('RGB', (total_width, total_height))

# Paste each image into the stitched image
x_offset, y_offset = 0, 0
for img in images:
    stitched_image.paste(img, (x_offset, y_offset))
    x_offset += max_width
    if x_offset >= total_width:
        x_offset = 0
        y_offset += max_height

# Save the stitched image
stitched_image.save("ori.png")

# Close all the image objects
for img in images:
    img.close()