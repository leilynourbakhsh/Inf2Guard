from PIL import Image

# Create a list to store the image objects
images = []

# Loop through the image files and open them
for i in range(10):
    filename = f"{i}_atk.png"
    img = Image.open(filename)
    images.append(img)

# Determine the size of the final stitched image
max_width = max(img.width for img in images)
total_height = sum(img.height for img in images)

# Create a new image with the determined size
stitched_image = Image.new('RGB', (max_width * 2, int(total_height // 5 * 5/2)))  # Ensure height is divisible by 5

# Paste each image into the stitched image
x_offset, y_offset = 0, 0
for img in images:
    stitched_image.paste(img, (x_offset, y_offset))
    x_offset += img.width
    if x_offset >= max_width * 2:
        x_offset = 0
        y_offset += img.height

# Save the stitched image
stitched_image.save("atk.png")

# Close all the image objects
for img in images:
    img.close()

# Create a list to store the image objects
images = []

# Loop through the image files and open them
for i in range(10):
    filename = f"{i}_ori.png"
    img = Image.open(filename)
    images.append(img)

# Determine the size of the final stitched image
max_width = max(img.width for img in images)
total_height = sum(img.height for img in images)

# Create a new image with the determined size
stitched_image = Image.new('RGB', (max_width * 2, int(total_height // 5 * 5/2)))  # Ensure height is divisible by 5

# Paste each image into the stitched image
x_offset, y_offset = 0, 0
for img in images:
    stitched_image.paste(img, (x_offset, y_offset))
    x_offset += img.width
    if x_offset >= max_width * 2:
        x_offset = 0
        y_offset += img.height

# Save the stitched image
stitched_image.save("ori.png")

# Close all the image objects
for img in images:
    img.close()
