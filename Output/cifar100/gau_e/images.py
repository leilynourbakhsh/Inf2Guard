from PIL import Image

# Function to crop the center of an image to 32x32
def crop_center(image):
    width, height = image.size
    left = (width - 370) // 2
    top = (height - 370) // 2
    right = (width + 370) // 2
    bottom = (height + 370) // 2
    return image.crop((left, top, right, bottom))

# Create a list to store the cropped image objects
cropped_images = []

# Loop through the image files and crop the center
for i in range(10):
    filename = f"{i}_atk.png"
    img = Image.open(filename)
    cropped_img = crop_center(img)
    cropped_images.append(cropped_img)

# Determine the size of the final stitched image
max_width = max(cropped_img.width for cropped_img in cropped_images)
max_height = max(cropped_img.height for cropped_img in cropped_images)

# Define the gap size (3 pixels)
gap = 3

# Calculate the size of the stitched image
stitched_width = (max_width + gap) * 2
stitched_height = (max_height + gap) * 5

# Create a new image with a white background
stitched_image = Image.new('RGB', (stitched_width, stitched_height), (255, 255, 255))

# Paste each cropped image into the stitched image with a gap
x_offset, y_offset = 0, 0

for cropped_img in cropped_images:
    stitched_image.paste(cropped_img, (x_offset, y_offset))
    y_offset += max_height + gap

    if y_offset >= stitched_height:
        y_offset = 0
        x_offset += max_width + gap

# Save the stitched image
stitched_image.save("atk.png")

# Close all the image objects
for cropped_img in cropped_images:
    cropped_img.close()
