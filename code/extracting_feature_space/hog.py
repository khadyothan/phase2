import numpy as np

# Function to calculate Histogram of Oriented Gradients (HOG) feature descriptor for an image
def HOG(image):
    # Initialize an empty array for the HOG feature descriptor
    hog_feature_descriptor = np.zeros((10, 10, 9), dtype=np.float64)
    
    # Convert the input image to grayscale
    image_grayscale = image.convert("L")
    
    # Resize the grayscale image to a fixed size (300x100)
    image_data = np.array(image_grayscale.resize((300, 100)))
    
    # Define the number of rows, columns, and bins for the HOG grid
    rows, cols, bins = 10, 10, 9
    
    # Calculate the height and width of each grid cell
    grid_height = image_data.shape[0] // rows
    grid_width = image_data.shape[1] // cols
    
    # Iterate over each grid cell
    for row in range(rows):
        for col in range(cols):
            # Calculate the coordinates of the current grid cell
            start_x = col * grid_width
            end_x = (col+1) * grid_width
            start_y = row * grid_height
            end_y = (row+1) * grid_height
            
            # Extract the pixel values within the current grid cell
            grid_cell = image_data[start_y:end_y, start_x:end_x]
            
            # Calculate the gradient of the grid cell in both X and Y directions
            gradient_x = np.gradient(grid_cell, axis=1)
            gradient_y = np.gradient(grid_cell, axis=0)
            
            # Calculate the magnitude and direction of the gradient
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            gradient_direction = np.arctan2(gradient_y, gradient_x)
            
            # Compute the histogram of gradient directions for the grid cell
            hist, bin_edges = np.histogram(gradient_direction, bins=bins, range=(-np.pi, np.pi), weights=gradient_magnitude)
            
            # Store the computed histogram in the HOG feature descriptor
            hog_feature_descriptor[row, col, :] = hist
            
    # Return the computed HOG feature descriptor
    return hog_feature_descriptor.tolist()
