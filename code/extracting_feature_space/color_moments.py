import numpy as np

# Function to calculate color moments feature descriptor for an image
def color_moments(image):
    # Convert the input image to a NumPy array and resize it to a fixed size (300x100)
    image_data = np.array(image.resize((300, 100)))
    
    # Initialize an empty array for the color moments feature descriptor
    color_moments_feature_descriptor = np.zeros((10, 10, 3, 3), dtype=np.float64)

    # Define the number of rows and columns for the grid
    rows, cols = 10, 10

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
            grid_cell = image_data[start_y:end_y, start_x:end_x,:]
            
            # Initialize lists to store channel values (R, G, B)
            r_values, g_values, b_values = [], [], []
            
            # Calculate the number of pixels in the grid cell
            N = grid_width * grid_height
            
            # Iterate over each pixel in the grid cell
            for i in grid_cell:
                for pixel in i:
                    r_values.append(pixel[0])  # Red channel
                    g_values.append(pixel[1])  # Green channel
                    b_values.append(pixel[2])  # Blue channel
            
            # Calculate mean, standard deviation, and skewness for each channel
            mean_r = np.mean(r_values)
            mean_g = np.mean(g_values)
            mean_b = np.mean(b_values)
            
            sd_r = np.sqrt(np.mean([(x - mean_r) ** 2 for x in r_values]))
            sd_g = np.sqrt(np.mean([(x - mean_g) ** 2 for x in g_values]))
            sd_b = np.sqrt(np.mean([(x - mean_b) ** 2 for x in b_values]))
            
            skew_r = np.cbrt(np.mean([(x - mean_r) ** 3 for x in r_values]))
            skew_g = np.cbrt(np.mean([(x - mean_g) ** 3 for x in g_values]))
            skew_b = np.cbrt(np.mean([(x - mean_b) ** 3 for x in b_values]))
            
            # Store the calculated moments in the feature descriptor array
            color_moments_feature_descriptor[row, col, 0, :] = [mean_r, sd_r, skew_r]
            color_moments_feature_descriptor[row, col, 1, :] = [mean_g, sd_g, skew_g]
            color_moments_feature_descriptor[row, col, 2, :] = [mean_b, sd_b, skew_b]
            
    # Return the computed color moments feature descriptor
    return color_moments_feature_descriptor.tolist()
