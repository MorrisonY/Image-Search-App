import os
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2

class ImageBrowser:
    # define constants (number of rows and columns for image display)
    ROW_COUNT = 4
    COLUMN_COUNT = 5

    def __init__(self, root, folder_path):
        self.root = root
        self.folder_path = folder_path
        self.images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        self.current_page = 0
        self.images_per_page = 20
        self.buttons = []
        self.histograms = self.intensity_search()
        self.color_codes = self.color_code_search()
        self.mode = 'intensity'
        self.weights = np.ones(len(self.images)) # initialize weights, no bias
        self.relevance = {image_name: 0 for image_name in self.images} # initialize relevance, 0 means non-relevant
        self.rel_vars = {}
        self.checkboxes = []  # Initialize the list for checkboxes

        self.setup_frames()
        self.display_images()
    
    def combined_search(self):
        combined_features = {}
        for image_name in self.images:
            intensity_hist = self.histograms[image_name]
            color_code_hist = self.color_codes[image_name]
            # Normalize the histograms by dividing by the image size
            intensity_hist = intensity_hist / intensity_hist.sum()
            color_code_hist = color_code_hist / color_code_hist.sum()
            # Concatenate the histograms to form the combined feature set
            combined_features[image_name] = np.concatenate((intensity_hist, color_code_hist))
        return combined_features
    
    def normalize_features(self, combined_features):
        # Convert combined features to a 2D numpy array for vectorized operations
        features_matrix = np.array(list(combined_features.values()))
        # Calculate the mean and standard deviation for each feature
        mean = np.mean(features_matrix, axis=0)
        std = np.std(features_matrix, axis=0)
        # Prevent division by zero in case there is a std of zero
        std[std == 0] = 1
        # Perform Gaussian normalization (z-score normalization)
        normalized_features_matrix = (features_matrix - mean) / std
        # Convert the normalized matrix back to the original combined_features format
        normalized_features = {name: feature for name, feature in zip(combined_features.keys(), normalized_features_matrix)}
        return normalized_features
    
    def compare_combined_features(self, base_features):
        scores = {}
        for image_name, features in self.combined_features.items():
            # Apply the weights in the distance calculation
            weighted_diff = self.weights * (base_features - features)
            score = np.sum(np.linalg.norm(weighted_diff))
            scores[image_name] = score
        sorted_images = sorted(scores.keys(), key=lambda x: scores[x])
        return sorted_images


    def intensity_search(self):
        histograms = {}
        bins = [i for i in range(0, 256, 10)]
        for image_name in self.images:
            image_path = os.path.join(self.folder_path, image_name)
            image = cv2.imread(image_path)
            image_intensity = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
            hist, _ = np.histogram(image_intensity, bins=bins)
            hist = hist / hist.sum()
            histograms[image_name] = hist
        return histograms

    def color_code_search(self):
        color_codes = {}
        for image_name in self.images:
            image_path = os.path.join(self.folder_path, image_name)
            image = cv2.imread(image_path)
            r_code = (image[:,:,2] >> 6) & 3
            g_code = (image[:,:,1] >> 6) & 3
            b_code = (image[:,:,0] >> 6) & 3
            color_code_6bit = (r_code << 4) | (g_code << 2) | b_code
            hist, _ = np.histogram(color_code_6bit, bins=64, range=(0, 64))
            color_codes[image_name] = hist
        return color_codes

    def manhattan_distance(self, hist1, hist2):
        return np.sum(np.abs(hist1 - hist2))

    def manhattan_distance_color_code(self, cc1, cc2):
        return sum(abs(a - b) for a, b in zip(cc1, cc2))

    def compare_histograms(self, base_hist):
        scores = {}
        for image_name, hist in self.histograms.items():
            score = self.manhattan_distance(base_hist, hist)
            scores[image_name] = score
        sorted_images = sorted(scores.keys(), key=lambda x: scores[x])
        return sorted_images

    def compare_color_codes(self, base_code):
        scores = {}
        for image_name, color_code in self.color_codes.items():
            score = self.manhattan_distance_color_code(base_code, color_code)
            scores[image_name] = score
        sorted_images = sorted(scores.keys(), key=lambda x: scores[x])
        return sorted_images

    def setup_frames(self):
        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.selected_image_label = tk.Label(self.left_frame)
        self.selected_image_label.pack(fill="both", expand=True)

        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.canvas = tk.Canvas(self.right_frame)
        self.canvas.pack(side="top", fill="both", expand=True)

        self.prev_button = tk.Button(self.right_frame, text="Previous", command=self.prev_page)
        self.prev_button.pack(side="left")

        self.next_button = tk.Button(self.right_frame, text="Next", command=self.next_page)
        self.next_button.pack(side="right")

        self.search_button_intensity = tk.Button(self.right_frame, text="Intensity Search", command=self.intensity_mode)
        self.search_button_intensity.pack(side="top")

        self.search_button_color = tk.Button(self.right_frame, text="Color Code Search", command=self.color_code_mode)
        self.search_button_color.pack(side="top")

        self.search_button_rf = tk.Button(self.right_frame, text="Relevance Feedback Search", command=self.rf_mode)
        self.search_button_rf.pack(side="top")

         # add submit button for relevance feedback mode
        self.submit_button = tk.Button(self.right_frame, text="Submit Feedback", command=self.submit_feedback)
        self.submit_button.pack(side="bottom", pady=10)
        self.submit_button.pack_forget()  # Hide the submit button initially

    def display_images(self):
        start_idx = self.current_page * self.images_per_page
        end_idx = start_idx + self.images_per_page
        display_images = self.images[start_idx:end_idx]
        if self.current_page == 0:
            self.prev_button.config(state=tk.DISABLED)
        else:
            self.prev_button.config(state=tk.NORMAL)
        if end_idx >= len(self.images):
            self.next_button.config(state=tk.DISABLED)
        else:
            self.next_button.config(state=tk.NORMAL)

        if self.mode == 'relevance_feedback':
            self.submit_button.pack(side="bottom", pady=10)
        else:
            self.submit_button.pack_forget()

        self.clear_buttons()  # Clear any existing buttons

        # Loop through the rows and columns
        for row in range(self.ROW_COUNT):
            for col in range(self.COLUMN_COUNT):
                index = row * self.COLUMN_COUNT + col
                if start_idx + index < len(self.images):  # Make sure we have an image to display
                    img_name = display_images[index]
                    img_path = os.path.join(self.folder_path, img_name)

                    img = Image.open(img_path)
                    img.thumbnail((100, 100))
                    photo = ImageTk.PhotoImage(img)

                    button = tk.Button(self.canvas, image=photo, command=lambda p=img_path: self.on_image_click(p))
                    button.image = photo
                    button.grid(row=row * 2, column=col)  # Place button in the correct grid position

                    if self.mode == 'relevance_feedback':
                        rel_var = tk.BooleanVar(value=self.relevance.get(img_name, False))
                        checkbox = tk.Checkbutton(self.canvas, text="Relevant", variable=rel_var)
                        checkbox.grid(row=row * 2 + 1, column=col)  # Place checkbox below the button
                        self.checkboxes.append(checkbox)  # Append the checkbox to the list for tracking
                        rel_var.trace_add('write', lambda *args, img_name=img_name, var=rel_var: self.handle_relevance_feedback(img_name, var.get()))
                        self.rel_vars[img_name] = rel_var

                    self.buttons.append(button)

    def on_image_click(self, img_path):

        # Uncheck all checkboxes and reset relevance when a new image is clicked
        if self.mode == 'relevance_feedback':
            # Reset the relevance feedback to non-relevant
            for img_name in self.relevance.keys():
                self.relevance[img_name] = 0
            # Reset all BooleanVar associated with checkboxes to False
            for var in self.rel_vars.values():
                var.set(False)

        if img_path:
            img_name = os.path.basename(img_path)
        else:
            img_name = self.images[0]
        if self.mode == 'relevance_feedback':
            base_features = self.combined_features[img_name]
            similar_images = self.compare_combined_features(base_features)
        elif self.mode == 'color_code':
            base_code = self.color_codes[img_name]
            similar_images = self.compare_color_codes(base_code)
        else:
            base_hist = self.histograms[img_name]
            similar_images = self.compare_histograms(base_hist)
        self.images = similar_images
        self.current_page = 0
        self.images = similar_images
        self.current_page = 0
        self.clear_buttons()
        self.display_images()
        if img_path:
            img = Image.open(img_path)
            img.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(img)
            self.selected_image_label.configure(image=photo)
            self.selected_image_label.image = photo

    def next_page(self):
        if (self.current_page + 1) * self.images_per_page < len(self.images):
            self.clear_buttons()
            self.current_page += 1
            self.display_images()

    def prev_page(self):
        if self.current_page > 0:
            self.clear_buttons()
            self.current_page -= 1
            self.display_images()

    def clear_buttons(self):
        for btn in self.buttons:
            btn.destroy()
        self.buttons.clear()

        # Destroy the checkboxes if they exist
        if hasattr(self, 'checkboxes'):
            for checkbox in self.checkboxes:
                checkbox.destroy()
            self.checkboxes.clear()  # Clear the list of checkbox widgets

    def intensity_mode(self):
        self.mode = 'intensity'
        self.clear_buttons()  # This will clear both buttons and checkboxes
        self.on_image_click(None)

    def color_code_mode(self):
        self.mode = 'color_code'
        self.clear_buttons()  # This will clear both buttons and checkboxes
        self.on_image_click(None)

    def rf_mode(self):
        self.mode = 'relevance_feedback'
        # Get combined features from the combined search
        combined_features = self.combined_search()
        # Normalize the features
        self.combined_features = self.normalize_features(combined_features)
        # Ensure the number of weights matches the number of features
        num_features = next(iter(self.combined_features.values())).shape[0]
        self.weights = np.ones(num_features)  # Initialize weights, no bias
        self.on_image_click(None)


    def handle_relevance_feedback(self, img_name, is_relevant):
        # Update the relevance based on the checkbox state
        self.relevance[img_name] = 1 if is_relevant == 1 else 0

        print(f"Relevance for {img_name}: {self.relevance[img_name]}")  # For debugging purposes


    def submit_feedback(self):
        # Collect feedback from checkboxes and update relevance
        for img_name, var in self.rel_vars.items():
            self.relevance[img_name] = 1 if var.get() == 1 else 0

        # Get the relevant images
        relevant_images = [img_name for img_name, is_relevant in self.relevance.items() if is_relevant == 1]

        # Calculate the mean and standard deviation for each feature of the relevant images
        relevant_features = np.array([self.combined_features[img_name] for img_name in relevant_images])
        mean = np.mean(relevant_features, axis=0)
        std = np.std(relevant_features, axis=0)

        # Set the standard deviation to be 0.5 times the minimum of the non-zero standard deviations if it is 0
        non_zero_std = std[std != 0]
        min_non_zero_std = 0.5 * np.min(non_zero_std) if non_zero_std.size > 0 else 1
        std[std == 0] = min_non_zero_std

        # Update the weights based on the standard deviation and mean
        for i in range(len(mean)):
            if std[i] == 0 and mean[i] != 0:
                self.weights[i] = 1 / min_non_zero_std  # Inverse of std deviation
            elif mean[i] == 0:
                self.weights[i] = 0
            else:
                self.weights[i] = 1 / std[i]  # Inverse of std deviation

        # Recalculate the scores with the new weights
        if self.mode == 'relevance_feedback':
            # Here we need to use a base feature that is the query or some aggregation of relevant images
            # Since it's not provided in the code, let's use the mean of relevant features as a simple proxy
            base_features = mean if relevant_images else np.zeros_like(self.weights)
            self.images = self.compare_combined_features(base_features)

        # Finally, refresh the image display
        self.current_page = 0
        self.clear_buttons()
        self.display_images()



def main():
    root = tk.Tk()
    root.geometry("1000x600")
    # name of folder that holds the images
    folder_path = "images"

    # checks if the image folder exists
    if os.path.exists(folder_path):
        app = ImageBrowser(root, folder_path)
        root.mainloop()
    else:
        print("Images folder does not exist")

if __name__ == '__main__':
    main()