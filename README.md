The Image Browser Application is a GUI-based tool designed to help users view and search through a collection of images using different criteria. Developed with Python's tkinter for the graphical interface, 
PIL for image handling, and opencv for image processing, this application provides functionalities for browsing images, performing searches based on image intensity or color codes, and applying relevance feedback to improve search results.

To use the application, first ensure that you have the required packages installed. You can do this by running pip install pillow numpy opencv-python. Prepare a directory with images in JPG format, typically named images, 
or any directory of your choice, and adjust the folder_path variable in the main function if necessary. Once your image folder is set up, execute the script to launch the application.

The user interface features buttons for navigating through image pages, searching by intensity, color code, or relevance feedback. 
In relevance feedback mode, users can check boxes next to images they find relevant and click "Submit Feedback" to refine search results based on their input. The application supports pagination, 
allowing users to view up to 20 images per page and navigate between pages using "Previous" and "Next" buttons.

The core functionality of the application is encapsulated in the ImageBrowser class. This class handles initialization, GUI setup, and image display, as well as various image processing tasks. 
Key methods include those for calculating combined features from intensity and color code histograms, normalizing features for comparison, and comparing images based on these features. 
The intensity_search and color_code_search methods compute histograms for intensity and 6-bit color codes, respectively, while the handle_relevance_feedback and submit_feedback methods manage user input in relevance feedback mode.

The main function sets up the main application window and starts the GUI event loop, checking for the existence of the images folder before proceeding. 
The application is designed to be user-friendly and provides robust functionality for image search and browsing, making it a valuable tool for managing and exploring image collections.
