# Interview Task: Real-Time Object Tracking Simulation

### Objective:

Develop a Python script that simulates real-time object tracking using input from a simulated video feed, __luxonis_task_video.mp4__. 

The task will involve processing a series of images (frames) to identify and track the movement of circles and rectangles of various colors across
these frames.
The output will include the coordinates of the object in each frame and a simple visualization showing the tracked path of the object over time.

## Task Details:

### Object Detection:

Utilize the __opencv-python__ library to create an algorithm that detects circles and rectangles in each frame.
The detection algorithm should identify the coordinates of the object within the frame.

### Object Tracking:

Once the object is detected, the script must track its movement across the frames, maintaining a history of its positions. 
A 2.5% chance exists for an object to disappear in one frame, only to reappear in the following frame. 
Your tracker should be able to handle this scenario and track the object correctly.

### Output Visualization:

Generate a final output that visually represents the path of the object across all frames. 
The visualization tool should be simple and easy to understand, allowing easy recognition of whether all objects were correctly tracked. 
Any detection or tracking mistakes should be easily identifiable.

### Documentation:

Provide a minimal documentation which lets other people understand, how to:
- use the object detector(s)
- use the tracker
- run the app

### Evaluation Criteria:

- Code Quality: Clear, readable, and well-organized code.
- Algorithm Efficiency: Efficient algorithms for detecting, and tracking the object.
- Accuracy: The ability to accurately track the object across all frames, as demonstrated by the output visualization tool.
- Problem-Solving Skills: Creativity and effectiveness in solving the given problem within the constraints.

### Submission

- Zip your complete solution and send it in reply to your conversation with the Luxonis hiring manager. 
