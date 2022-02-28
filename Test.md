# ARTFM
Supporting clinical staff with mixed reality applications has been investigated more and more in recent years. Especially in highly complex surgeries mixed reality can be an assistive application.  Implants or prostheses used in orthopedic surgery are often assembled during surgery. Through our ARTFM application, the appropriate parts for the next assembly step, their location on the tray, and the instruction steps to follow are visualized to the scrub nurse via a mixed reality head mounted display.

# Clinical background
As many surgical instruments look alike even experienced staff can pick the wrong one in the first place which extends the assembly time of the surgical instrument overall.  Furthermore, for complex assembly steps a paper manual is considered from time to time in the operating room (OR). 

# State-of-the-art
Augmented reality (AR) has proved itself to be a valuable asset in the OR [5]. AR and head mounted displays (HMDs) such as the HoloLens are a good fit for instructional content [6]. The effectiveness of hands-free operation and ease of use of spatially well placed data was especially noted for the HoloLens [6]. 
Challenges were mainly founded in the missing object detection and location capabilities of the HoloLens. AR guided assembly is evaluated in [4] with the use of building blocks. This approach uses a voxel-based approach and compares the current building block with the final digital 3D model. Depending on the discrepancy between the builded model and the digital 3D model the next step is displayed. As instruction mode a partial-wireframe and a side-by-side mode is considered.


For surgical surgical instrument detection there is already a great research interest in detecting them while they are used in laparoscopic
surgery [1].

Besides the great success in various computer vision tasks with object detection and semantic segmentation there is often the problem of missing or insufficient data for training and evaluating neural network approaches. Therefore, synthetic data sets developed and proved to be successful [2, 3].


While the most prominent usage with HMDs in medical settings is training and overlaid information while operating, we plan to use the device to provide spatial highlights, contextual information and assembly instructions. Interviewing experts shows that especially for implants there are not so common steps that the scrub nurse is using
everyday. Therefore, our approach ARTFM helps the scrub nurse to select and assemble the required surgical instruments correctly. Also, for scrub nurses which are quite new in their field, it could help them orientate on up to four different surgical instrument tables with over 1000 surgical instruments.

We present 
- i) a synthetically generated data set 
- ii) an object detection model trained on our synthethic dataset
- ii) an augmented reality manual, guiding the user to the correct assembly part and providing assembly instructions.

# Methodology
## Synthetic Data Set!
3D models of surgical instruments for assembly tasks are often not  publicly available we prototype or idea using LEGO Bricks. The usage of these building blocks has already proved to be successful in literature [4]. As no real world LEGO data set for our object detection tasks exist, we used BlenderProc [2] to synthetically generate
our data set consisting of 44 classes with the most common bricks, see Figure . The bricks are randomly selected and randomly placed on a ground plate. Further, the material for the background and for the LEGO bricks is set randomly based on a predefined color set. For training and testing 5000 images were generated and split 80:20.

## Object Detection
A well-known task in computer vision is object detection. In our approach Yolo5s  is applied and trained on our synthetic data set.  For training Yolo5s a learning rate of 0.001 was chosen using Adam optimizer with a weight decay set to 0.0001. the batch size is set to 16. The network was trained on a RTX 3070 with 8GB VRAM. As  mean Average Precision (mAP) we report 84% on our eval split. Using a REST API and Flask our Kinect Azure can be called, constantly monitoring the instrument table, see Fig. 1b.

## Augmented Reality Manual
As in various Mixed Reality (MR) applications a HoloLens is used  in our approach too. Specifically HoloLens 2. To guide the user through assembly steps the HoloLens 2 communicates with the server where the object detection model runs. To achieve a matching of the individual camera coordinate systems from Kinect Azure and our moving HoloLens 2 a Vuforia marker is used.



# Current project status
We trained our object detection model and currently evalute and test it on real world data. Furthermore, the augmented reality manual is currently implemented using Micrsofts Mixed Reality Toolkit. 


# REFERENCES
[1] B. Choi, K. Jo, S. Choi, and J. Choi. Surgical-tools detection based on convolutional neural network in laparoscopic robot-assisted surgery. In 2017 39th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), pp. 1756–1759. doi: 10. 1109/EMBC.2017.8037183
[2] M. Denninger, M. Sundermeyer, D. Winkelbauer, Y. Zidan, D. Olefir,M. Elbadrawy, A. Lodhi, and H. Katam. Blenderproc, 2019.
[3] T. Hodan, V. Vineet, R. Gal, E. Shalev, J. Hanzelka, T. Connell, P. Urbina, S. Sinha, and B. Guenter. Photorealistic image synthesis for object instance detection.
[4] B. M. Khuong, K. Kiyokawa, A. Miller, J. J. La Viola, T. Mashita, and H. Takemura. The effectiveness of an AR-based context-aware assembly support system in object assembly. In 2014 IEEE Virtual Reality (VR), pp. 57–62. doi: 10.1109/VR.2014.6802051
[5] S. Park, S. Bokijonov, and Y. Choi. Review of microsoft HoloLens applications over the past five years. 11(16):7259. Number: 16 Publisher: Multidisciplinary Digital Publishing Institute. doi: 10.3390/app11167259
[6] R. Radkowski and J. Ingebrand. HoloLens for assembly assistance - a focus group report. In S. Lackey and J. Chen, eds., Virtual, Augmented and Mixed Reality, Lecture Notes in Computer Science, pp. 274–282. Springer International Publishing. doi: 10.1007/978-3-319-57987-0 22
