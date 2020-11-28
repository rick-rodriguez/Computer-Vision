addpath GitHub/Lectures/Code/00_common/00_utilities/
addpath GitHub/Lectures/Code/00_common/00_utilities/spread_figures/
addpath GitHub/Lectures/Code/00_common/00_images/
addpath GitHub/Lectures/Code/00_common/00_detection/
addpath GitHub/Lectures/Code/11_boosting/
addpath GitHub/CS_FINAL/Computer-Vision/
addpath GitHub/CS_FINAL/Computer-Vision/Code/
addpath GitHub/CS_FINAL/Computer-Vision/Data/
addpath GitHub/CS_FINAL/Computer-Vision/Data/test_cropped_faces/
addpath GitHub/CS_FINAL/Computer-Vision/Data/test_face_photos/
addpath GitHub/CS_FINAL/Computer-Vision/Data/test_nonfaces/
addpath GitHub/CS_FINAL/Computer-Vision/Data/training_faces/
addpath GitHub/CS_FINAL/Computer-Vision/Data/training_nonfaces/
addpath GitHub/CS_FINAL/possible_code/



infoTestFaces=dir('GitHub/CS_FINAL/Computer-Vision/Data/test_face_photos/*.JPG');

testingFacesName = zeros(length(infoTestFaces),1);

name = infoTestFaces(5).name;

image= imread(name);

imshow (image); 

tic; 

result = boosted_detector(image, 1, boosted_classifier, ...
                                weak_classifiers, [100, 100], 1); toc


disp ("imdone");