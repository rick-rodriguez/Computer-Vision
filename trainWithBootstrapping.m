restoredefaultpath;
clear all;
close all;

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


%Bootstrapping
%1.(Initialization:) Choose some training examples. Not too few, not too many.
%2.Train a detector.
%3.Apply the detector to all training images.
%4.Identify mistakes.
%5.Add mistakes to training examples.
%6.If needed, remove some examples to make room.
%Go to step 2, unless performance has stopped improving.

infoFaces = dir('GitHub/CS_FINAL/Computer-Vision/Data/training_faces/*.bmp');
infoNonFaces = dir('GitHub/CS_FINAL/Computer-Vision/Data/training_nonfaces/*.jpg');

face_vertical = 100;
face_horizontal = 100;

faces = zeros(face_vertical, face_horizontal, 500);
nonfaces = zeros(face_vertical, face_horizontal, 100);

face_vertical = 100;
face_horizontal = 100;

tic;
%1.(Initialization:) Choose some training examples. Not too few, not too many.

for i = 1:500
  rand = random_number(1, length(infoFaces));   
  faces(:, :, i) = imread(infoFaces(rand).name);
end

for i = 1:100
    rand = random_number(1, length(infoNonFaces));  
    myImg = imread(infoNonFaces(rand).name);
    [myImgX, myImgY] = size(myImg);
    randnum_x = random_number(1, myImgX-100);
    randnum_y = random_number(1, myImgY-100);
    nonfaces(:, :, i) = myImg(randnum_x:randnum_x+99, randnum_y:randnum_y+99);
end

%2.Train a detector.
% loop back to here to retrain the detector
number = 1000;
weak_classifiers = cell(1, number);
for i = 1:number
weak_classifiers{i} = generate_classifier(face_vertical, face_horizontal);
end

face_integrals = zeros(face_vertical, face_horizontal, length(infoFaces));
for i = 1:size(infoFaces)
face_integrals(:, :, i) = integral_image(faces(:, :, i));
end

nonface_integrals = zeros(face_vertical, face_horizontal, length(infoNonFaces));
for i = 1:size(infoNonFaces)
nonface_integrals(:, :, i) = integral_image(nonfaces(:, :, i));
end


example_number = size(faces, 3) + size(nonfaces, 3);
labels = zeros(example_number, 1);
labels (1:size(faces, 3)) = 1;
labels((size(faces, 3)+1):example_number) = -1;
examples = zeros(face_vertical, face_horizontal, example_number);
examples (:, :, 1:size(faces, 3)) = face_integrals;
examples(:, :, (size(faces, 3)+1):example_number) = nonface_integrals;

classifier_number = numel(weak_classifiers);

responses =  zeros(classifier_number, example_number);

for example = 1:example_number
    integral = examples(:, :, example);
    for feature = 1:classifier_number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
    %disp(example);
end


boosted_classifier = AdaBoost(responses, labels, 15);

% call the detector here to classify all the photos and find the mistakes
% if a mistake was made add this back into the training set and re-train
% do this x number of times until it is inefficienet/less accurate

toc;
disp ("imdone");
