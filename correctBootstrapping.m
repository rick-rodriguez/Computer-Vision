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

% training add a boost needs to stop and round n, n is found by trial and
% error
% use rectangle filters as weight classifiers
% make skin a weight classifier as well
%%%% IMPORTANT: use 1 and -1 for labels to facilitate the adaboost
%%%% algorihmim
%This makes sence bc if guessed id and correct id are equal it will always 
% be positive
%load in the faces and nonfaces in same order
% pass those on to the adaboost function for about 15 rounds
% this is trial and fail number, he started with this (15)

tic;

infoFaces = dir('GitHub/CS_FINAL/Computer-Vision/Data/training_faces/*.bmp');
infoNonFaces = dir('GitHub/CS_FINAL/Computer-Vision/Data/training_nonfaces/*.JPG');

disp ("Istart");

face_vertical = 100;
face_horizontal = 100;

TrainingFaces = zeros(face_vertical, face_horizontal, length(infoFaces));
TrainingNonfaces = zeros(face_vertical, face_horizontal, length(infoNonFaces));

faces = zeros(face_vertical, face_horizontal, length(infoFaces));
nonfaces = zeros(face_vertical, face_horizontal, length(infoNonFaces));

next_ID_faces = 501;
next_ID_nonfaces = 101;

for i = 1:length(infoFaces)
TrainingFaces(:, :, i) = imread(infoFaces(i).name);
end

for i = 1:length(infoNonFaces) 
myImg = imread(infoNonFaces(i).name);
[myImgX, myImgY, myImgZ] = size(myImg);
randnum_x = random_number(1, myImgX-100);
randnum_y = random_number(1, myImgY-100);
TrainingNonfaces(:, :, i) = myImg(randnum_x:randnum_x+99, randnum_y:randnum_y+99);
end


training_example_number = size(TrainingFaces, 3) + size(TrainingNonfaces, 3); % create a place to store the photos
training_labels = zeros(training_example_number, 1); % place for labels
training_labels (1:size(TrainingFaces, 3)) = 1; % assign 1 to yes faces
training_labels((size(TrainingFaces, 3)+1):training_example_number) = -1; % assign -1 to nonfaces
training_examples = zeros(face_vertical, face_horizontal, training_example_number); % a place for the photos themselves
start_here = 1;
for i = 1: size(TrainingFaces, 3)
    training_examples(:,:,i) = TrainingFaces(:, :, i);
    start_here = start_here + 1;
end

y = 1;

for x = start_here:training_example_number
    training_examples(:,:,x) = TrainingNonfaces (:,:,y);
    y = y + 1;
end

for i = 1:500
    img = ['GitHub/CS_FINAL/Computer-Vision/Data/training_faces/', infoFaces(i).name];
    faces(:, :, i) = imread(img);
end

for i = 1:100
    img = ['GitHub/CS_FINAL/Computer-Vision/Data/training_nonfaces/', infoNonFaces(i).name];
    img = imread(img);
    [myImgX, myImgY, myImgZ] = size(img);
    randnum_x = random_number(1, myImgX-100);
    randnum_y = random_number(1, myImgY-100);
    nonfaces(:, :, i) = img(randnum_x:randnum_x+99, randnum_y:randnum_y+99);
end


classifier_count = 300;
weak_classifiers = cell(1, classifier_count);
for i = 1:classifier_count
    weak_classifiers{i} = generate_classifier(face_vertical, face_horizontal);
end


accuracy = 100;
previous_accuracy = 0;
improvement = accuracy - previous_accuracy;

while (improvement > .05 || previous_accuracy == 100)   
previous_accuracy = accuracy;
face_integrals = zeros(face_vertical, face_horizontal, length(faces));
for i = 1:size(faces)
    face_integrals(:, :, i) = integral_image(faces(:, :, i));
end

nonface_integrals = zeros(face_vertical, face_horizontal, length(nonfaces));
for i = 1:size(nonfaces)
    nonface_integrals(:, :, i) = integral_image(nonfaces(:, :, i));
end
    

example_number = size(faces, 3) + size(nonfaces, 3);
labels = zeros(example_number, 1);
labels(1:size(faces, 3)) = 1;
labels((size(faces, 3)+1):example_number) = -1;
examples = zeros(face_vertical, face_horizontal, example_number);
examples(:, :, 1:size(faces, 3)) = face_integrals;
examples(:, :, (size(faces, 3)+1):example_number) = nonface_integrals;

classifier_number = numel(weak_classifiers);

responses = zeros(classifier_number, example_number);

for example = 1:example_number
    integral = examples(:, :, example);
    for feature = 1:classifier_number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
    %disp(example);
end


%%%%%%set up for next round


% call the detector here to classify all the photos and find the mistakes
%3.Apply the detector to all training images.

weights = ones(example_number, 1) / example_number;

[index, error, threshold, alpha] = find_best_classifier(responses, labels, weights);

index

threshold


for i = 1:example_number
    if abs(responses(index, i)) > abs(threshold)
        predictedLabels(i,1) = -1;
    end
    if abs(responses(index, i)) < abs(threshold)
        predictedLabels(i,1) = 1;
    end
end

mistakes = 0;

for count = 1:size(training_examples, 3)
    correct = labels(count,1);
    guessed = predictedLabels(count,1);
    determine = correct * guessed;
    if determine == -1
        mistakes = mistakes + 1;
        if  correct == 1
            faces (:,:,next_ID_faces) = training_examples (:,:,count);
            next_ID_faces = next_ID_faces + 1;
        end
        if correct == -1
            nonfaces(:, :,next_ID_nonfaces) = training_examples(:,:,count);
            next_ID_nonfaces = next_ID_nonfaces + 1;  
        end         
    end
end

correct = example_number - mistakes;

accuracy = (correct/example_number) * 100;

accuracy

improvement = accuracy - previous_accuracy;

if (improvement > .05 || previous_accuracy == 100)
    boosted_classifier = AdaBoost(responses, labels, 15);
end


% if a mistake was made add this back into the training set and re-train
% (loop again)


% do this x number of times until it is inefficienet/less accurate
% we will set this threshold
%weighted_error(responses, labels, weights, classifier);
end

toc;
disp ("imdone");
