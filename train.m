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





infoFaces = dir('GitHub/CS_FINAL/Computer-Vision/Data/training_faces/');
infoNonFaces = dir('GitHub/CS_FINAL/Computer-Vision/Data/training_nonfaces/');

face_vertical = 100;
face_horizontal = 100;

number = 1000;
weak_classifiers = cell(1, number);
for i = 1:number
    weak_classifiers{i} = generate_classifier(face_vertical, face_horizontal);
end


example_number = size(infoFaces, 1) + size(infoNonFaces, 1);
labels = zeros(example_number, 1);
labels (1:size(infoFaces,1)) = 1;
labels((size(infoFaces, 1)+1):example_number) = -1;
examples = zeros(face_vertical, face_horizontal, example_number);
examples (:, :, 1:size(infoFaces,1)) = face_integrals;
examples(:, :, (size(infoFaces,1)+1):example_number) = nonface_integrals;

classifier_number = numel(weak_classifiers);

responses =  zeros(classifier_number, example_number);

for example = 1:example_number
    integral = examples(:, :, example);
    for feature = 1:classifier_number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
    disp(example)
end

















%%
negative_histogram = read_double_image('negatives.bin');
positive_histogram = read_double_image('positives.bin');


photo = imread("obama8.JPG");
imshow(photo);



newPhoto = detect_skin(photo, positive_histogram, negative_histogram);

figure(2);
imshow(newPhoto > .90);

%%

info = dir('GitHub/CS_FINAL/Computer-Vision/Data/training_faces/');

index = 749;

title = info(index,1);

title.name


labels = zeros(1000);


for x = (1:500) 
    labels (1:500, 1) = 1; 
    randomIndex= randi([1 3050]);
    title = info(index,1);
    fileName=title.name;
    workingFace = imread(filename);
    b = integral_image(workingFace);
    
    
    % choosing a set of random weak classifiers

    number = 1000;
    weak_classifiers = cell(1, number);
    for i = 1:number
    weak_classifiers{i} = generate_classifier(face_vertical, face_horizontal);
    end
    
    

end




%%
for x = (500:100)
    labels (1, 501:1000) = -1; 
    randomIndex= rand;
    responces(x) = picAtRandomIndexfromnonFaces;
    % create lables for some training examples for nonfaces
    % labels should be -1
end


result = AdaBoost (responces,labels,15);