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


infoFaces = dir('GitHub/CS_FINAL/Computer-Vision/Data/training_faces/*.bmp');
infoNonFaces = dir('GitHub/CS_FINAL/Computer-Vision/Data/training_nonfaces/*.JPG');

face_vertical = 100;
face_horizontal = 100;

faces = zeros(face_vertical, face_horizontal, length(infoFaces));
nonfaces = zeros(face_vertical, face_horizontal, length(infoNonFaces));

tic;

for i = 1:length(infoFaces)
    img = ['GitHub/CS_FINAL/Computer-Vision/Data/training_faces/', infoFaces(i).name];
    faces(:, :, i) = imread(img);
end

for i = 1:length(infoNonFaces) 
    img = ['GitHub/CS_FINAL/Computer-Vision/Data/training_nonfaces/', infoNonFaces(i).name];
    img = imread(img);
    [myImgX, myImgY, myImgZ] = size(img);
    randnum_x = random_number(1, myImgX-100);
    randnum_y = random_number(1, myImgY-100);
    nonfaces(:, :, i) = img(randnum_x:randnum_x+99, randnum_y:randnum_y+99);
end

classifier_count = 100;
weak_classifiers = cell(1, classifier_count);
for i = 1:classifier_count
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

<<<<<<< Updated upstream
responses = zeros(classifier_number, example_number);
=======
responses =  zeros(classifier_number, example_number);
responsesFACE =  zeros(1, example_number);

>>>>>>> Stashed changes

for example = 1:example_number
    integral = examples(:, :, example);
    for feature = 1:classifier_number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
    %disp(example);
end



boosted_classifier = AdaBoost(responses, labels, 15);

<<<<<<< Updated upstream
weights = ones(example_number, 1) / example_number;

[index, error, threshold, alpha] = find_best_classifier(responses, labels, weights);

for i = 1:example_number
    if responses(index, i) > threshold
        predictedLabels(i,1) = -1;
    end
    if responses(index, i) < threshold
        predictedLabels(i,1) = 1;
    end
end

mistakes = 0;

for count = 1:size(examples, 3)
    correct = labels(count,1);
    guessed = predictedLabels(count,1);
    determine = correct * guessed;
    if determine == -1
        mistakes = mistakes + 1;
    end
end

correct = example_number - mistakes;

accuracy = (correct/example_number) * 100;

accuracy

c1 = cascade(face_vertical, face_horizontal, infoFaces, infoNonFaces,...
            faces, nonfaces, 3);      
c2 = cascade(face_vertical, face_horizontal, infoFaces, infoNonFaces,...
            faces, nonfaces, 7);
c3 = cascade(face_vertical, face_horizontal, infoFaces, infoNonFaces,...
            faces, nonfaces, 10);
c4 = cascade(face_vertical, face_horizontal, infoFaces, infoNonFaces,...
            faces, nonfaces, 25);
c5 = cascade(face_vertical, face_horizontal, infoFaces, infoNonFaces,...
            faces, nonfaces, 50);
=======



for example = 1:example_number
    integral = examples(:, :, example);
    responsesFACE(1, example) = eval_weak_classifier(classifier, integral);
end

>>>>>>> Stashed changes
toc;
disp ("imdone");
