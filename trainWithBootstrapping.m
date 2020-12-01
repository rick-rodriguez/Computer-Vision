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

TrainingFaces = zeros(face_vertical, face_horizontal, length(infoFaces));
TrainingNonfaces = zeros(face_vertical, face_horizontal, length(infoNonfaces));

faces = zeros(face_vertical, face_horizontal, length(infoFaces));
nonfaces = zeros(face_vertical, face_horizontal, length(infoNonfaces));

face_vertical = 100;
face_horizontal = 100;

tic;

% Here we are creating a variable called training example so that when we
% loopp we do not create it each time

for i = 1:length(infoFaces)
TrainingFaces(:, :, i) = imread(infoFaces(i).name);
end

for i = 1:length(infoNonFaces) 
myImg = imread(infoNonFaces(i).name);
[myImgX, myImgY] = size(myImg);
randnum_x = random_number(1, myImgX-100);
randnum_y = random_number(1, myImgY-100);
TrainingNonfaces(:, :, i) = myImg(randnum_x:randnum_x+99, randnum_y:randnum_y+99);
end


training_example_number = size(TrainingFaces, 3) + size(TrainingNonfaces, 3); % create a place to store the photos
training_labels = zeros(training_example_number, 1); % place for labels
training_labels (1:size(Trainingfaces, 3)) = 1; % assign 1 to yes faces
training_labels((size(Trainingfaces, 3)+1):training_example_number) = -1; % assign -1 to nonfaces
training_examples = zeros(face_vertical, face_horizontal, training_example_number); % a place for the photos themselves
start_here = 1;
for i = 1: size(TrainingFaces, 3)
    training_examples(:,:,i) = TrainingFaces(:, :, i);
    start_here = start_here + 1;
end

y = 1;

for x = start_here:training_example_number
    training_examples(:,:,i) = TrainingNonfaces (:,:,y);
    y = y + 1;
end


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

next_ID_faces = 501;
next_ID_nonfaces = 101;


%%
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

%%%%%%set up for next round


% call the detector here to classify all the photos and find the mistakes
%3.Apply the detector to all training images.
% for each test use Rick's code to test if it is a face or not a face
% if the correct label and the guessed label mulitplied = -1
% replace correct exaples with negative examples


weights = ones(example_number, 1) / example_number;
% next line takes about 8.5 seconds.
[index error threshold] = find_best_classifier(responses, labels, weights);

% if a mistake was made add this back into the training set and re-train
% (loop again)

%if example # ----- rendered a -1 (aka labels 1 * -1 = 1)
for x = 1:training_example_numer % look at every test image
    if (%if example # ----- rendered a -1 (aka labels 1 * -1 = 1))
         % add this to training set
        if  correct ID == 1
            faces (:,:,next_ID_faces) = training_examples (:,:,x);
            next_ID_faces = next_ID_faces + 1;
        end
        if correct ID == -1
            nonfaces(:, :,next_ID_nonfaces) = training_examples(:,:,x);
            next_ID_nonfaces = next_ID_nonfaces + 1;
        end
    end
end


% do this x number of times until it is inefficienet/less accurate
% we will set this threshold
%weighted_error(responses, labels, weights, classifier);

 
disp ("imdone");

