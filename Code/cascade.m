function [new_classifier] = cascade(face_vertical, face_horizontal, ...
                                infoFaces, infoNonFaces, faces, nonfaces, ...
                                classifier_count)
%CASCADE

weak_classifiers = cell(1, classifier_count);
weak_classifiers{1} = boosted_classifier;
for i = 2:classifier_count
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

new_classifier = AdaBoost(responses, labels, 15);

end

