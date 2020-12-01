infoTestFaces=dir('Data\test_face_photos\*.JPG');

posHist = read_double_image('Data\positives.bin');
negHist = read_double_image('Data\negatives.bin');

testingFacesName = zeros(length(infoTestFaces),1);

name = ['Data\test_face_photos\', infoTestFaces(18).name];

coloredImage = imread(name);

figure(2);
imshow(coloredImage,[]);

binaryImage = (detect_skin(coloredImage, posHist, negHist) >.8);

figure(1);
imshow (binaryImage > .8);

image= read_gray(name);

[rows,cols] = size(image);

for x = 1:rows
    for y = 1: cols
        if binaryImage (x,y) == 0 
            image (x,y) = 0;
        end
    end
end

figure(4);
imshow(image,[]);


boxes = boosted_detector(image, 1, boosted_classifier, ...
                                weak_classifiers, [100, 100], 1);                        
               
result = image;


result_number = 1;

for number = 1:result_number
    result = draw_rectangle1(result, boxes(number, 1), boxes(number, 2), ...
                             boxes(number, 3), boxes(number, 4));
end                                          


figure(3);
imshow(result,[]);

disp ("imdone");