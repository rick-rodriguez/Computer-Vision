tic;
infoTestFaces = dir('Data/test_face_photos/*.JPG');

posHist = read_double_image('Data/positives.bin');
negHist = read_double_image('Data/negatives.bin');

testingFacesName = zeros(length(infoTestFaces),1);
predictedLabels = zeros(example_number, 1);

faces_num = [2, 2, 1, 2, 2, 3, 1, 1, 2, 2, 4, 5, 4, 2, 2, 2, 2, 1, 2, 7 ];

for picture_num = 1:20
    
name = [data_directory, 'test_face_photos/', infoTestFaces(picture_num).name];

coloredImage = imread(name);

binaryImage = (detect_skin(coloredImage, posHist, negHist) >.7);

image = read_gray(name);

[rows,cols] = size(image);

for x = 1:rows
    for y = 1: cols
        if binaryImage (x,y) == 0 
            image (x,y) = 0;
        end
    end
end

boxes = boosted_detector(image, 1, boosted_classifier, ...
                                weak_classifiers, [100, 100], faces_num(picture_num));                        
               
result = coloredImage;

%isItAFace

result_number = faces_num(picture_num);

for number = 1:result_number
    result = draw_rectangle1(result, boxes(number, 1), boxes(number, 2), ...
                             boxes(number, 3), boxes(number, 4));
end                                          


%probablityOfFace = zeros(rows,cols);

%for i = 1:length(infoNonFaces) 
%myImg = coloredImage;
%[myImgX, myImgY, myImgZ] = size(myImg);
%for x = 1:(rows-100)
 %   for y = 1:(cols-100)
%randnum_x = random_number(1, myImgX-100);
 % randnum_y = random_number(1, myImgY-100);
%    window = myImg(x:x+99, y:y+99);
%end
%    classification1 = cascade_classify(window, c1,weak_classifiers); 
%    if classification1 == 1
 %      probablityOfFace(x,y) = 1;
 %      classification2 = cascade_classify(window, c2,weak_classifiers); 
 %      if classification2 == 1
 %         classification3 = cascade_classify(window, c3,weak_classifiers);
  %        if classification3 == 1
  %          classification4 = cascade_classify(window, c4,weak_classifiers); 
 %           if classification4 == 1
  %              classification5 = cascade_classify(window, c5,weak_classifiers);
  %              if classification5 == 1
  %                  probablityOfFace(x,y) = 1;
  %              end 
  %          end
  %        end 
  %     end
  %  end
  % end
%end


figure(picture_num);
imshow(result,[]);

end
toc;
