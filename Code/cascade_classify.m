function [result] = cascade_classify(window,classifier, weak_classifiers)


result = boosted_multiscale_search(window, 1, classifier, weak_classifiers, [100, 100]);

result = (result > .4);
end

