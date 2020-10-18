function [plus_minus_labels] = from_binary_code_to_plus_minus(binary_code_labels, indices_known, class_train_test)
%FROM_BINARY_CODE_TO_PLUS_MINUS Summary of this function goes here
%   Detailed explanation goes here

K = length(binary_code_labels);

plus_minus_labels_temp = zeros(K,1);

for i = 1:K
    
    if binary_code_labels(i,1) > binary_code_labels(i,2)
        
       plus_minus_labels_temp(i) = max(class_train_test); 
       
    else
        
       plus_minus_labels_temp(i) = min(class_train_test);
        
    end
    
end

plus_minus_labels = zeros(length(indices_known),1);

plus_minus_labels(indices_known) = class_train_test(indices_known);

plus_minus_labels(~indices_known) = plus_minus_labels_temp;

end

