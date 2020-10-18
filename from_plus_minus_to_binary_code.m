function [binary_code_labels] = from_plus_minus_to_binary_code(plus_minus_labels, indices_known)
%FROM_PLUS_MINUS_TO_BINARY_CODE Summary of this function goes here
%   Detailed explanation goes here

plus_minus_labels = plus_minus_labels(indices_known);

K = length(plus_minus_labels);

binary_code_labels = zeros(K,2);

for i = 1:K
    
    if plus_minus_labels(i) > 0
        
       binary_code_labels(i,1) = 1; 
       
    else
        
       binary_code_labels(i,2) = 1; 
        
    end
    
end

end

